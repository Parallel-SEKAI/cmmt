import argparse
import json
import os
import subprocess
import yaml
import tiktoken
import fnmatch
from openai import OpenAI

VERSION = "v0.1.0"
CONFIG_PATH = os.path.expanduser("~/.cmmt.yml")

# Constants for commit types and branch types
COMMIT_TYPES = [
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
]
BRANCH_TYPES = ["feat", "fix", "docs", "style", "refactor", "perf", "test", "chore"]

# Default model
DEFAULT_MODEL = "gpt-3.5-turbo"


def load_config():
    """Load YAML configuration file."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config):
    """Save configuration to YAML file."""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)


def get_git_status():
    """Get output of `git status`."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        print("Not in a git repository or git status failed.")
        return None


def get_git_diff(config):
    """Get output of `git diff --staged`, optionally ignoring specified files."""
    try:
        command = ["git", "diff", "--staged"]
        if config.get("ignore_files"):
            for file in config["ignore_files"]:
                command.extend(["--", f":(exclude){file}"])
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return ""


def get_git_log(config: dict) -> str:
    """
    Get recent git logs based on configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Git log output, or empty string if disabled or failed.
    """
    log_level = config.get("git_log_level", "brief")
    if log_level == "none":
        return ""

    log_count = config.get("git_log_count", 5)

    try:
        command = ["git", "log"]
        if log_count != -1:
            command.append(f"-n{log_count}")

        if log_level == "brief":
            command.append("--oneline")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        print("Failed to get git log.")
        return ""


def _get_ignored_patterns(root_dir: str) -> list[str]:
    """Read ignore rules from the project's .gitignore file."""
    ignore_file_path = os.path.join(root_dir, ".gitignore")
    patterns = []
    if os.path.exists(ignore_file_path):
        with open(ignore_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    return patterns


def get_project_structure(config: dict) -> str:
    """
    Generate project file structure tree string.
    
    Ignores .git, .gitignore patterns, and custom patterns defined in config.
    """
    if not config.get("project_structure_enabled", True):
        return ""

    root_dir = os.getcwd()
    max_depth = config.get("project_structure_max_depth", 3)

    default_ignore = [".git/"]
    gitignore_patterns = _get_ignored_patterns(root_dir)
    config_ignore = config.get("project_structure_ignore", [])
    all_ignore_patterns = default_ignore + gitignore_patterns + config_ignore

    structure_lines = []

    def build_tree(dir_path, prefix, current_depth):
        if max_depth != -1 and current_depth >= max_depth:
            return

        rel_path_from_root = os.path.relpath(dir_path, root_dir)

        try:
            entries = sorted(os.listdir(dir_path))
        except OSError:
            return

        def is_ignored(name):
            full_path = os.path.join(dir_path, name)
            is_dir = os.path.isdir(full_path)
            entry_rel_path = (
                os.path.join(rel_path_from_root, name)
                if rel_path_from_root != "."
                else name
            )

            for pattern in all_ignore_patterns:
                p = pattern
                if p.endswith("/"):
                    p = p.rstrip("/")
                    if not is_dir:
                        continue

                if fnmatch.fnmatch(entry_rel_path, p) or fnmatch.fnmatch(name, p):
                    return True
            return False

        filtered_entries = [e for e in entries if not is_ignored(e)]

        for i, entry in enumerate(filtered_entries):
            connector = "└── "
            is_dir = os.path.isdir(os.path.join(dir_path, entry))

            structure_lines.append(f"{prefix}{connector}{entry}{'/' if is_dir else ''}")

            if is_dir:
                build_tree(
                    os.path.join(dir_path, entry), prefix + "    ", current_depth + 1
                )

    build_tree(root_dir, "", 0)
    return "\n".join(structure_lines)


def build_prompt(status, diff, git_log, project_structure, args, config):
    """Build the prompt used for generating the commit message."""
    prompt = """# Task
I will provide you with the output of `git status` and `git diff`. Based on this, generate:
- A **Commit Message** that follows the **Conventional Commits** specification.
"""
    if args.branch:
        prompt += "- A **Branch Name** that follows the specification.\n"

    prompt += """# Requirements
## Commit Message Specification
- **Header**: `<type>(<scope>): <short summary>` (no more than 50 characters)
  - Type: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.
  - Scope: Optional, indicates the module (e.g., auth, ui, parser, gradle).
  - Summary: Starts with a verb in present tense, lowercase first letter, no period at the end, describes the changes in detail.
- **Body**: (if necessary) Detailed explanation of the reason and logic for the changes.
- **Footer**: (if necessary) List Breaking Changes or related Issue IDs.
Example:

```

feat(auth): add login with OAuth 2.0

* Implement OAuth 2.0 login flow using Google and Facebook.
* Update user model to store OAuth tokens.

BREAKING CHANGE: user passwords are no longer stored in the database.
Closes #123

```
"""
    if args.branch:
        prompt += """## Branch Name Specification
- Format: `type/short-description` (e.g., feat/login-api, fix/overflow-issue)
- Use lowercase letters, separate words with hyphens `-`.
- Types: feat, fix, docs, style, refactor, perf, test, chore.
"""

    format_str = """{
    "commit_message": "<type>(<scope>): <subject>\n\n<body>\n\n<footer>",
    "branch_name": "<type>/<scope>/<issue_id>-<short_description>"
}"""
    
    if config.get("force_think"):
        output_format_instruction = f"""# Important
You MUST first output your thinking process within <tool_call> and </tool_call> tags. After the </tool_call> tag, you MUST provide the JSON output.
# Output Format

```

<tool_call>
Your detailed analysis and reasoning for the commit message and branch name go here.
</tool_call>
{format_str}

```
"""
    else:
        output_format_instruction = f"""# Important
You must strictly follow the specifications without any deviations.
# Output Format
You can only output content similar to the following:
{format_str}
"""
    prompt += output_format_instruction
    prompt += "\n"

    if args.extra_info or config.get("extra_info"):
        prompt += "# Extra Information\n"
        if config.get("extra_info"):
            prompt += f"{config.get('extra_info')}\n"
        if args.extra_info:
            prompt += f"{args.extra_info}\n"

    prompt += f"""# Context
## Git status:
{status}
## Git diff:
{diff}
"""
    if git_log:
        prompt += f"""## Recent Commits (from git log):
{git_log}
"""
    if project_structure:
        prompt += f"""## Project Structure:\n{project_structure}\n"""
    return prompt


def call_openai(client, prompt, config):
    """Call OpenAI API to generate commit message."""
    try:
        response = client.chat.completions.create(
            model=config.get("model", DEFAULT_MODEL),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.get("max_tokens") or None,
        )
        result = response.choices[0].message.content.strip()
        usage = response.usage
        if usage:
            print(f"Total tokens: {usage.total_tokens}")
            print(f"Prompt tokens: {usage.prompt_tokens}")
            print(f"Completion tokens: {usage.completion_tokens}")
        return result
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None


def parse_response(result: str, args, config: dict):
    """Parse the response string from AI model."""
    result = result.strip()

    if config.get("force_think") and "</tool_call>" in result:
        try:
            *think, json_part = result.split("</tool_call>")
            think_content = "".join(think).strip()
            think_content = think_content.replace("<tool_call>", "", 1).strip()
            print("\n--- AI Thinking Process ---")
            print(think_content)
            print("--------------------------\n")
            result = json_part.strip()
        except ValueError:
            print("AI response did not follow the expected <tool_call> format.")
            pass

    if result.startswith("```json"):
        result = result[7:]
    if result.endswith("```"):
        result = result[:-3]
    result = result.strip()

    try:
        data = json.loads(result)
        commit_msg = data["commit_message"]
        branch_name = data.get("branch_name") if args.branch else None
        return commit_msg, branch_name
    except json.JSONDecodeError:
        print(f"AI Response (after parsing): {result}")
        print("Failed to parse JSON response.")
        return None, None


def execute_git_commands(commit_msg, branch_name, args):
    """Execute git commands (commit, checkout, push)."""
    try:
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        print("Committed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Git commit failed: {e}")
        return False

    if branch_name:
        try:
            subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            print(f"Switched to new branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            print(f"Git checkout failed: {e}")
            return False

    if args.push:
        try:
            subprocess.run(
                ["git", "push", "-u", "origin", branch_name or "main"], check=True
            )
            print("Pushed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Git push failed: {e}")
            return False

    return True


def main():
    """Main function to handle arguments and workflow."""
    parser = argparse.ArgumentParser(prog="cmmt", description="AI Git Assistant")
    parser.add_argument("-v", "--version", action="version", version=f"cmmt {VERSION}")
    parser.add_argument("--init", action="store_true", help="Initialize configuration file")
    parser.add_argument("-p", "--push", action="store_true", help="Post-commit push")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-confirm all prompts")
    parser.add_argument("-b", "--branch", action="store_true", help="Suggest branch name")
    parser.add_argument("-e", "--extra-info", help="Extra information for prompt")
    parser.add_argument("-o", "--output", help="Output prompt to a file")
    args = parser.parse_args()

    if args.init:
        config = load_config()
        api_key = input("Enter OpenAI API key: ")
        model = input(f"Enter model (default: {DEFAULT_MODEL}): ") or DEFAULT_MODEL
        base_url = input("Enter base URL (optional): ") or None
        max_tokens = input("Enter max tokens (optional): ") or 0
        config["openai_api_key"] = api_key
        config["model"] = model
        if base_url:
            config["base_url"] = base_url
        config["max_tokens"] = int(max_tokens)
        config["ignore_files"] = []
        config["extra_info"] = ""
        
        valid_log_levels = {"n": "none", "b": "brief", "d": "detailed"}
        while True:
            log_level_input = (
                input("Enter git log level (n)one/(b)rief/(d)etailed [Default: b]: ")
                or "b"
            ).lower()

            if log_level_input in valid_log_levels.values():
                config["git_log_level"] = log_level_input
                break
            elif log_level_input in valid_log_levels:
                config["git_log_level"] = valid_log_levels[log_level_input]
                break
            else:
                print("Invalid input. Please enter 'n', 'b', 'd' or the full name.")

        git_log_count = input("Enter git log count (-1 for all, Default: 5): ") or 5
        config["git_log_count"] = int(git_log_count)

        config["project_structure_enabled"] = (
            input("Enable project structure context? (y/n) [Default: y]: ") or "y"
        ).lower() == "y"
        config["project_structure_max_depth"] = int(
            input("Enter project structure max depth (-1 for unlimited, Default: 3): ")
            or 3
        )
        config["project_structure_ignore"] = []

        config["force_think"] = (
            input("Enable forced thinking mode? (y/n) [Default: n]: ") or "n"
        ).lower() == "y"

        save_config(config)
        print("Configuration initialized.")
        return

    config = load_config()
    api_key = config.get("openai_api_key")
    if not api_key:
        print("API key not found. Please run --init first.")
        return

    base_url = config.get("base_url")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    status = get_git_status()
    if status is None:
        return
    diff = get_git_diff(config)
    git_log = get_git_log(config)
    project_structure = get_project_structure(config)

    prompt = build_prompt(status, diff, git_log, project_structure, args, config)

    if args.output:
        with open(args.output, "w") as f:
            f.write(prompt)
        print(f"Prompt written to {args.output}")

    model = config.get("model", DEFAULT_MODEL)
    try:
        enc = tiktoken.encoding_for_model(model)
        token_count = len(enc.encode(prompt))
        print(f"Prompt tokens: {token_count}")
    except KeyError:
        print("Warning: Unknown model, cannot calculate tokens.")
        print("Prompt length (chars):", len(prompt))

    if not args.yes:
        confirm = input("Generate commit message? (y/n): ")
        if confirm.lower() != "y":
            return

    result = call_openai(client, prompt, config)
    if result is None:
        return

    commit_msg, branch_name = parse_response(result, args, config)
    if commit_msg is None:
        return

    print("Generated:")
    if branch_name:
        print(f"Branch Name: {branch_name}")
    print(f"Commit Message: {commit_msg}")

    if not args.yes:
        confirm = input("Execute git commit/checkout? (y/n): ")
        if confirm.lower() != "y":
            return

    execute_git_commands(commit_msg, branch_name, args)


if __name__ == "__main__":
    main()
