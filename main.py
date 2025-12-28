import argparse
import json
import os
import subprocess
import yaml
import tiktoken
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
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)


def get_git_status():
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
    try:
        command = ["git", "diff", "--staged"]
        if config.get("ignore_files"):
            for file in config["ignore_files"]:
                command.extend(["--", f":(exclude){file}"])
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return ""


def build_prompt(status, diff, args, config):
    format_str = '{"commit_message": "..."'
    if args.branch:
        format_str += ', "branch_name": "..."'
    format_str += "}"

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

- Implement OAuth 2.0 login flow using Google and Facebook.
- Update user model to store OAuth tokens.

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

    prompt += (
        """# Important
You must strictly follow the specifications without any deviations.
# Output Format
You can only output content similar to the following: """
        + format_str
        + "\n"
    )

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
    return prompt


def call_openai(client, prompt, config):
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


def parse_response(result, args):
    # Clean the result: remove markdown code blocks if present
    result = result.strip()
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
        print(f"AI Response: {result}")
        print("Failed to parse JSON response.")
        return None, None


def execute_git_commands(commit_msg, branch_name, args):
    # Execute git commit
    try:
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        print("Committed.")
    except subprocess.CalledProcessError as e:
        print(f"Git commit failed: {e}")
        return False

    # Execute git checkout -b if branch
    if branch_name:
        try:
            subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            print(f"Checked out to new branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            print(f"Git checkout failed: {e}")
            return False

    # Push if --push
    if args.push:
        try:
            subprocess.run(
                ["git", "push", "-u", "origin", branch_name or "main"], check=True
            )
            print("Pushed.")
        except subprocess.CalledProcessError as e:
            print(f"Git push failed: {e}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(prog="cmmt", description="AI Git Helper")
    parser.add_argument("-v", "--version", action="version", version=f"cmmt {VERSION}")
    parser.add_argument("--init", action="store_true", help="Initialize config file")
    parser.add_argument("-p", "--push", action="store_true", help="Post commit push")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto confirm")
    parser.add_argument(
        "-b", "--branch", action="store_true", help="Suggest branch name"
    )
    parser.add_argument("-e", "--extra-info", help="Extra information for prompt")
    parser.add_argument("-o", "--output", help="Output prompt to file")
    args = parser.parse_args()

    if args.init:
        config = load_config()
        api_key = input("Enter OpenAI API key: ")
        model = input("Enter model (default: gpt-3.5-turbo): ") or DEFAULT_MODEL
        base_url = input("Enter base URL (optional): ") or None
        max_tokens = input("Enter max tokens (optional): ") or 0
        config["openai_api_key"] = api_key
        config["model"] = model
        if base_url:
            config["base_url"] = base_url
        config["max_tokens"] = int(max_tokens)
        config["ignore_files"] = []
        config["extra_info"] = ""
        save_config(config)
        print("Config initialized.")
        return

    config = load_config()
    api_key = config.get("openai_api_key")
    if not api_key:
        print("API key not found. Run --init first.")
        return

    base_url = config.get("base_url")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    # Get git status and diff
    status = get_git_status()
    if status is None:
        return
    diff = get_git_diff(config)

    # Build prompt
    prompt = build_prompt(status, diff, args, config)

    # Output prompt to file if specified
    if args.output:
        with open(args.output, "w") as f:
            f.write(prompt)
        print(f"Prompt written to {args.output}")

    # Calculate token count
    model = config.get("model", DEFAULT_MODEL)
    try:
        enc = tiktoken.encoding_for_model(model)
        token_count = len(enc.encode(prompt))
        print(f"Prompt token count: {token_count}")
    except KeyError:
        print("Warning: Unknown model for token counting.")
        print("Prompt length (chars):", len(prompt))

    # Ask user if to generate
    if not args.yes:
        confirm = input("Generate commit message? (y/n): ")
        if confirm.lower() != "y":
            return

    # Call OpenAI API
    result = call_openai(client, prompt, config)
    if result is None:
        return

    # Parse JSON
    commit_msg, branch_name = parse_response(result, args)
    if commit_msg is None:
        return

    # Output result
    print("Generated:")
    if branch_name:
        print(f"Branch name: {branch_name}")
    print(f"Commit message: {commit_msg}")

    # Ask user if to execute
    if not args.yes:
        confirm = input("Execute git commit and checkout? (y/n): ")
        if confirm.lower() != "y":
            return

    # Execute git commands
    execute_git_commands(commit_msg, branch_name, args)


if __name__ == "__main__":
    main()
