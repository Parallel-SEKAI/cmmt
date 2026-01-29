#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import yaml
import tiktoken
import re
from openai import OpenAI

VERSION = "v0.1.1"
CONFIG_PATH = os.path.expanduser("~/.cmmt.yml")
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
        return subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], 
            text=True
        )
    except subprocess.CalledProcessError:
        print("Error: Not a git repository.")
        return None

def get_git_diff(config):
    try:
        command = ["git", "diff", "--staged"]
        if config.get("ignore_files"):
            for file in config["ignore_files"]:
                command.extend(["--", f":(exclude){file}"])
        diff = subprocess.check_output(command, text=True)
        # 简单截断防止 Token 溢出 (保留前 10000 字符)
        return diff[:10000] + "\n...(diff truncated)" if len(diff) > 10000 else diff
    except subprocess.CalledProcessError:
        return ""

def get_git_log(config: dict) -> str:
    log_level = config.get("git_log_level", "brief")
    if log_level == "none": return ""
    log_count = config.get("git_log_count", 5)
    try:
        cmd = ["git", "log", f"-n{log_count}"]
        if log_level == "brief": cmd.append("--oneline")
        return subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError:
        return ""

def get_project_structure(config: dict) -> str:
    """利用 Git 原生能力获取项目树结构"""
    if not config.get("project_structure_enabled", True): return ""
    max_depth = config.get("project_structure_max_depth", 3)
    try:
        files = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"], 
            text=True
        ).splitlines()
        
        tree = {}
        for f in files:
            parts = f.split(os.sep)
            if max_depth != -1 and len(parts) > max_depth: continue
            curr = tree
            for part in parts: curr = curr.setdefault(part, {})

        def render(node, indent=""):
            lines = []
            items = sorted(node.items())
            for i, (name, children) in enumerate(items):
                is_last = (i == len(items) - 1)
                connector = "└── " if is_last else "├── "
                lines.append(f"{indent}{connector}{name}")
                if children:
                    lines.extend(render(children, indent + ("    " if is_last else "│   ")))
            return lines
        return "\n".join(render(tree))
    except Exception:
        return ""

def build_prompt(status, diff, git_log, project_structure, args, config):
    prompt = f"""# Task
Generate a Commit Message (Conventional Commits) based on git status and diff.
{"Generate a Branch Name (type/desc) if requested." if args.branch else ""}

# Specification
- Message: <type>(<scope>): <subject> (max 50 chars, present tense)
- Body/Footer: Optional, for details or breaking changes.
- Branch: type/short-description (lowercase, hyphenated)

# Output Format
JSON ONLY:
{{
    "commit_message": "...",
    "branch_name": "..."
}}
"""
    if config.get("force_think"):
        prompt += "\n# Important\nWrap your reasoning in <tool_call> tags before the JSON.\n"
    
    extra = (config.get("extra_info", "") + "\n" + (args.extra_info or "")).strip()
    if extra: prompt += f"\n# Extra Info\n{extra}\n"

    prompt += f"\n# Context\n## Status\n{status}\n## Diff\n{diff}\n"
    if git_log: prompt += f"## Recent Logs\n{git_log}\n"
    if project_structure: prompt += f"## Structure\n{project_structure}\n"
    return prompt

def parse_response(result: str, args):
    # 提取思考过程
    think_match = re.search(r"<tool_call>(.*?)</tool_call>", result, re.DOTALL)
    if think_match:
        print(f"\n--- AI Thinking ---\n{think_match.group(1).strip()}\n")

    # 提取 JSON
    json_match = re.search(r"\{.*\}", result, re.DOTALL)
    if not json_match:
        print("Error: No JSON found."); return None, None
    
    try:
        data = json.loads(json_match.group(0))
        return data.get("commit_message"), data.get("branch_name")
    except json.JSONDecodeError:
        print("Error: JSON parse failed."); return None, None

def execute_git_commands(commit_msg, branch_name, args):
    if branch_name:
        try:
            # 检查分支是否存在，存在则切换，不存在则创建
            res = subprocess.run(["git", "rev-parse", "--verify", branch_name], capture_output=True)
            cmd = ["git", "checkout", branch_name] if res.returncode == 0 else ["git", "checkout", "-b", branch_name]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("Abort: Branch switch failed."); return False

    try:
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        print("Successfully committed.")
    except subprocess.CalledProcessError:
        print("Error: Commit failed."); return False

    if args.push:
        try:
            curr_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
            subprocess.run(["git", "push", "-u", "origin", curr_branch], check=True)
            print(f"Successfully pushed to origin/{curr_branch}.")
        except subprocess.CalledProcessError:
            print("Error: Push failed.")
    return True

def main():
    parser = argparse.ArgumentParser(prog="cmmt")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("-p", "--push", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("-b", "--branch", action="store_true")
    parser.add_argument("-e", "--extra-info")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    config = load_config()
    if args.init:
        config.update({
            "openai_api_key": input("API Key: "),
            "model": input(f"Model [{DEFAULT_MODEL}]: ") or DEFAULT_MODEL,
            "base_url": input("Base URL (optional): ") or None,
            "force_think": input("Force Think? (y/n) [n]: ").lower() == "y",
            "project_structure_enabled": True,
            "project_structure_max_depth": 3
        })
        save_config(config); return

    if not config.get("openai_api_key"):
        print("Run --init first."); return

    client = OpenAI(api_key=config["openai_api_key"], base_url=config.get("base_url"))
    
    status = get_git_status()
    if not status: return
    
    diff = get_git_diff(config)
    prompt = build_prompt(status, diff, get_git_log(config), get_project_structure(config), args, config)

    if args.output:
        with open(args.output, "w") as f: f.write(prompt)

    # Token Count
    try:
        enc = tiktoken.encoding_for_model(config.get("model", DEFAULT_MODEL))
    except:
        enc = tiktoken.get_encoding("cl100k_base")
    print(f"Prompt Tokens: {len(enc.encode(prompt))}")

    if not args.yes and input("Generate? (y/n): ").lower() != "y": return

    res = client.chat.completions.create(
        model=config.get("model", DEFAULT_MODEL),
        messages=[{"role": "user", "content": prompt}]
    )
    
    commit_msg, branch_name = parse_response(res.choices[0].message.content, args)
    if not commit_msg: return

    print(f"\nProposed Commit: {commit_msg}")
    if branch_name: print(f"Proposed Branch: {branch_name}")

    if not args.yes and input("\nExecute? (y/n): ").lower() != "y": return
    execute_git_commands(commit_msg, branch_name, args)

if __name__ == "__main__":
    main()