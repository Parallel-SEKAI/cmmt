# cmmt

[![PyPI version](https://badge.fury.io/py/cmmt.svg)](https://badge.fury.io/py/cmmt)

`cmmt` 是一个由 AI 驱动的命令行工具，旨在帮助您自动生成符合 [Conventional Commits](https://www.conventionalcommits.org/) 规范的 Git 提交消息和分支名称。

## 核心功能

- **AI 驱动**: 利用 OpenAI 的强大语言模型，根据您的代码变更 (`git diff`) 生成高质量的提交消息。
- **遵循规范**: 严格遵循 Conventional Commits 规范，提升您代码库的可读性和可维护性。
- **分支命名**: 自动建议清晰、规范的分支名称。
- **高度可配置**: 您可以通过 `~/.cmmt.yml` 文件自定义模型、提示语、忽略文件等。
- **上下文感知**: 能够获取 `git status`, `git diff`, `git log` 以及项目文件结构作为上下文，生成更精准的提交信息。
- **交互式与非交互式**: 支持交互式确认，也支持 `-y` 参数进行自动化操作。

## 安装部署

您可以通过 `pip` 来安装 `cmmt`:

```bash
pip install cmmt
```

或者，如果您使用 `uv`:

```bash
uv pip install cmmt
```

安装完成后，您需要初始化配置文件并设置您的 OpenAI API 密钥：

```bash
cmmt --init
```

该命令会引导您在 `~/.cmmt.yml` 文件中设置 API 密钥、模型、以及其他可选配置。

## 使用方法

在您的 Git 仓库中，当您已经将变更添加到暂存区后，运行以下命令：

```bash
git add .
cmmt
```

`cmmt` 将会分析您的变更，并生成建议的提交消息。

### 常用选项

- **生成分支名称**:
  ```bash
  cmmt -b
  ```
- **自动确认并提交**:
  ```bash
  cmmt -y
  ```
- **提交后自动推送到远程仓库**:
  ```bash
  cmmt -p
  ```
- **组合使用**:
  ```bash
  cmmt -b -y -p
  ```
- **提供额外信息**:
  您可以通过 `-e` 参数为 AI 提供额外的上下文信息。
  ```bash
  cmmt -e "This change is part of the new authentication feature."
  ```

## 配置文件

`cmmt` 的所有配置都存储在 `~/.cmmt.yml` 文件中。您可以通过 `cmmt --init` 命令来初始化该文件，也可以手动编辑它。

以下是一个配置文件的示例：

```yaml
openai_api_key: "sk-..."
model: "gpt-4o"
base_url: null
max_tokens: 2048
ignore_files:
  - "package-lock.json"
  - "*.log"
extra_info: "Always respond in Chinese."
git_log_level: "brief" # Can be 'none', 'brief', 'detailed'
git_log_count: 10 # -1 for all
project_structure_enabled: true
project_structure_max_depth: 3
project_structure_ignore:
  - "node_modules/"
force_think: false
```

## 许可协议

该项目基于 [MIT License](LICENSE) 开源。
