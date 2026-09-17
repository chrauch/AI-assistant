# AIA, your local AI assistant

Artificial Intelligence Assistant (AIA) is a modular, local-first assistant
for the terminal. It lets you have conversations, build persistent per-context
memories, work with files and web content, and use reusable instructions while
keeping your data and model runtime under your control. Its model-agnostic
design allows you to change the underlying language model without changing the
assistant's core workflows.

Copyright (C) 2026 Christian Rauch. Distributed under the terms of the
[GNU General Public License version 3](LICENSE).

With AIA, you can:

- Have persistent conversations with local AI models.
- Choose the language model that best suits your needs.
- Set agent instructions and active tools per conversation.
- Save and recall memories for each conversation context.
- Stream responses and let the model call enabled tools when needed.
- Read files, list directories, fetch public webpages, and manage memories.
- Revise files into timestamped generated copies without changing the originals.
- Use AIA interactively or pass commands and prompts directly from the shell.
- Define reusable command scripts in `commands/`.


## Installation

AIA requires Python 3.14 or newer. Install Python using your operating
system's package manager or from [python.org](https://www.python.org/), then
verify the version:

```bash
python --version
```

From the AIA project directory, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell, use:

```powershell
.venv\Scripts\Activate.ps1
```

Install AIA and its Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Start AIA from the project directory:

```bash
python -m aia
```

If the virtual environment is not activated, use its Python executable
explicitly:

```bash
./.venv/bin/python -m aia
```

The compatibility launcher can be run the same way:

```bash
./.venv/bin/python aia_cli.py
```


On first start, the model manager downloads the model configured in
`config.json` into `models/` if it is not already present. Model downloads can
be large, and local model execution can require substantial memory. The
runtime uses CUDA when it is available and otherwise falls back to the CPU.


## Configuration

Choose the model and runtime paths in `config.json`. Network access is normally
needed only during the initial setup (download from https://huggingface.co/).

Filesystem tools are restricted to `data/` by default. To permit absolute
paths outside that directory, set `allow_external_files` to `true`.

Relative paths continue to resolve inside `data/`; only absolute external
paths are enabled by this setting. Enable it only when the assistant should be
allowed to read or create generated files elsewhere. Paths beginning with `~`
are expanded to the current user's home directory.


## Interactive Commands

Type `/exit` or `//` to leave interactive mode. Common commands include:

| Command | Function |
| --- | --- |
| `/help` or `/?` | List built-in and user-defined commands. |
| `/new [INSTRUCTION]` | Create and activate a new conversation context. |
| `/load` | List saved conversation contexts. |
| `/load ID` | Activate a context by number or identifier. |
| `/resume ID` | Alias for `/load ID`. |
| `/continue ID` | Alias for `/load ID`. |
| `/agent` | Show the active agent instruction. |
| `/agent INSTRUCTION` | Replace the active agent instruction. |
| `/clear` | Delete the active context and create a new one. |
| `/clean` | Delete inactive contexts without a response. |
| `/delete ID` | Delete an inactive context. |
| `/truncate [ID]` | Delete saved messages from a context; defaults to the active context. |
| `/history` | Print the latest saved context snapshot. |
| `/compact [INSTRUCTION]` | Summarize and compact the active context. |
| `/file` | List files appended to the context. |
| `/file + FILE` | Read and append a file to the context. |
| `/file - FILE` | Remove appended content for a file. |
| `/file -` | Remove appended content for all files. |
| `/tool` | List available tools and their active state. |
| `/tool + NAME` | Activate one tool. |
| `/tool - NAME` | Deactivate one tool. |
| `/tool +` | Activate all tools. |
| `/tool -` | Deactivate all tools. |
| `/memory` | List memory tools and their active state. |
| `/memory +` | Activate all memory tools. |
| `/memory -` | Deactivate all memory tools. |
| `/revise FILE [INSTRUCTION]` | Generate a timestamped revised copy and show its changes. |
| `/system` | Show runtime, model, device, and assistant information. |
| `/exit` or `//` | Exit interactive mode. |
| `Ctrl+C` | Interrupt the current model operation. |

Commands are also accepted in non-interactive mode. Each quoted argument is
executed as one instruction.

```bash
./.venv/bin/python -m aia "Translate to French: I'm just here for the snacks."
```

## Reusable Command Scripts

Each file in `commands/` defines a command whose name is the filename stem.
Each file contains a JSON list of strings. The strings are executed in order as
if they had been entered interactively.

For example, `commands/ensp.json` defines `/ensp`:

```text
[
  "/agent Act as a bilingual English-Spanish language assistant."
]
```

Run it interactively with:

```text
/ensp
```

Command scripts can contain other commands and ordinary prompts. Nested command
files are supported, and recursive command-file loops are rejected.


## Tools

Tools are discovered automatically from concrete subclasses of `Tool` in
`aia/tools/`. Each tool declares a name, description, and parameter schema. Active
tools are exposed to the model for ordinary responses and can be changed per
conversation context.

The current tool families include:

- Time lookup.
- Directory listing.
- UTF-8 file reading.
- Timestamped AI-generated file writing.
- Public webpage extraction.
- Per-context memory storage, lookup, listing, appending, and deletion.

File and directory tools restrict access to `data/`. The webpage tool accepts
HTTP and HTTPS URLs and rejects local or private network destinations.


## File Revision

`/revise FILE` reads the source file in the program, sends its content to the
model without exposing tool schemas, and expects the model to return only the
complete revised content. The program then uses the normal timestamped writer
to create a generated copy, leaving the source untouched.

When the Unix `diff` executable is available, the assistant compares the
original and generated files with a unified diff using one surrounding line of
context. The output uses readable `Original` and `Revised` labels.

An optional instruction is appended to the default revision requirements:

```text
/revise paper.tex Use American English and preserve all LaTeX commands.
```

## Persistence

Contexts are stored under `contexts/<context-id>/`. Each context has a JSON
conversation snapshot and a `settings.json` file containing its agent
instruction and active tools. Memory entries are stored in a separate
per-context `memory/` directory.

The context store writes snapshots through temporary files and preserves
backups and compacted snapshots where appropriate. Context deletion removes
the context's nested memory directory as well.


## Project Structure

| Path | Purpose |
| --- | --- |
| `aia/cli/main.py` | Terminal interface, command parsing, command scripts, and output formatting. |
| `aia/services/ai_assistant.py` | Public assistant facade that coordinates the services. |
| `aia/services/` | Conversation, file, and tool workflows. |
| `aia/domain/` | Core in-memory conversation models. |
| `aia/infrastructure/` | Configuration, persistence, logging, model loading, and model runtime. |
| `aia/tools/` | Auto-discovered concrete tool implementations. |
| `aia_cli.py` | Compatibility launcher for running AIA directly from the project directory. |
| `pyproject.toml` | Package metadata, dependencies, and the `aia` command definition. |
| `config.json` | User-editable runtime configuration, including the model identifier, storage paths, and file-access policy. |
| `commands/` | JSON command scripts containing ordered lists of instructions. |
| `data/` | Data directory used by file and directory tools. |
| `contexts/` | Persisted conversation state, settings, and per-context memories. |
| `models/` | Local model files. |

`AIAssistant` is intentionally kept as the public facade used by the terminal
interface. The implementation is divided into focused services: context state
and settings are handled by `ContextManager`, model-facing conversation work by
`ConversationWorkflow`, and file operations by `FileWorkflow`.


## License

Copyright (C) 2026 Christian Rauch.
AIA is distributed under the [GNU General Public License version 3](LICENSE).
