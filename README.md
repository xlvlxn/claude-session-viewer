# Claude Session Viewer

A static web page that renders a Claude Code session `.jsonl` file as a chat transcript — similar to the UI in the Claude app / Claude desktop.

**Hosted:** https://xlvlxn.github.io/claude-session-viewer/

Drop your `.jsonl` file into the page. Everything runs client-side; the file never leaves your browser.

## Where are my sessions?

Claude Code writes session logs to:

```
~/.claude/projects/<project-slug>/<session-id>.jsonl
```

## Features

- User / assistant turns rendered in a Claude-app style chat feed.
- Thinking blocks are collapsible.
- Tool calls rendered with per-tool formatting:
  - `Write` — file header + preview, **View full content** opens a syntax-highlighted modal.
  - `Edit` / `NotebookEdit` — old/new blocks with a diff-style view in the modal.
  - `Read` — file_path / offset / limit as structured fields.
  - `Bash` — command in a shell code block.
  - `Grep` / `Glob` / `WebFetch` / `WebSearch` / `Agent` / `TodoWrite` — tool-specific layouts.
- Long tool results are truncated with a **View full** button.
- Full-text search with match highlighting and navigation.
- Per-turn **copy** button and a global **Copy all** that produces Markdown.
- Toggles to hide thinking or tool blocks for a cleaner read.
- Light / dark mode follows your OS.
- Optional `?url=…` query to load a remote `.jsonl` directly (CORS-permitting).

## Local offline-first HTML build

If you'd rather ship a single self-contained HTML with the session embedded, use the included Python generator:

```
python3 build_viewer.py path/to/session.jsonl -o viewer.html
open viewer.html
```
