#!/usr/bin/env python3
"""Build a self-contained HTML viewer for a Claude Code session JSONL file.

Usage:
    python3 build_viewer.py <session.jsonl> [-o output.html]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SKIP_TYPES = {
    "queue-operation",
    "file-history-snapshot",
    "last-prompt",
    "agent-name",
    "progress",
}

SKIP_SYSTEM_SUBTYPES = {"bridge_status", "turn_duration"}


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def collect_tool_results(records: list[dict]) -> dict[str, dict]:
    """First pass — map tool_use_id -> the result payload."""
    out: dict[str, dict] = {}
    for r in records:
        if r.get("type") != "user":
            continue
        msg = r.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "tool_result":
                continue
            out[part.get("tool_use_id")] = {
                "content": part.get("content"),
                "is_error": bool(part.get("is_error")),
                "timestamp": r.get("timestamp"),
            }
    return out


def build_events(records: list[dict]) -> list[dict]:
    """Second pass — produce a flat list of render-ready events.

    Assistant streaming emits multiple JSONL lines that share message.id.
    We merge those lines back into a single assistant turn with an ordered
    list of parts (text, thinking, tool_use with embedded result).
    """
    results_by_id = collect_tool_results(records)
    events: list[dict] = []
    current_assistant: dict | None = None

    def flush():
        nonlocal current_assistant
        if current_assistant is not None:
            events.append(current_assistant)
            current_assistant = None

    for r in records:
        t = r.get("type")
        if t in SKIP_TYPES:
            continue

        if t == "system":
            sub = r.get("subtype")
            if sub in SKIP_SYSTEM_SUBTYPES:
                continue
            flush()
            if sub == "compact_boundary":
                events.append({"kind": "compact", "timestamp": r.get("timestamp")})
            elif sub == "local_command":
                events.append(
                    {
                        "kind": "system",
                        "content": r.get("content", ""),
                        "timestamp": r.get("timestamp"),
                    }
                )
            continue

        if t == "user":
            msg = r.get("message") or {}
            content = msg.get("content")
            text_parts: list[str] = []
            image_parts: list[dict] = []
            has_plain = False

            if isinstance(content, str):
                text_parts.append(content)
                has_plain = True
            elif isinstance(content, list):
                for p in content:
                    pt = p.get("type")
                    if pt == "text":
                        text_parts.append(p.get("text", ""))
                        has_plain = True
                    elif pt == "image":
                        image_parts.append(
                            {
                                "media_type": (p.get("source") or {}).get("media_type"),
                                "data": (p.get("source") or {}).get("data"),
                            }
                        )
                        has_plain = True
                    # tool_result is handled inline with its tool_use below

            if has_plain:
                flush()
                events.append(
                    {
                        "kind": "user",
                        "text": "\n\n".join(text_parts).strip(),
                        "images": image_parts,
                        "timestamp": r.get("timestamp"),
                    }
                )
            continue

        if t == "assistant":
            msg = r.get("message") or {}
            mid = msg.get("id")
            model = msg.get("model")
            parts = msg.get("content") or []

            if current_assistant is None or current_assistant.get("id") != mid:
                flush()
                current_assistant = {
                    "kind": "assistant",
                    "id": mid,
                    "model": model,
                    "timestamp": r.get("timestamp"),
                    "parts": [],
                }

            for p in parts:
                pt = p.get("type")
                if pt == "text":
                    text = p.get("text", "")
                    if text:
                        current_assistant["parts"].append({"type": "text", "text": text})
                elif pt == "thinking":
                    th = p.get("thinking", "")
                    if th.strip():
                        current_assistant["parts"].append({"type": "thinking", "text": th})
                elif pt == "tool_use":
                    tid = p.get("id")
                    result = results_by_id.get(tid)
                    current_assistant["parts"].append(
                        {
                            "type": "tool_use",
                            "id": tid,
                            "name": p.get("name"),
                            "input": p.get("input"),
                            "result": _normalize_result(result),
                        }
                    )
            continue

    flush()
    return events


def _normalize_result(result: dict | None) -> dict | None:
    if result is None:
        return None
    content = result.get("content")
    texts: list[str] = []
    images: list[dict] = []
    tool_refs: list[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for p in content:
            pt = p.get("type")
            if pt == "text":
                texts.append(p.get("text", ""))
            elif pt == "image":
                src = p.get("source") or {}
                images.append({"media_type": src.get("media_type"), "data": src.get("data")})
            elif pt == "tool_reference":
                tool_refs.append(p.get("tool_name", ""))
    return {
        "text": "\n\n".join(texts),
        "images": images,
        "tool_refs": tool_refs,
        "is_error": result.get("is_error", False),
        "timestamp": result.get("timestamp"),
    }


def extract_meta(records: list[dict]) -> dict:
    session_id = None
    cwd = None
    title = None
    first_model = None
    models: set[str] = set()
    version = None
    started = None
    ended = None
    for r in records:
        if session_id is None:
            session_id = r.get("sessionId")
        if cwd is None and r.get("cwd"):
            cwd = r.get("cwd")
        if version is None and r.get("version"):
            version = r.get("version")
        if r.get("type") == "custom-title" and title is None:
            title = r.get("customTitle")
        ts = r.get("timestamp")
        if ts:
            if started is None:
                started = ts
            ended = ts
        if r.get("type") == "assistant":
            m = (r.get("message") or {}).get("model")
            if m and m != "<synthetic>":
                models.add(m)
                if first_model is None:
                    first_model = m
    return {
        "session_id": session_id,
        "cwd": cwd,
        "title": title,
        "model": first_model,
        "models": sorted(models),
        "version": version,
        "started": started,
        "ended": ended,
    }


# ---------------------------------------------------------------- HTML template

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.0/dist/purify.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github.min.css" />
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/common.min.js"></script>
<style>
  :root {
    --bg: #faf9f7;
    --panel: #ffffff;
    --border: #e8e5de;
    --border-strong: #d6d2c9;
    --text: #1f1e1c;
    --muted: #6b6860;
    --user-bubble: #efece4;
    --accent: #c96442;
    --accent-soft: #f5e9e2;
    --tool-bg: #f4f1ea;
    --tool-border: #d8d3c7;
    --tool-accent: #8a6b4a;
    --thinking-bg: #f6f3ec;
    --error: #b8433b;
    --code-bg: #f4f1ea;
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --font-mono: "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1a1a18;
      --panel: #242421;
      --border: #36342f;
      --border-strong: #45423c;
      --text: #ece9e2;
      --muted: #a09d93;
      --user-bubble: #2e2c27;
      --accent: #d97757;
      --accent-soft: #3a2b23;
      --tool-bg: #2a2824;
      --tool-border: #403d36;
      --tool-accent: #c89775;
      --thinking-bg: #24221e;
      --error: #e07a73;
      --code-bg: #1f1d1a;
    }
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--font-sans); }
  body { line-height: 1.55; font-size: 15px; }
  a { color: var(--accent); }
  header.topbar {
    position: sticky; top: 0; z-index: 10;
    background: var(--panel); border-bottom: 1px solid var(--border);
    padding: 10px 20px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
  }
  header.topbar h1 { font-size: 15px; font-weight: 600; margin: 0; }
  header.topbar .meta { font-size: 12px; color: var(--muted); display: flex; gap: 12px; flex-wrap: wrap; }
  header.topbar .meta code { font-family: var(--font-mono); font-size: 11px; }
  .controls { margin-left: auto; display: flex; gap: 8px; align-items: center; }
  .controls input[type="search"] {
    padding: 6px 10px; border: 1px solid var(--border-strong); border-radius: 6px;
    background: var(--bg); color: var(--text); font-size: 13px; width: 220px;
  }
  .controls button {
    padding: 6px 10px; border: 1px solid var(--border-strong); border-radius: 6px;
    background: var(--panel); color: var(--text); font-size: 13px; cursor: pointer;
  }
  .controls button:hover { background: var(--user-bubble); }
  .controls label { font-size: 12px; color: var(--muted); display: flex; gap: 4px; align-items: center; cursor: pointer; }

  main.feed { max-width: 820px; margin: 0 auto; padding: 24px 20px 120px; }
  .turn { margin-bottom: 24px; position: relative; }
  .turn.user { display: flex; justify-content: flex-end; }
  .bubble {
    background: var(--user-bubble); border-radius: 14px; padding: 10px 14px;
    max-width: 85%; white-space: pre-wrap; word-wrap: break-word;
  }
  .turn.user .bubble { border-bottom-right-radius: 4px; }
  .turn.assistant .content { padding: 0; }
  .turn .ts { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .turn.user .ts { text-align: right; }

  .md { line-height: 1.6; }
  .md p { margin: 0 0 0.6em; }
  .md p:last-child { margin-bottom: 0; }
  .md pre {
    background: var(--code-bg); padding: 10px 12px; border-radius: 8px;
    overflow-x: auto; font-family: var(--font-mono); font-size: 13px;
    border: 1px solid var(--border);
  }
  .md code { font-family: var(--font-mono); font-size: 0.92em; background: var(--code-bg); padding: 1px 5px; border-radius: 4px; }
  .md pre code { padding: 0; background: transparent; border-radius: 0; }
  .md ul, .md ol { margin: 0.4em 0; padding-left: 1.6em; }
  .md h1, .md h2, .md h3, .md h4 { margin: 1em 0 0.4em; line-height: 1.3; }
  .md blockquote { margin: 0.5em 0; padding: 4px 12px; border-left: 3px solid var(--border-strong); color: var(--muted); }
  .md table { border-collapse: collapse; margin: 0.5em 0; }
  .md th, .md td { border: 1px solid var(--border-strong); padding: 4px 8px; }

  .thinking {
    margin: 8px 0; border: 1px dashed var(--border-strong); border-radius: 8px;
    background: var(--thinking-bg);
  }
  .thinking > summary {
    list-style: none; padding: 6px 12px; cursor: pointer; font-size: 12px;
    color: var(--muted); user-select: none;
  }
  .thinking > summary::before { content: "▸ "; display: inline-block; width: 1em; }
  .thinking[open] > summary::before { content: "▾ "; }
  .thinking .body { padding: 4px 14px 10px; font-size: 13px; color: var(--muted); white-space: pre-wrap; font-style: italic; }

  .tool {
    margin: 10px 0; border: 1px solid var(--tool-border); border-radius: 10px;
    background: var(--tool-bg); overflow: hidden;
  }
  .tool > summary {
    list-style: none; padding: 8px 12px; cursor: pointer; font-size: 13px;
    display: flex; gap: 8px; align-items: center; user-select: none;
  }
  .tool > summary::before { content: "▸"; color: var(--muted); font-size: 10px; width: 1em; }
  .tool[open] > summary::before { content: "▾"; }
  .tool .name { font-weight: 600; color: var(--tool-accent); font-family: var(--font-mono); font-size: 12px; }
  .tool .preview { color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
  .tool .error-badge { background: var(--error); color: white; font-size: 10px; padding: 1px 6px; border-radius: 999px; }
  .tool .body { border-top: 1px solid var(--tool-border); padding: 10px 12px; background: var(--panel); }
  .tool .section-label { font-size: 11px; color: var(--muted); font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; margin: 8px 0 4px; }
  .tool .section-label:first-child { margin-top: 0; }
  .tool pre.input { background: var(--code-bg); padding: 8px 10px; border-radius: 6px; font-family: var(--font-mono); font-size: 12px; overflow-x: auto; margin: 0; max-height: 420px; }
  .tool .result-text { background: var(--code-bg); padding: 8px 10px; border-radius: 6px; font-family: var(--font-mono); font-size: 12px; white-space: pre-wrap; word-wrap: break-word; max-height: 420px; overflow: auto; }
  .tool .result-error { border-left: 3px solid var(--error); padding-left: 8px; }

  .system-divider { text-align: center; color: var(--muted); font-size: 11px; padding: 12px 0; border-top: 1px dashed var(--border-strong); border-bottom: 1px dashed var(--border-strong); margin: 20px 0; text-transform: uppercase; letter-spacing: 0.08em; }
  .system-event { font-family: var(--font-mono); font-size: 12px; color: var(--muted); background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; margin: 8px 0; }

  .role-label { font-size: 11px; color: var(--muted); font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 4px; display: flex; gap: 8px; align-items: center; }
  .role-label .model { font-family: var(--font-mono); font-size: 10px; text-transform: none; letter-spacing: 0; color: var(--muted); opacity: 0.8; }

  .copy-btn {
    position: absolute; top: 0; right: 0; opacity: 0; transition: opacity 0.15s;
    background: var(--panel); border: 1px solid var(--border-strong); border-radius: 6px;
    font-size: 11px; padding: 3px 8px; cursor: pointer; color: var(--muted);
  }
  .turn:hover .copy-btn { opacity: 1; }
  .copy-btn:hover { color: var(--text); }
  .copy-btn.copied { color: var(--accent); }

  img.inline-img { max-width: 100%; border-radius: 8px; display: block; margin: 6px 0; border: 1px solid var(--border); }

  .hide-thinking .thinking { display: none; }
  .hide-tools .tool { display: none; }
  .expand-all .tool, .expand-all .thinking { }

  footer.bottombar { position: fixed; bottom: 0; left: 0; right: 0; padding: 8px 16px; background: var(--panel); border-top: 1px solid var(--border); font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; }
  #match-count { font-variant-numeric: tabular-nums; }

  mark.hl { background: #ffe37a; color: #1f1e1c; padding: 0 2px; border-radius: 2px; }
</style>
</head>
<body>
<header class="topbar">
  <h1 id="session-title"></h1>
  <div class="meta">
    <span id="session-id-meta"></span>
    <span id="session-cwd-meta"></span>
    <span id="session-model-meta"></span>
    <span id="session-time-meta"></span>
    <span id="session-count-meta"></span>
  </div>
  <div class="controls">
    <input id="search" type="search" placeholder="Search conversation…" />
    <label><input type="checkbox" id="toggle-thinking" checked /> thinking</label>
    <label><input type="checkbox" id="toggle-tools" checked /> tools</label>
    <button id="expand-all">Expand all</button>
    <button id="collapse-all">Collapse all</button>
    <button id="copy-all">Copy all</button>
  </div>
</header>

<main class="feed" id="feed"></main>

<footer class="bottombar">
  <span>Claude session viewer</span>
  <span id="match-count"></span>
</footer>

<script id="session-data" type="application/json">__SESSION_JSON__</script>
<script>
(function() {
  const DATA = JSON.parse(document.getElementById('session-data').textContent);

  // marked + DOMPurify setup
  marked.setOptions({
    gfm: true, breaks: false, mangle: false, headerIds: false,
    highlight: function(code, lang) {
      try {
        if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
        return hljs.highlightAuto(code).value;
      } catch (e) { return code; }
    }
  });
  const renderMd = (text) => {
    if (!text) return '';
    const html = marked.parse(text);
    return DOMPurify.sanitize(html, { ADD_ATTR: ['target'] });
  };

  const escapeHtml = (s) => (s == null ? '' : String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;'));

  const fmtTime = (iso) => {
    if (!iso) return '';
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'medium' });
  };

  const el = (tag, props, children) => {
    const n = document.createElement(tag);
    if (props) Object.entries(props).forEach(([k, v]) => {
      if (k === 'class') n.className = v;
      else if (k === 'html') n.innerHTML = v;
      else if (k === 'onclick') n.addEventListener('click', v);
      else if (k === 'dataset') Object.assign(n.dataset, v);
      else n.setAttribute(k, v);
    });
    if (children) (Array.isArray(children) ? children : [children]).forEach(c => {
      if (c == null) return;
      if (typeof c === 'string') n.appendChild(document.createTextNode(c));
      else n.appendChild(c);
    });
    return n;
  };

  // --- Header meta
  document.title = DATA.meta.title || ('Claude session ' + (DATA.meta.session_id || ''));
  document.getElementById('session-title').textContent = DATA.meta.title || 'Claude session';
  const sid = DATA.meta.session_id ? DATA.meta.session_id.slice(0, 8) : '';
  const $ = (id) => document.getElementById(id);
  $('session-id-meta').innerHTML = sid ? '<code>' + escapeHtml(sid) + '</code>' : '';
  $('session-cwd-meta').textContent = DATA.meta.cwd || '';
  $('session-model-meta').textContent = DATA.meta.models && DATA.meta.models.length ? DATA.meta.models.join(', ') : '';
  if (DATA.meta.started) {
    $('session-time-meta').textContent = fmtTime(DATA.meta.started);
  }
  $('session-count-meta').textContent = DATA.events.length + ' events';

  // --- Render a single event
  const renderUser = (e) => {
    const bubble = el('div', { class: 'bubble' });
    if (e.text) bubble.appendChild(el('div', { class: 'md', html: renderMd(e.text) }));
    (e.images || []).forEach(img => {
      if (img.data) {
        bubble.appendChild(el('img', { class: 'inline-img', src: 'data:' + (img.media_type || 'image/png') + ';base64,' + img.data, alt: 'image' }));
      }
    });
    const wrap = el('div', { class: 'turn user' });
    const inner = el('div');
    if (e.timestamp) inner.appendChild(el('div', { class: 'ts' }, fmtTime(e.timestamp)));
    inner.appendChild(bubble);
    inner.appendChild(makeCopyBtn(() => e.text || ''));
    wrap.appendChild(inner);
    return wrap;
  };

  const renderPart = (part) => {
    if (part.type === 'text') {
      return el('div', { class: 'md', html: renderMd(part.text) });
    }
    if (part.type === 'thinking') {
      const d = el('details', { class: 'thinking' });
      d.appendChild(el('summary', null, 'Thinking'));
      d.appendChild(el('div', { class: 'body' }, part.text));
      return d;
    }
    if (part.type === 'tool_use') {
      return renderToolUse(part);
    }
    return null;
  };

  const summarizeInput = (input) => {
    if (!input || typeof input !== 'object') return '';
    // Heuristics for common tools
    if (input.file_path) return input.file_path;
    if (input.path) return input.path;
    if (input.pattern) return input.pattern;
    if (input.command) return input.command;
    if (input.url) return input.url;
    if (input.description) return input.description;
    if (input.prompt) return String(input.prompt).slice(0, 100);
    if (input.query) return input.query;
    try { return JSON.stringify(input).slice(0, 120); } catch (e) { return ''; }
  };

  const renderToolUse = (part) => {
    const tool = el('details', { class: 'tool' });
    const summary = el('summary');
    summary.appendChild(el('span', { class: 'name' }, part.name || 'tool'));
    summary.appendChild(el('span', { class: 'preview' }, summarizeInput(part.input)));
    if (part.result && part.result.is_error) {
      summary.appendChild(el('span', { class: 'error-badge' }, 'error'));
    }
    tool.appendChild(summary);

    const body = el('div', { class: 'body' });

    body.appendChild(el('div', { class: 'section-label' }, 'Input'));
    const inputStr = (() => {
      try { return JSON.stringify(part.input, null, 2); } catch (e) { return String(part.input); }
    })();
    body.appendChild(el('pre', { class: 'input' }, inputStr));

    if (part.result) {
      body.appendChild(el('div', { class: 'section-label' }, part.result.is_error ? 'Result (error)' : 'Result'));
      if (part.result.text) {
        const rt = el('div', { class: 'result-text' + (part.result.is_error ? ' result-error' : '') });
        rt.textContent = part.result.text;
        body.appendChild(rt);
      }
      (part.result.images || []).forEach(img => {
        if (img.data) {
          body.appendChild(el('img', { class: 'inline-img', src: 'data:' + (img.media_type || 'image/png') + ';base64,' + img.data, alt: 'tool image' }));
        }
      });
      if (part.result.tool_refs && part.result.tool_refs.length) {
        body.appendChild(el('div', { class: 'section-label' }, 'Loaded tools'));
        body.appendChild(el('div', null, part.result.tool_refs.join(', ')));
      }
    }

    tool.appendChild(body);
    return tool;
  };

  const renderAssistant = (e) => {
    const wrap = el('div', { class: 'turn assistant' });
    const label = el('div', { class: 'role-label' });
    label.appendChild(document.createTextNode('Claude'));
    if (e.model) label.appendChild(el('span', { class: 'model' }, e.model));
    wrap.appendChild(label);
    if (e.timestamp) wrap.appendChild(el('div', { class: 'ts' }, fmtTime(e.timestamp)));
    const content = el('div', { class: 'content' });
    (e.parts || []).forEach(p => {
      const n = renderPart(p);
      if (n) content.appendChild(n);
    });
    wrap.appendChild(content);
    wrap.appendChild(makeCopyBtn(() => assistantToText(e)));
    return wrap;
  };

  const assistantToText = (e) => {
    return (e.parts || []).map(p => {
      if (p.type === 'text') return p.text;
      if (p.type === 'thinking') return '(thinking)\n' + p.text;
      if (p.type === 'tool_use') {
        let s = '[tool: ' + (p.name || '') + ']';
        const preview = summarizeInput(p.input);
        if (preview) s += ' ' + preview;
        return s;
      }
      return '';
    }).filter(Boolean).join('\n\n');
  };

  const renderCompact = (e) => el('div', { class: 'system-divider' }, '⎯ compact boundary ⎯');

  const renderSystem = (e) => {
    const n = el('div', { class: 'system-event' });
    n.textContent = e.content || '';
    return n;
  };

  const makeCopyBtn = (getText) => {
    const btn = el('button', { class: 'copy-btn' }, 'copy');
    btn.addEventListener('click', async (ev) => {
      ev.stopPropagation();
      try {
        await navigator.clipboard.writeText(getText());
        btn.textContent = 'copied';
        btn.classList.add('copied');
        setTimeout(() => { btn.textContent = 'copy'; btn.classList.remove('copied'); }, 1200);
      } catch (err) {
        btn.textContent = 'error';
      }
    });
    return btn;
  };

  // --- Mount
  const feed = document.getElementById('feed');
  const frag = document.createDocumentFragment();
  DATA.events.forEach(e => {
    let node;
    switch (e.kind) {
      case 'user': node = renderUser(e); break;
      case 'assistant': node = renderAssistant(e); break;
      case 'compact': node = renderCompact(e); break;
      case 'system': node = renderSystem(e); break;
    }
    if (node) frag.appendChild(node);
  });
  feed.appendChild(frag);

  // Apply syntax highlight to code blocks rendered
  feed.querySelectorAll('.md pre code').forEach((el) => {
    try { hljs.highlightElement(el); } catch (e) {}
  });

  // --- Toggles
  $('toggle-thinking').addEventListener('change', (e) => {
    document.body.classList.toggle('hide-thinking', !e.target.checked);
  });
  $('toggle-tools').addEventListener('change', (e) => {
    document.body.classList.toggle('hide-tools', !e.target.checked);
  });
  $('expand-all').addEventListener('click', () => {
    feed.querySelectorAll('details').forEach(d => d.open = true);
  });
  $('collapse-all').addEventListener('click', () => {
    feed.querySelectorAll('details').forEach(d => d.open = false);
  });
  $('copy-all').addEventListener('click', async () => {
    const parts = [];
    DATA.events.forEach(e => {
      if (e.kind === 'user') parts.push('## User\n\n' + (e.text || ''));
      else if (e.kind === 'assistant') parts.push('## Claude\n\n' + assistantToText(e));
      else if (e.kind === 'compact') parts.push('--- compact boundary ---');
    });
    try {
      await navigator.clipboard.writeText(parts.join('\n\n'));
      const btn = $('copy-all');
      const prev = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = prev; }, 1500);
    } catch (err) { alert('Copy failed: ' + err.message); }
  });

  // --- Search / highlight
  const searchInput = $('search');
  let searchTimer;
  searchInput.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => doSearch(searchInput.value), 150);
  });

  function clearHighlights() {
    feed.querySelectorAll('mark.hl').forEach(m => {
      const parent = m.parentNode;
      parent.replaceChild(document.createTextNode(m.textContent), m);
      parent.normalize();
    });
  }

  function highlightIn(node, re) {
    let count = 0;
    const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => {
        if (!n.nodeValue || !n.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        const p = n.parentNode;
        if (!p) return NodeFilter.FILTER_REJECT;
        if (['SCRIPT', 'STYLE'].includes(p.nodeName)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const targets = [];
    let cur;
    while ((cur = walker.nextNode())) targets.push(cur);
    targets.forEach(textNode => {
      const val = textNode.nodeValue;
      re.lastIndex = 0;
      if (!re.test(val)) return;
      re.lastIndex = 0;
      const frag = document.createDocumentFragment();
      let last = 0, m;
      while ((m = re.exec(val)) !== null) {
        if (m.index > last) frag.appendChild(document.createTextNode(val.slice(last, m.index)));
        const mark = document.createElement('mark');
        mark.className = 'hl';
        mark.textContent = m[0];
        frag.appendChild(mark);
        last = m.index + m[0].length;
        count++;
        if (m.index === re.lastIndex) re.lastIndex++;
      }
      if (last < val.length) frag.appendChild(document.createTextNode(val.slice(last)));
      textNode.parentNode.replaceChild(frag, textNode);
    });
    return count;
  }

  function doSearch(q) {
    clearHighlights();
    const mc = $('match-count');
    if (!q || q.length < 2) { mc.textContent = ''; return; }
    const safe = q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(safe, 'gi');
    const count = highlightIn(feed, re);
    mc.textContent = count ? count + ' matches' : 'no matches';
    const first = feed.querySelector('mark.hl');
    if (first) {
      // open any parent details so match is visible
      let p = first.parentNode;
      while (p && p !== feed) {
        if (p.tagName === 'DETAILS') p.open = true;
        p = p.parentNode;
      }
      first.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }
})();
</script>
</body>
</html>
"""


def build_html(meta: dict, events: list[dict], out_path: str) -> None:
    payload = {"meta": meta, "events": events}
    # JSON embedded inside a <script type="application/json"> block — we must prevent
    # an inner "</script>" from terminating the script tag. Escape the forward slash.
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    title = meta.get("title") or f"Claude session {meta.get('session_id') or ''}".strip()
    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SESSION_JSON__", data_json)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to session .jsonl")
    ap.add_argument("-o", "--output", help="Output HTML path (defaults to <input>.html)")
    args = ap.parse_args()

    in_path = args.input
    if not os.path.exists(in_path):
        print(f"input not found: {in_path}", file=sys.stderr)
        return 1

    out_path = args.output or (os.path.splitext(in_path)[0] + ".html")

    records = load_records(in_path)
    meta = extract_meta(records)
    events = build_events(records)

    build_html(meta, events, out_path)

    size = os.path.getsize(out_path)
    print(f"wrote {out_path} ({size/1024/1024:.2f} MB) — {len(events)} events from {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
