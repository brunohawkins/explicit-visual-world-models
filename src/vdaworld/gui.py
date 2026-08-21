import os
import subprocess
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

app = FastAPI()
BASE_DIR = Path.cwd()
RESULTS_DIR = BASE_DIR / "results"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vdaworld - Results Explorer</title>
    <style>
        :root {
            --bg: #1e1e1e;
            --sidebar-bg: #252526;
            --text: #d4d4d4;
            --text-muted: #858585;
            --accent: #007acc;
            --border: #3c3c3c;
            --font-mono: 'Consolas', 'Courier New', monospace;
            --scrollbar-bg: #1e1e1e;
            --scrollbar-thumb: #424242;
        }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 10px; height: 10px; background: var(--scrollbar-bg); }
        ::-webkit-scrollbar-thumb { background: var(--scrollbar-thumb); border-radius: 5px; }
        body {
            margin: 0; padding: 0;
            font-family: var(--font-mono);
            background-color: var(--bg);
            color: var(--text);
            display: flex;
            height: 100vh;
            overflow: hidden;
            font-size: 14px;
        }
        #sidebar {
            width: 320px;
            min-width: 200px;
            background: var(--sidebar-bg);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }
        .header {
            padding: 15px;
            border-bottom: 1px solid var(--border);
            font-weight: bold;
            font-size: 1.1em;
            color: #ccc;
        }
        .tree { padding: 10px; list-style: none; margin: 0; }
        .tree-item {
            cursor: pointer;
            padding: 4px 6px;
            user-select: none;
            word-break: break-all;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            display: block;
        }
        .tree-item:hover { background-color: #2a2d2e; }
        .tree-item.active { background-color: #37373d; color: #fff; }
        .tree-item.result-item { font-weight: bold; color: #9cdcfe; }
        ul.tree-list {
            list-style: none;
            padding-left: 15px;
            margin: 0;
            display: none;
            border-left: 1px solid #333;
            margin-left: 5px;
        }
        ul.tree-list.open { display: block; }
        #main {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            background-color: #1e1e1e;
        }
        #topbar {
            padding: 10px 15px;
            border-bottom: 1px solid var(--border);
            background: var(--sidebar-bg);
            display: flex;
            align-items: center;
            gap: 15px;
        }
        #content {
            flex: 1;
            padding: 20px;
            overflow: auto;
        }
        button {
            background: #333;
            color: #ccc;
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-family: var(--font-mono);
            font-size: 13px;
            transition: all 0.2s;
        }
        button:hover { background: var(--accent); color: white; border-color: var(--accent); }
        pre {
            background: inherit;
            margin: 0;
            padding: 0;
            overflow-x: auto;
            white-space: pre-wrap;
            font-family: var(--font-mono);
            font-size: 13px;
        }
        img { max-width: 100%; border: 1px dashed var(--border); }
        video { max-width: 100%; outline: none; border: 1px solid var(--border); }
        .diff-added { color: #85e89d; background: rgba(35, 75, 35, 0.4); display: block; width: 100%; }
        .diff-removed { color: #f97583; background: rgba(92, 24, 24, 0.4); display: block; width: 100%; }
        .diff-header { color: #79b8ff; }
        .code-container { counter-reset: line; line-height: 1.5; }
        .code-line { display: block; }

        /* Dashboard Styles */
        .dashboard { display: flex; flex-direction: column; gap: 20px; }
        .dash-section { background: var(--sidebar-bg); padding: 15px; border-radius: 5px; border: 1px solid var(--border); }
        .dash-header { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; color: #fff; border-bottom: 1px solid var(--border); padding-bottom: 5px; }
        .dash-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .media-grid { display: flex; flex-wrap: wrap; gap: 10px; }
        .media-item { flex: 1; min-width: 200px; max-width: 400px; background: #1e1e1e; padding: 10px; border-radius: 5px; text-align: center; }
        .media-item p { margin: 5px 0 0 0; font-size: 12px; color: var(--text-muted); word-break: break-all; }
        .tool-call { border-left: 3px solid var(--accent); background: #1e1e1e; margin-bottom: 10px; padding: 10px; border-radius: 0 4px 4px 0;}
    </style>
</head>
<body>

<div id="sidebar">
    <div class="header">📂 results/</div>
    <div id="tree-container" class="tree">Loading...</div>
</div>

<div id="main">
    <div id="topbar">
        <button onclick="fetchGitDiff()">Show git diff HEAD</button>
        <button onclick="fetchGitLog()">Show git diff (Last Commit)</button>
        <span id="current-file" style="color: var(--text-muted); margin-left: auto; font-size: 13px;"></span>
    </div>
    <div id="content">Select a file or a highlighted result directory to view</div>
</div>

<script>
    async function fetchTree() {
        const res = await fetch('/api/tree');
        const data = await res.json();
        document.getElementById('tree-container').innerHTML = buildTreeHtml(data);
    }

    function buildTreeHtml(node) {
        if (node.type === 'file') {
            return `<div class="tree-item" onclick="viewFile('${node.path}', this)">📄 ${node.name}</div>`;
        }
        
        let extraClass = node.is_result ? 'result-item' : '';
        let icon = node.is_result ? '📊' : '📁';
        
        let html = `<div class="tree-item ${extraClass}" onclick="handleDirClick(this, '${node.path}', ${node.is_result})">${icon} ${node.name}</div>`;
        html += `<ul class="tree-list">`;
        for (let child of node.children) {
            html += `<li>${buildTreeHtml(child)}</li>`;
        }
        html += `</ul>`;
        return html;
    }

    function handleDirClick(el, path, isResult) {
        let ul = el.nextElementSibling;
        if (ul && ul.tagName === 'UL') {
            ul.classList.toggle('open');
            // update icon only if not a result
            if (!isResult) {
                el.innerText = ul.classList.contains('open') ? '📂 ' + el.innerText.slice(2) : '📁 ' + el.innerText.slice(2);
            }
        }
        
        if (isResult) {
            viewResult(path, el);
        }
    }

    function setActive(el) {
        document.querySelectorAll('.tree-item.active').forEach(i => i.classList.remove('active'));
        if (el) el.classList.add('active');
    }

    async function viewResult(path, el) {
        setActive(el);
        document.getElementById('current-file').innerText = "Dashboard: " + path;
        const out = document.getElementById('content');
        out.innerHTML = '<span style="color:#888;">Loading Result Dashboard...</span>';
        
        try {
            const res = await fetch(`/api/result_data?path=${encodeURIComponent(path)}`);
            if (!res.ok) {
                out.innerHTML = `<pre>Failed to load result data for ${path}</pre>`;
                return;
            }
            const data = await res.json();
            
            let html = `<div class="dashboard">`;
            
            // Overview Section
            html += `<div class="dash-section">
                <div class="dash-header">Overview</div>
                <div class="dash-grid">
                    <div>
                        <strong>State</strong>
                        <pre class="code-container" style="max-height:300px;overflow-y:auto;">${syntaxHighlightJSON(JSON.stringify(data.state || {}, null, 2))}</pre>
                    </div>
                    <div>
                        <strong>Files Present</strong>
                        <ul style="padding-left:20px;margin-top:5px;">
                            ${data.has_prompt ? `<li><a href="#" onclick="viewFile('${path}/prompt.md', null)" style="color:var(--accent);">prompt.md</a></li>` : ''}
                            ${data.has_response ? `<li><a href="#" onclick="viewFile('${path}/response.md', null)" style="color:var(--accent);">response.md</a></li>` : ''}
                            ${data.has_turns ? `<li><a href="#" onclick="viewFile('${path}/turns.md', null)" style="color:var(--accent);">turns.md</a></li>` : ''}
                            ${data.has_simulator ? `<li><a href="#" onclick="viewFile('${path}/simulator_gen.py', null)" style="color:var(--accent);">simulator_gen.py</a></li>` : ''}
                        </ul>
                    </div>
                </div>
            </div>`;
            
            // Visualizations Section
            if (data.visualizations && data.visualizations.length > 0) {
                html += `<div class="dash-section">
                    <div class="dash-header">Visualizations</div>
                    <div class="media-grid">`;
                for (let vis of data.visualizations) {
                    let mediaUrl = `/api/file?path=${encodeURIComponent(vis.path)}`;
                    html += `<div class="media-item">`;
                    if (vis.name.match(/\\.(mp4|webm|ogg)$/i)) {
                        html += `<video controls src="${mediaUrl}"></video>`;
                    } else if (vis.name.match(/\\.(png|jpg|jpeg|gif|webp)$/i)) {
                        html += `<img src="${mediaUrl}" loading="lazy"/>`;
                    } else {
                        html += `<a href="${mediaUrl}" target="_blank">File</a>`;
                    }
                    html += `<p>${vis.name}</p></div>`;
                }
                html += `</div></div>`;
            }
            
            // Tool Calls Section
            if (data.tool_calls && data.tool_calls.length > 0) {
                html += `<div class="dash-section" style="max-height: 600px; overflow-y: auto;">
                    <div class="dash-header">Tool Calls</div>`;
                for (let tc of data.tool_calls) {
                    html += `<div class="tool-call">
                        <strong>[${tc.index}] ${tc.tool || 'Unknown Tool'}</strong>
                        <div style="font-size:12px;color:var(--text-muted);margin-bottom:5px;">${tc.path}</div>
                        <pre style="font-size:12px; background:#111; max-height:200px; overflow-y:auto;">${escapeHtml(JSON.stringify(tc.args, null, 2) || '{}')}</pre>`;
                    if (tc.summary) {
                        html += `<div style="margin-top:5px; font-size:12px; color:#c586c0;">${escapeHtml(tc.summary)}</div>`;
                    }
                    html += `</div>`;
                }
                html += `</div>`;
            }
            
            html += `</div>`;
            out.innerHTML = html;
            
        } catch (e) {
            out.innerHTML = `<pre>Error rendering dashboard.</pre>`;
        }
    }

    async function viewFile(path, el) {
        if (el) setActive(el);
        document.getElementById('current-file').innerText = path;
        const out = document.getElementById('content');
        out.innerHTML = '<span style="color:#888;">Loading...</span>';
        
        if (path.match(/\\.(mp4|webm|ogg)$/i)) {
            out.innerHTML = `<video controls autoplay style="max-width:100%; max-height:80vh;" src="/api/file?path=${encodeURIComponent(path)}"></video>`;
            return;
        }
        if (path.match(/\\.(png|jpg|jpeg|gif|webp)$/i)) {
            out.innerHTML = `<img src="/api/file?path=${encodeURIComponent(path)}" style="max-width:100%; max-height:80vh;" />`;
            return;
        }
        
        try {
            const res = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
            if (!res.ok) {
                out.innerHTML = `<pre>Status ${res.status}: Failed to load file.</pre>`;
                return;
            }
            const text = await res.text();
            
            if (path.endsWith('.json')) {
                try {
                    const obj = JSON.parse(text);
                    out.innerHTML = `<pre class="code-container">${syntaxHighlightJSON(JSON.stringify(obj, null, 2))}</pre>`;
                } catch {
                    out.innerHTML = `<pre class="code-container">${escapeHtml(text)}</pre>`;
                }
            } else {
                out.innerHTML = `<pre class="code-container">${escapeHtml(text)}</pre>`;
            }
        } catch (e) {
            out.innerHTML = `<pre>Error loading file.</pre>`;
        }
    }

    async function fetchGitDiff() {
        setActive(null);
        document.getElementById('current-file').innerText = 'git diff HEAD';
        const out = document.getElementById('content');
        out.innerHTML = '<span style="color:#888;">Loading diff...</span>';
        try {
            const res = await fetch('/api/diff');
            const data = await res.json();
            if (data.diff) {
                out.innerHTML = renderDiff(data.diff);
            } else {
                out.innerHTML = '<pre>No changes</pre>';
            }
        } catch (e) {
            out.innerHTML = '<pre>Error loading diff.</pre>';
        }
    }
    
    async function fetchGitLog() {
        setActive(null);
        document.getElementById('current-file').innerText = 'git diff HEAD~1';
        const out = document.getElementById('content');
        out.innerHTML = '<span style="color:#888;">Loading diff...</span>';
        try {
            const res = await fetch('/api/diff?commit=HEAD~1');
            const data = await res.json();
            if (data.diff) {
                out.innerHTML = renderDiff(data.diff);
            } else {
                out.innerHTML = '<pre>No changes</pre>';
            }
        } catch (e) {
            out.innerHTML = '<pre>Error loading diff.</pre>';
        }
    }

    function renderDiff(diffText) {
        let lines = diffText.split('\\n');
        let html = '<pre class="code-container">';
        for (let line of lines) {
            if (line.startsWith('+') && !line.startsWith('+++')) {
                html += `<span class="code-line diff-added">${escapeHtml(line)}</span>`;
            } else if (line.startsWith('-') && !line.startsWith('---')) {
                html += `<span class="code-line diff-removed">${escapeHtml(line)}</span>`;
            } else if (line.startsWith('@@ ') || line.startsWith('diff --git ')) {
                html += `<span class="code-line diff-header">${escapeHtml(line)}</span>`;
            } else {
                html += `<span class="code-line">${escapeHtml(line)}</span>`;
            }
        }
        html += '</pre>';
        return html;
    }

    function escapeHtml(unsafe) {
        return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function syntaxHighlightJSON(json) {
        let text = escapeHtml(json);
        text = text.replace(/("(\\\\u[a-zA-Z0-9]{4}|\\\\[^u]|[^\\\\"])*"(\\s*:)?|\\b(true|false|null)\\b|-?\\d+(?:\\.\\d*)?(?:[eE][+\\-]?\\d+)?)/g, function (match) {
            let cls = 'number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    return `<span style="color: #9cdcfe;">${match.slice(0, -1)}</span><span style="color: #d4d4d4;">:</span>`;
                } else {
                    return `<span style="color: #ce9178;">${match}</span>`;
                }
            } else if (/true|false/.test(match)) {
                return `<span style="color: #569cd6;">${match}</span>`;
            } else if (/null/.test(match)) {
                return `<span style="color: #569cd6;">${match}</span>`;
            }
            return `<span style="color: #b5cea8;">${match}</span>`;
        });
        return text;
    }

    fetchTree();
</script>

</body>
</html>
"""

@app.get("/")
def index():
    return HTMLResponse(HTML_TEMPLATE)

def get_directory_tree(path: Path, rel_to: Path):
    tree = {
        "name": path.name, 
        "type": "directory", 
        "path": str(path.relative_to(rel_to).as_posix()), 
        "children": [],
        "is_result": False
    }
    
    if not path.is_dir():
        return tree
    
    # Simple heuristic to identify if it's a "result" directory
    # Contains a state.json or prompt.md or simulator_gen.py
    if (path / "state.json").exists() or (path / "prompt.md").exists() or (path / "simulator_gen.py").exists():
        tree["is_result"] = True
    
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not (p.is_dir() and not p.is_symlink()), p.name.lower()))
        for entry in entries:
            # Ignore hidden files, caches, etc.
            if entry.name.startswith(".") or entry.name in ["__pycache__", ".git", ".venv", ".mypy_cache"]:
                continue
            if entry.is_dir():
                tree["children"].append(get_directory_tree(entry, rel_to))
            else:
                tree["children"].append({
                    "name": entry.name,
                    "type": "file",
                    "path": str(entry.relative_to(rel_to).as_posix())
                })
    except PermissionError:
        pass
    return tree

@app.get("/api/tree")
def get_tree():
    if not RESULTS_DIR.exists():
        return {"name": "results", "type": "directory", "path": "", "children": [], "is_result": False}
    return get_directory_tree(RESULTS_DIR, RESULTS_DIR)

@app.get("/api/result_data")
def get_result_data(path: str):
    target = (RESULTS_DIR / path).resolve()
    try:
        target.relative_to(RESULTS_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
        
    if not target.is_dir():
        return JSONResponse({"error": "Not a directory"}, status_code=400)
    
    data = {
        "state": None,
        "has_prompt": (target / "prompt.md").exists(),
        "has_response": (target / "response.md").exists(),
        "has_turns": (target / "turns.md").exists(),
        "has_simulator": (target / "simulator_gen.py").exists(),
        "visualizations": [],
        "tool_calls": []
    }
    
    # Load State
    state_file = target / "state.json"
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                data["state"] = json.load(f)
        except Exception:
            pass

    # Gather Visualizations
    vis_dir = target / "visualizations"
    if vis_dir.exists() and vis_dir.is_dir():
        for item in vis_dir.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                data["visualizations"].append({
                    "name": item.name,
                    "path": str(item.relative_to(RESULTS_DIR).as_posix())
                })

    # Gather Tool Calls
    tc_dir = target / "tool_calls"
    if tc_dir.exists() and tc_dir.is_dir():
        for tc_folder in sorted(tc_dir.iterdir(), key=lambda x: x.name):
            if tc_folder.is_dir():
                meta_file = tc_folder / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r") as f:
                            meta = json.load(f)
                        data["tool_calls"].append({
                            "index": meta.get("index", tc_folder.name),
                            "tool": meta.get("tool", "unknown"),
                            "args": meta.get("args", {}),
                            "summary": meta.get("result_summary", ""),
                            "path": str(tc_folder.relative_to(RESULTS_DIR).as_posix())
                        })
                    except Exception:
                        pass
                        
    # Sort tool calls numerically if possible
    try:
        data["tool_calls"].sort(key=lambda x: int(str(x["index"]).split('_')[0]) if str(x["index"]).split('_')[0].isdigit() else 999)
    except Exception:
        pass
        
    return JSONResponse(data)

@app.get("/api/file")
def get_file(path: str):
    # Ensure path doesn't escape RESULTS_DIR
    target = (RESULTS_DIR / path).resolve()
    try:
        target.relative_to(RESULTS_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    
    if not target.exists() or not target.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)
        
    return FileResponse(target)

@app.get("/api/diff")
def git_diff(commit: str = "HEAD"):
    try:
        result = subprocess.run(["git", "diff", commit], cwd=BASE_DIR, capture_output=True, text=True)
        return {"diff": result.stdout}
    except Exception as e:
        return {"diff": f"Error running git diff: {e}"}

def main():
    import sys
    print("Starting vdaworld GUI server...")
    print("Open http://127.0.0.1:8000 in your browser.")
    uvicorn.run("vdaworld.gui:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
