# Smart Calculator + Web Search Agent (Gemini-Powered Agentic AI)

A minimal but complete agentic AI system built with Python and the Gemini API.  
The agent can autonomously decide when to perform a web search, evaluate a mathematical expression, or read/write to a persistent memory file — and can chain these tools together — to answer user queries.

This project is designed as a **first agentic AI project** to teach:
- The core agent loop (plan → act → observe → repeat)
- Function calling (tool use) with Gemini
- **Manual vs. automatic function calling** — same tools, two execution modes you can compare side-by-side
- Safe evaluation of math expressions
- Free web search integration (no extra API keys)
- A tiny persistent memory store for cross-session facts

---

## 📂 Project Structure


---

## 🧠 How the Agent Works

The agent runs in a **loop**:

1. **User sends a message** to the agent.
2. **Gemini decides** whether to reply directly or call a tool.
   - If a tool is requested, the model returns a `function_call` with the tool name and arguments.
3. **Your Python code executes the requested tool** (web search or calculator) and returns the result as a `function_response` to the model.
4. **Gemini processes the result** and may:
   - Request another tool call (e.g., search first, then calculate with the number), or
   - Generate the final text answer.
5. **Loop ends** when the model outputs a plain text response (no more tool calls).

A safety limit (5 turns) prevents infinite loops.

---

## 🔧 Tools

### `web_search`
- Uses the free **DuckDuckGo** instant answer API (via `duckduckgo-search` library).
- Returns the top 3 results (title, URL, snippet) formatted as plain text.
- No API key required.

### `calculator`
- Safely evaluates a mathematical expression.
- Only allows a whitelist of functions from Python’s `math` module (e.g., `sqrt`, `sin`, `cos`, `log`, etc.) and a few built‑ins (`abs`, `round`).
- Uses `eval()` with restricted globals (`__builtins__` = `{}`) to block dangerous calls.
- Returns the result as a string.

### `save_fact` / `recall_facts`
- A persistent memory store backed by a JSON file at `~/.smartcal/memory.json`.
- `save_fact(fact, namespace="default")` appends a fact; `recall_facts(query="", namespace="default")` returns a numbered list of matching facts (substring match; empty query returns all).
- Use it to remember things across turns: *"Remember that my favourite colour is teal."* then *"What's my favourite colour?"*
- ⚠️ On Streamlit Cloud the file lives on the server's filesystem, so all visitors share one memory — fine for a demo, **not** for multi-user production.

---

## ⚡ Execution modes: manual vs. automatic function calling

The agent supports two ways to run its tool loop:

| Mode | Where the loop lives | Streaming events | How to enable |
|---|---|---|---|
| **Manual** (default) | Your code: `run_agent_stream` in `core.py` inspects each model response, runs tools, and feeds results back | Per-tool `tool_call` / `tool_result` events render live | (default) |
| **AFC** | The `google-genai` SDK runs tools automatically inside `send_message` | Only a final answer plus an `afc_summary` listing the calls the SDK made | CLI: `set SMARTCAL_AFC=1` · Web: sidebar checkbox |

Both modes register the **same four tools** — `web_search`, `calculator`, `save_fact`, `recall_facts` — and the **same model**. The difference is who drives the loop. Try the same prompt in both modes to see the difference:

```bash
# Manual (default)
python agent.py

# Automatic function calling
set SMARTCAL_AFC=1
python agent.py
```

In the Streamlit app, tick **⚡ Use automatic function calling** in the sidebar to switch modes.

---

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.9+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### 2. Clone / Create the project
```bash
mkdir agent-calc-search
cd agent-calc-search
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate