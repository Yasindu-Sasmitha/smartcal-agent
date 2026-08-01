"""Agent core for the Smart Calculator + Web Search agent.

Exposes:
- `run_agent_stream(user_message)`: manual loop, yields structured events live.
- `run_agent_stream_afc(user_message)`: SDK-driven automatic function calling.
- `run_agent_stream_dispatch(user_message, use_afc)`: choose between the two.
- `MODEL_NAME`, `TOOLS`, `perform_web_search`, `safe_calculator`,
  `save_fact`, `recall_facts`: shared.

Frontend-agnostic: no Rich, no Streamlit imports. The CLI lives in
`agent.py`; the Streamlit app lives in `app.py`.
"""

# NOTE: deliberately NOT using `from __future__ import annotations` here.
# PEP-563 turns all annotations into strings, which breaks the
# google-genai SDK's automatic function calling introspection (it does
# `isinstance(arg, annotation)` and chokes on the string).

import json
import math
import os
import re
import ssl
import sys
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Generator

# --------------------------------------------------------------------------- #
# Use certifi's CA bundle for all HTTPS calls. The system cert store on Windows
# (and on Streamlit Cloud) is frequently out of date; without this, calls to
# Wikipedia / DuckDuckGo / the Gemini API fail with CERTIFICATE_VERIFY_FAILED.
# --------------------------------------------------------------------------- #
try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # certifi missing — fall back to system defaults
    _SSL_CTX = ssl.create_default_context()

# --------------------------------------------------------------------------- #
# Silence the duckduckgo_search deprecation noise                              #
# --------------------------------------------------------------------------- #
#
# The `duckduckgo_search` package emits a RuntimeWarning on every DDGS()
# invocation. Standard `warnings.filterwarnings(...)` rules are inconsistent
# across Python versions for RuntimeWarning raised from a `with`-block, so we
# monkey-patch `warnings.warn` to swallow exactly that family of messages and
# pass everything else through untouched. This must run BEFORE the first
# DDGS() call (i.e. at module import time).
import html  # noqa: E402

_warn = warnings.warn


def _quiet_warn(message, *args, **kwargs):
    s = message if isinstance(message, str) else str(message)
    low = s.lower()
    if "duckduckgo" in low or "explicit none" in low:
        return  # swallow
    return _warn(message, *args, **kwargs)


warnings.warn = _quiet_warn
warnings.simplefilter("default", RuntimeWarning)

# Point all HTTPS clients (including primp, used by duckduckgo_search) at
# certifi's CA bundle. This is a no-op if certifi isn't installed.
try:
    import certifi as _certifi

    os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except ImportError:
    pass

# Now safe to import the noisy library.
from dotenv import load_dotenv  # noqa: E402
from duckduckgo_search import DDGS  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

load_dotenv()
_api_key = os.getenv("GEMINI_API_KEY")

if not _api_key or _api_key == "your-gemini-api-key-here":
    sys.stderr.write(
        "\nGEMINI_API_KEY is not set.\n"
        "  1. Get a free key at https://aistudio.google.com/apikey\n"
        "  2. Paste it into `agent-calc-search/.env` after `GEMINI_API_KEY=`\n\n"
    )
    sys.exit(1)

client = genai.Client(api_key=_api_key)
MODEL_NAME = 'models/gemini-3.1-flash-lite-preview'
MAX_TURNS = 5

# --------------------------------------------------------------------------- #
# Tool declarations                                                           #
# --------------------------------------------------------------------------- #

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for current information. Returns the top 3 results "
            "(title, link, snippet)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression safely. Supports +, -, *, /, "
            "**, and math functions (sqrt, sin, cos, etc.). Returns the result "
            "as a number."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The mathematical expression to calculate, e.g. "
                        "'sqrt(16) + 2'."
                    ),
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "save_fact",
        "description": (
            "Save a fact to long-term memory so it can be recalled in future "
            "conversations. Use this when the user asks you to remember "
            "something (name, preference, a number, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact to remember, written as a complete sentence.",
                },
                "namespace": {
                    "type": "string",
                    "description": (
                        "Optional category to group facts under (e.g. "
                        "'preferences', 'people'). Defaults to 'default'."
                    ),
                },
            },
            "required": ["fact"],
        },
    },
    {
        "name": "recall_facts",
        "description": (
            "Recall previously saved facts from long-term memory. Returns a "
            "numbered list of matching facts, or 'No matching facts.' if none "
            "match. Call this when the user asks you to remember something "
            "from earlier."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Substring to filter facts by. Empty string returns "
                        "all facts in the namespace."
                    ),
                },
                "namespace": {
                    "type": "string",
                    "description": "Optional category. Defaults to 'default'.",
                },
            },
            "required": [],
        },
    },
]

# --------------------------------------------------------------------------- #
# Tool implementations                                                        #
# --------------------------------------------------------------------------- #

_SEARCH_EMPTY = "__NO_RESULTS__"


def _ddg_search(query: str):
    """DuckDuckGo backend. Returns a list of result dicts or _SEARCH_EMPTY."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception:
        return _SEARCH_EMPTY
    return results if results else _SEARCH_EMPTY


def _wikipedia_search(query: str):
    """Wikipedia REST fallback. Returns a list of dicts shaped like the
    DuckDuckGo results so downstream code does not need to branch."""
    try:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": "3",
            }
        )
        url = "https://en.wikipedia.org/w/api.php?" + params
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "SmartCal/1.0 (https://github.com/local/smartcal)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return _SEARCH_EMPTY
        results = []
        for h in hits:
            title = h.get("title", "No title")
            slug = title.replace(" ", "_")
            snippet = html.unescape(h.get("snippet", ""))
            snippet = re.sub(r"<[^>]+>", "", snippet)
            results.append(
                {
                    "title": title,
                    "href": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(slug),
                    "body": snippet,
                }
            )
        return results
    except Exception as exc:
        # Surface the failure to stderr so we can see why the fallback failed.
        sys.stderr.write(f"[wikipedia fallback] {type(exc).__name__}: {exc}\n")
        return _SEARCH_EMPTY


def perform_web_search(query: str):
    """Try DuckDuckGo first; if it returns nothing, fall back to Wikipedia.
    Returns a list of result dicts, the `_SEARCH_EMPTY` sentinel, or an error
    string."""
    results = _ddg_search(query)
    if results != _SEARCH_EMPTY:
        return results
    return _wikipedia_search(query)


def safe_calculator(expression: str) -> str:
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("__")
    }
    allowed_names.update({"abs": abs, "round": round})
    safe_globals = {"__builtins__": {}}
    safe_globals.update(allowed_names)
    try:
        return str(eval(expression, safe_globals, {}))  # noqa: S307 — sandboxed
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


# --------------------------------------------------------------------------- #
# Memory store (persistent, per-user JSON file)                                #
# --------------------------------------------------------------------------- #
#
# A small key-value store keyed by namespace. Each namespace holds a list of
# fact strings. Default location: ~/.smartcal/memory.json. Survives across
# sessions and across the CLI / Streamlit frontends.
#
# This is intentionally simple — substring matching, no embeddings, no TTL.
# Adequate for "remember my name" / "what's my favorite color" use cases.

MEMORY_PATH = Path.home() / ".smartcal" / "memory.json"


class MemoryStore:
    """Tiny JSON-backed key-value store for facts."""

    def __init__(self, path: Path = MEMORY_PATH):
        self.path = path

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
        except (OSError, json.JSONDecodeError):
            # Corrupt file: start fresh rather than crash.
            return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def save(self, namespace: str, fact: str) -> str:
        data = self._load()
        data.setdefault(namespace, [])
        if fact not in data[namespace]:
            data[namespace].append(fact)
        self._save(data)
        return f"Saved to memory [{namespace}]: {fact}"

    def recall(self, namespace: str, query: str = "") -> str:
        data = self._load()
        facts = data.get(namespace, [])
        if not facts:
            return "No matching facts."
        if query:
            q = query.lower()
            matches = [f for f in facts if q in f.lower()]
        else:
            matches = list(facts)
        if not matches:
            return "No matching facts."
        matches = matches[:20]
        return "\n".join(f"{i}. {f}" for i, f in enumerate(matches, 1))


# Module-level singleton so callers (and the AFC callables) share state.
_memory = MemoryStore()


def save_fact(fact: str, namespace: str = "default") -> str:
    """Save a fact to long-term memory. Returns a short confirmation
    string the model can quote back to the user."""
    return _memory.save(namespace, fact)


def recall_facts(query: str = "", namespace: str = "default") -> str:
    """Recall facts from long-term memory. Substring-filtered by `query`;
    empty query returns all facts in the namespace. Returns a numbered
    list, or 'No matching facts.'"""
    return _memory.recall(namespace, query)


# --------------------------------------------------------------------------- #
# AFC wrappers (used by the automatic-function-calling path)                    #
# --------------------------------------------------------------------------- #
#
# The google-genai SDK with automatic_function_calling enabled accepts Python
# callables directly. It inspects their signatures and docstrings to build the
# JSON schema, then transparently runs the function and feeds the result back
# to the model. These wrappers exist for two reasons:
#
#   1. They must return plain JSON-serializable values — no Python sentinels,
#      no `__NO_RESULTS__` marker. Empty search results are mapped to a
#      human-readable string ("No results found.") instead.
#   2. Their docstrings act as the tool description sent to the model.
#
# Keep these thin — they only translate between AFC's expectations and the
# underlying implementations.

def web_search_afc(query: str) -> str:
    """Search the web for current information. Returns the top 3 results
    (title, link, snippet) as a numbered list, or 'No results found.'"""
    results = perform_web_search(query)
    if results == _SEARCH_EMPTY or not isinstance(results, list):
        return "No results found."
    return _serialise_for_model(results)


def calculator_afc(expression: str) -> str:
    """Evaluate a mathematical expression safely. Supports +, -, *, /, **,
    and math functions (sqrt, sin, cos, log, etc.). Returns the result as
    a number string."""
    return safe_calculator(expression)


def save_fact_afc(fact: str, namespace: str = "default") -> str:
    """Save a fact to long-term memory so it can be recalled in future
    conversations. Use when the user asks you to remember something."""
    return save_fact(fact, namespace)


def recall_facts_afc(query: str = "", namespace: str = "default") -> str:
    """Recall previously saved facts from long-term memory. Returns a
    numbered list of matching facts, or 'No matching facts.' if none
    match. Call this when the user asks you to remember something from
    earlier."""
    return recall_facts(query, namespace)


def _serialise_for_model(payload) -> str:
    """Coerce tool output into a string the model can read."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        if not payload:
            return "No results found."
        chunks = []
        for i, r in enumerate(payload, 1):
            chunks.append(
                f"{i}. {r.get('title', 'No title')}\n"
                f"   {r.get('href') or r.get('url', '')}\n"
                f"   {r.get('body') or r.get('snippet', 'No description')}"
            )
        return "\n\n".join(chunks)
    return str(payload)


# --------------------------------------------------------------------------- #
# Streaming agent                                                             #
# --------------------------------------------------------------------------- #
#
# Yields structured event dicts so any UI (terminal, Streamlit, web) can
# render them. Event shapes:
#
#   {"type": "status",  "text": "Thinking..."}
#   {"type": "tool_call", "name": "web_search"|"calculator"|"save_fact"|"recall_facts", "args": {...}}
#   {"type": "tool_result", "name": ..., "payload": ...}
#   {"type": "afc_summary", "calls": [{"name": ..., "args": ...}, ...]}  # only in AFC mode
#   {"type": "final",   "text": "..."}
#   {"type": "aborted", "reason": "no_results"}
#
# Frontends should treat `final` / `aborted` as terminal.


def run_agent_stream(user_message: str) -> Generator[dict, None, None]:
    chat = client.chats.create(
        model=MODEL_NAME,
        config={"tools": [{"function_declarations": TOOLS}]},
    )

    yield {"type": "status", "text": "Thinking..."}
    response = chat.send_message(user_message)

    for _ in range(MAX_TURNS):
        fn_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call
        ]
        if not fn_calls:
            break

        search_failed = False
        for fn in fn_calls:
            args = dict(fn.args)
            yield {"type": "tool_call", "name": fn.name, "args": args}

            if fn.name == "web_search":
                payload = perform_web_search(args.get("query", ""))
                if payload == _SEARCH_EMPTY:
                    search_failed = True
            elif fn.name == "calculator":
                payload = safe_calculator(args.get("expression", ""))
            elif fn.name == "save_fact":
                payload = save_fact(
                    args.get("fact", ""),
                    namespace=args.get("namespace", "default"),
                )
            elif fn.name == "recall_facts":
                payload = recall_facts(
                    query=args.get("query", ""),
                    namespace=args.get("namespace", "default"),
                )
            else:
                payload = f"Unknown tool: {fn.name}"

            yield {"type": "tool_result", "name": fn.name, "payload": payload}

        if search_failed:
            yield {
                "type": "aborted",
                "reason": "no_results",
                "text": (
                    "I couldn't find any web results for that query, "
                    "so I'm not going to guess. Try rephrasing, or ask "
                    "something I can answer without looking it up."
                ),
            }
            return

        yield {"type": "status", "text": "Thinking..."}
        response = chat.send_message(
            types.Part(
                function_response=types.FunctionResponse(
                    name=fn_calls[0].name,
                    response={"result": _serialise_for_model(payload)},
                )
            )
        )

    try:
        final_text = response.text
    except (ValueError, AttributeError):
        final_text = "_No final text response from the model._"
    yield {"type": "final", "text": final_text}


# --------------------------------------------------------------------------- #
# Automatic Function Calling (AFC) path                                       #
# --------------------------------------------------------------------------- #
#
# Same four tools, but the SDK runs them automatically inside send_message().
# We only see the final text plus a list of calls recorded in
# `response.automatic_function_calling_history`. We surface that list as a
# single `afc_summary` event so the UI can show *what* the agent did without
# us having to reconstruct every intermediate step.


def run_agent_stream_afc(user_message: str) -> Generator[dict, None, None]:
    """Run the agent via the SDK's automatic function calling mode.

    Yields the same event shapes as `run_agent_stream`, plus a single
    `afc_summary` event after the SDK finishes. No per-tool live events
    because AFC does not expose them through `chats.create`."""
    yield {"type": "status", "text": "Thinking..."}
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            tools=[
                web_search_afc,
                calculator_afc,
                save_fact_afc,
                recall_facts_afc,
            ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=10,
            ),
        ),
    )
    response = chat.send_message(user_message)

    # Extract the call history. `response.automatic_function_calling_history`
    # is a list of Content objects, each with `.parts`. Each part that
    # represents a function call has `.function_call`.
    calls: list[dict] = []
    history = getattr(response, "automatic_function_calling_history", None) or []
    for content in history:
        for part in getattr(content, "parts", None) or []:
            fc = getattr(part, "function_call", None)
            if fc is None:
                continue
            calls.append(
                {
                    "name": getattr(fc, "name", "?"),
                    "args": dict(getattr(fc, "args", {}) or {}),
                }
            )

    yield {"type": "afc_summary", "calls": calls}
    try:
        final_text = response.text
    except (ValueError, AttributeError):
        final_text = "_No final text response from the model._"
    yield {"type": "final", "text": final_text or "_No final text response from the model._"}


def run_agent_stream_dispatch(user_message: str, use_afc: bool = False) -> Generator[dict, None, None]:
    """Dispatch to manual or AFC loop based on `use_afc`. Frontends call this
    one entry point so the toggle lives in exactly one place here."""
    if use_afc:
        yield from run_agent_stream_afc(user_message)
    else:
        yield from run_agent_stream(user_message)