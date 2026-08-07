"""Rich terminal REPL for the Smart Calculator + Web Search agent.

Run with:  python agent.py

For the Streamlit web UI, see `app.py`.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from core import _detect_mime, run_agent_stream_dispatch

console = Console()

BANNER = r"""
[bold cyan]
   ____                      __      __        _    ____
  / ___|  __ _ _   _ _ __   __\ \    / /_____ _| |_ / ___|
  \___ \ / _` | | | | '_ \  / _ \ \/\/ / _ \ \/ / __\___ \
   ___) | (_| | |_| | | | |  __/\  /\  /  __/>  <| |_ ___) |
  |____/ \__,_|\__,_|_| |_|\___| \/  \/ \___/_/\_\\__|____/
[/bold cyan]
[dim]Calculator + Web Search · powered by Gemini[/dim]
"""


def _header(text: str) -> None:
    console.print(Rule(text, style="cyan", align="left"))


def _render_search_panel(results, query: str) -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column(style="bold cyan", width=3)
    table.add_column(style="bold", ratio=2)
    table.add_column(style="dim", ratio=3)
    table.add_column(ratio=5)
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        link = r.get("href") or r.get("url") or ""
        snippet = r.get("body") or r.get("snippet") or "No description"
        table.add_row(
            f"{i}.",
            Text(title, overflow="ellipsis"),
            Text(link, overflow="ellipsis"),
            Text(snippet, overflow="ellipsis"),
        )
    return Panel(table, title=f"🔎 [bold]{query}[/bold]", border_style="blue")


def _render_tool_call(name: str, args: dict) -> Panel:
    icon = {
        "web_search": "🔎",
        "calculator": "🧮",
        "save_fact": "💾",
        "recall_facts": "🧠",
    }.get(name, "🛠")
    body = Syntax(
        "\n".join(f"{k} = {v!r}" for k, v in args.items()),
        "python",
        theme="monokai",
        word_wrap=True,
        background_color="default",
    )
    return Panel(
        body,
        title=f"{icon} [bold yellow]tool call[/bold yellow]  {name}",
        border_style="yellow",
        title_align="left",
    )


def _render_tool_result(name: str, payload) -> Panel:
    if name == "web_search":
        if payload == "__NO_RESULTS__":
            return Panel(
                "[yellow]No results returned by the search backend.[/yellow]",
                title="⚠  empty result",
                border_style="yellow",
            )
        if isinstance(payload, str):
            return Panel(f"[red]{payload}[/red]", title="⚠  search error", border_style="red")
        return _render_search_panel(payload, query="…")
    if name == "calculator":
        if isinstance(payload, str) and payload.startswith("Error"):
            return Panel(f"[red]{payload}[/red]", title="⚠  calculator error", border_style="red")
        return Panel(
            Syntax(str(payload), "python", theme="monokai", background_color="default"),
            title="[bold green]result[/bold green]",
            border_style="green",
        )
    if name == "save_fact":
        return Panel(
            f"[green]{payload}[/green]",
            title="💾 [bold green]saved[/bold green]",
            border_style="green",
        )
    if name == "recall_facts":
        if payload == "No matching facts.":
            return Panel(
                "[yellow]No matching facts.[/yellow]",
                title="🧠 [bold]recall[/bold]",
                border_style="yellow",
            )
        return Panel(
            payload,
            title="🧠 [bold]recalled[/bold]",
            border_style="magenta",
        )
    return Panel(str(payload), title="result")


def _handle(event: dict) -> None:
    kind = event["type"]
    if kind == "status":
        # Status updates are picked up by the spinner context manager.
        return
    if kind == "tool_call":
        _header(f"tool · {event['name']}")
        console.print(_render_tool_call(event["name"], event["args"]))
    elif kind == "tool_result":
        console.print(_render_tool_result(event["name"], event["payload"]))
    elif kind == "afc_summary":
        calls = event.get("calls", [])
        if not calls:
            body = "[dim]No tool calls were made.[/dim]"
        else:
            lines = [
                f"[bold]{len(calls)} call(s) handled automatically by the SDK:[/bold]"
            ]
            for c in calls:
                arg_str = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
                lines.append(f"  • [cyan]{c['name']}[/cyan]([dim]{arg_str}[/dim])")
            body = "\n".join(lines)
        console.print(
            Panel(
                body,
                title="⚡ [bold magenta]afc[/bold magenta]",
                border_style="magenta",
            )
        )
    elif kind == "final":
        console.print(
            Panel(
                Markdown(event["text"]),
                title="🤖 [bold cyan]agent[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
    elif kind == "aborted":
        console.print(
            Panel(
                f"[yellow]{event['text']}[/yellow]",
                title="🤖 [bold]agent[/bold]",
                border_style="cyan",
            )
        )


def _render_attached_images(images: list[bytes]) -> None:
    """Render each attached image as a small panel before the agent responds.

    Uses `rich.pil.Image` for terminal rendering (SIXEL / iTerm / Kitty, depends
    on the terminal). If Pillow isn't installed, fall back to a text note with
    sizes — the prompt still works either way."""
    try:
        from rich.pil import Image as PilImage  # Pillow-backed widget
    except ImportError:
        sizes = ", ".join(f"{len(b) // 1024} KiB" for b in images)
        console.print(
            Panel(
                f"[dim]{len(images)} attached image(s) ({sizes}). "
                f"Install Pillow to render thumbnails in the terminal.[/dim]",
                title="📎 [bold]attached[/bold]",
                border_style="cyan",
            )
        )
        return
    for img_bytes in images:
        try:
            console.print(
                Panel(
                    PilImage.from_bytes(io.BytesIO(img_bytes), width=40),
                    border_style="cyan",
                    title="📎 [bold]attached[/bold]",
                )
            )
        except Exception as exc:
            console.print(
                Panel(
                    f"[red]Could not render image: {exc}[/red]",
                    title="📎 [bold red]attached (error)[/bold red]",
                    border_style="red",
                )
            )


def run_agent(
    user_message: str, use_afc: bool = False, images: list[bytes] | None = None
) -> None:
    if images:
        _render_attached_images(images)
    with console.status("[bold cyan]Thinking…[/bold cyan]", spinner="dots") as status:
        for event in run_agent_stream_dispatch(
            user_message, use_afc=use_afc, images=images
        ):
            if event["type"] == "status":
                status.update(f"[bold cyan]{event['text']}[/bold cyan]")
                continue
            status.stop()
            _handle(event)
            if event["type"] in ("final", "aborted"):
                return
            # Re-enter status context for the next model call
            status.start()
            status.update("[bold cyan]Thinking…[/bold cyan]")


def main() -> None:
    use_afc = os.environ.get("SMARTCAL_AFC") == "1"
    console.print(BANNER)
    mode = "[bold magenta]automatic function calling[/bold magenta]" if use_afc else "[dim]manual tool dispatch[/dim]"
    console.print(
        f"[dim]Type a question and press Enter. Use [bold]/attach <path>[/bold] "
        f"to add an image to your next prompt. Type [bold]exit[/bold] or "
        f"[bold]quit[/bold] to leave. Mode: {mode}[/dim]\n"
    )

    pending_images: list[bytes] = []

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            return
        if user_input.lower() in {"exit", "quit"}:
            console.print("[dim]Goodbye.[/dim]")
            return
        if not user_input:
            continue

        # Slash commands.
        if user_input.startswith("/attach"):
            path_str = user_input[len("/attach"):].strip()
            if not path_str:
                console.print("[red]Usage: /attach <path-to-image>[/red]")
                continue
            try:
                data = Path(path_str).read_bytes()
                _detect_mime(data)  # raises ValueError for unsupported formats
                pending_images.append(data)
                console.print(
                    f"[green]Attached {path_str} "
                    f"({len(data) // 1024} KiB). It will be sent with your "
                    f"next prompt.[/green]"
                )
            except FileNotFoundError:
                console.print(f"[red]File not found: {path_str}[/red]")
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
            continue

        # Consume any attached images and send.
        images = pending_images
        pending_images = []
        run_agent(user_input, use_afc=use_afc, images=images)
        console.print()


if __name__ == "__main__":
    main()