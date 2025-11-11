"""Generic utility helpers used across notebooks."""

from __future__ import annotations

from difflib import SequenceMatcher
from html import escape

from IPython.display import HTML, display


def show_text_diff(text_a: str, text_b: str) -> None:
    """Render a highlighted side-by-side diff of two strings."""
    sm = SequenceMatcher(None, text_a, text_b)
    a_chunks: list[str] = []
    b_chunks: list[str] = []

    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        seg_a = escape(text_a[a0:a1])
        seg_b = escape(text_b[b0:b1])

        if tag == "equal":
            a_chunks.append(seg_a)
            b_chunks.append(seg_b)
        elif tag == "delete":
            a_chunks.append(f'<span style="background:#ffd6d6;">{seg_a}</span>')
        elif tag == "insert":
            b_chunks.append(f'<span style="background:#d6ffd6;">{seg_b}</span>')
        elif tag == "replace":
            a_chunks.append(f'<span style="background:#ffe5b4;">{seg_a}</span>')
            b_chunks.append(f'<span style="background:#ffe5b4;">{seg_b}</span>')

    html = (
        "<div style=\"display:flex; gap:12px; align-items:flex-start;\">"
        "<span style=\"flex:1; padding:8px; font-family:monospace; white-space:pre-wrap; border:1px solid #ddd;\">"
        f"{''.join(a_chunks)}"
        "</span>"
        "<span style=\"flex:1; padding:8px; font-family:monospace; white-space:pre-wrap; border:1px solid #ddd;\">"
        f"{''.join(b_chunks)}"
        "</span>"
        "</div>"
    )
    display(HTML(html))
