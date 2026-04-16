"""HTML ASCII renderer for browser output.

Produces color-rich HTML using <span> elements with inline RGB colors,
using the same character mapping math as the terminal renderer.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Extended character ramp — 70 characters for much finer brightness detail
# ---------------------------------------------------------------------------
ASCII_CHARS = (
    " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
)

# Shorter ramp for dense modes
ASCII_CHARS_DENSE = " .:-=+*#%@█"


def render_html(
    image: np.ndarray,
    combined: np.ndarray,
    chars: str = ASCII_CHARS,
) -> str:
    """Render image as a fully-coloured HTML ASCII block.

    Parameters
    ----------
    image:
        RGB float64 array  (H, W, 3) — used for colour information.
    combined:
        Combined brightness matrix (H, W) in [0, 255] — drives char choice.
    chars:
        Character ramp from darkest to brightest.

    Returns
    -------
    str
        An HTML ``<pre>`` block with one ``<span>`` per pixel.
    """
    height, width, _ = image.shape
    n_chars = len(chars)
    lines: list[str] = []

    # Normalise combined to [0, 1] for index lookup
    cmin = combined.min()
    cmax = combined.max()
    span = cmax - cmin if cmax - cmin > 1e-10 else 1.0
    norm = (combined - cmin) / span          # (H, W) in [0, 1]

    # Clip colour arrays once
    rgb = np.clip(image, 0, 255).astype(np.uint8)

    for i in range(height):
        row_parts: list[str] = []
        for j in range(width):
            brightness = float(norm[i, j])
            idx = int(brightness * (n_chars - 1))
            idx = max(0, min(idx, n_chars - 1))
            char = chars[idx]

            r, g, b = int(rgb[i, j, 0]), int(rgb[i, j, 1]), int(rgb[i, j, 2])
            # Escape HTML-special characters
            if char == ' ':
                row_parts.append(' ')
            elif char == '<':
                row_parts.append(f'<span style="color:rgb({r},{g},{b})">&lt;</span>')
            elif char == '>':
                row_parts.append(f'<span style="color:rgb({r},{g},{b})">&gt;</span>')
            elif char == '&':
                row_parts.append(f'<span style="color:rgb({r},{g},{b})">&amp;</span>')
            else:
                row_parts.append(f'<span style="color:rgb({r},{g},{b})">{char}</span>')

        lines.append(''.join(row_parts))

    inner = '\n'.join(lines)
    return f'<pre id="ascii-output">{inner}</pre>'


def render_html_fast(
    image: np.ndarray,
    combined: np.ndarray,
    chars: str = ASCII_CHARS,
    edges: np.ndarray | None = None,
    edge_dirs: np.ndarray | None = None,
) -> str:
    """Vectorised HTML renderer — avoids Python inner loop for speed.

    Same math as :func:`render_html`; builds the full HTML in one
    string-join pass rather than character-by-character.
    """
    height, width, _ = image.shape
    n_chars = len(chars)

    cmin = float(combined.min())
    cmax = float(combined.max())
    span = cmax - cmin if cmax - cmin > 1e-10 else 1.0

    # Vectorised index computation
    indices = np.clip(
        ((combined - cmin) / span * (n_chars - 1)).astype(int),
        0, n_chars - 1,
    )                                          # (H, W)

    rgb = np.clip(image, 0, 255).astype(np.uint8)

    # char_grid[i,j] = the ASCII character for pixel (i,j)
    chars_arr = np.array(list(chars))
    char_grid = chars_arr[indices]             # (H, W) array of single chars

    if edges is not None and edge_dirs is not None:
        edge_min = float(edges.min())
        edge_max = float(edges.max())
        edge_span = edge_max - edge_min + 1e-10
        edge_norm = (edges - edge_min) / edge_span
        
        edge_chars = np.array(['|', '\\', '_', '/'])
        dir_char_grid = edge_chars[edge_dirs]
        
        # Override characters where the edge is strong enough
        char_grid = np.where(edge_norm > 0.35, dir_char_grid, char_grid)

    lines: list[str] = []
    for i in range(height):
        row_parts: list[str] = []
        for j in range(width):
            ch = char_grid[i, j]
            r, g, b = int(rgb[i, j, 0]), int(rgb[i, j, 1]), int(rgb[i, j, 2])
            if ch == ' ':
                row_parts.append(' ')
            else:
                row_parts.append(f'<span style="color:rgb({r},{g},{b})">{ch}</span>')
        lines.append(''.join(row_parts))

    inner = '\n'.join(lines)
    return f'<pre id="ascii-output">{inner}</pre>'
