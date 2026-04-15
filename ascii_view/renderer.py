"""ASCII rendering and terminal output module."""

import numpy as np


ASCII_CHARS = " .:-=+*#%@"


def map_to_ascii(value: float, chars: str = ASCII_CHARS) -> str:
    """Map brightness value to ASCII character.

    Args:
        value: Brightness value in range [0, 255].
        chars: Character ramp from dark to light.

    Returns:
        Single ASCII character.
    """
    idx = int(value * (len(chars) - 1))
    idx = max(0, min(idx, len(chars) - 1))
    return chars[idx]


def apply_color(r: float, g: float, b: float, char: str) -> str:
    """Apply ANSI escape code for RGB color.

    Args:
        r: Red component (0-255).
        g: Green component (0-255).
        b: Blue component (0-255).
        char: Character to colorize.

    Returns:
        Colorized character string with ANSI codes.
    """
    r = int(np.clip(r, 0, 255))
    g = int(np.clip(g, 0, 255))
    b = int(np.clip(b, 0, 255))

    return f"\033[38;2;{r};{g};{b}m{char}\033[0m"


def render_ascii(
    image: np.ndarray,
    combined: np.ndarray,
    chars: str = ASCII_CHARS
) -> list:
    """Render image as colored ASCII art.

    Args:
        image: RGB image array for color information.
        combined: Combined brightness matrix for character selection.
        chars: Character ramp for brightness mapping.

    Returns:
        List of formatted ASCII lines.
    """
    height, width, _ = image.shape
    lines = []

    for i in range(height):
        line = []
        for j in range(width):
            brightness = combined[i, j]
            char = map_to_ascii(brightness, chars)

            r, g, b = image[i, j]
            colored_char = apply_color(r, g, b, char)
            line.append(colored_char)

        lines.append(''.join(line))

    return lines


def print_debug_info(
    grayscale: np.ndarray,
    edges: np.ndarray,
    combined: np.ndarray,
    rank: int,
    error: float
) -> None:
    """Print debug information to stderr.

    Args:
        grayscale: Grayscale matrix.
        edges: Edge magnitude matrix.
        combined: Combined output matrix.
        rank: SVD rank used.
        error: Reconstruction error.
    """
    print(f"DEBUG: Grayscale - min: {np.min(grayscale):.2f}, max: {np.max(grayscale):.2f}", file=__import__('sys').stderr)
    print(f"DEBUG: SVD rank {rank}, reconstruction error: {error:.2f}", file=__import__('sys').stderr)
    print(f"DEBUG: Edges - min: {np.min(edges):.2f}, max: {np.max(edges):.2f}", file=__import__('sys').stderr)
    print(f"DEBUG: Combined - min: {np.min(combined):.2f}, max: {np.max(combined):.2f}", file=__import__('sys').stderr)
