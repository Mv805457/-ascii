#!/usr/bin/env python3
"""
ASCII Image Viewer with Linear Algebra Pipeline
Academic project implementing SVD compression and Sobel edge enhancement

Usage:
    python3 main.py image.jpg [OPTIONS]
    python3 main.py image.jpg -mw 100 -mh 50 -et 0.3 -r 50
"""

import sys
import os

from ascii_view.args import parse_args
from ascii_view.image import load_image, resize_image, to_grayscale
from ascii_view.linalg import svd_compress, project_matrix, sobel_edges, combine_matrices
from ascii_view.renderer import render_ascii, print_debug_info


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        terminal_size = os.get_terminal_size()
        default_width = terminal_size.columns - 2
        default_height = terminal_size.lines - 2
    except OSError:
        default_width = 80
        default_height = 48

    max_width = args.max_width or default_width
    max_height = args.max_height or default_height

    try:
        image = load_image(args.image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)

    image = resize_image(image, max_width, max_height, args.char_ratio)

    grayscale = to_grayscale(image)

    compressed, U, S, Vt = svd_compress(grayscale, args.rank)

    projected, error = project_matrix(grayscale, U, S, Vt)

    edges = sobel_edges(projected)

    combined = combine_matrices(projected, edges, args.edge_threshold)

    print_debug_info(grayscale, edges, combined, len(S), error)

    ascii_lines = render_ascii(image, combined)

    for line in ascii_lines:
        print(line)


if __name__ == '__main__':
    main()
