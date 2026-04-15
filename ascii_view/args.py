"""Command-line argument parsing module."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.

    Example:
        python3 main.py image.jpg -mw 100 -mh 50 -et 0.3 -r 50
    """
    parser = argparse.ArgumentParser(
        description='ASCII Image Viewer with Linear Algebra pipeline',
        epilog='Example: python3 main.py image.jpg -mw 100 -mh 50 -et 0.3 -r 50'
    )
    parser.add_argument(
        'image_path',
        help='Path to input image'
    )
    parser.add_argument(
        '-mw', '--max-width',
        type=int,
        default=None,
        help='Maximum width in characters'
    )
    parser.add_argument(
        '-mh', '--max-height',
        type=int,
        default=None,
        help='Maximum height in characters'
    )
    parser.add_argument(
        '-et', '--edge-threshold',
        type=float,
        default=0.5,
        help='Edge enhancement threshold (0.0-1.0)'
    )
    parser.add_argument(
        '-cr', '--char-ratio',
        type=float,
        default=2.0,
        help='Character aspect ratio (height/width)'
    )
    parser.add_argument(
        '-r', '--rank',
        type=int,
        default=50,
        help='SVD rank for compression'
    )

    return parser.parse_args()
