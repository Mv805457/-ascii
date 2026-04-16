"""Flask web application for ASCII Art viewer.

Routes
------
GET  /                  — serve the frontend
POST /api/process-image — upload an image → return HTML ASCII art
POST /api/webcam-frame  — post base64 webcam frame → return HTML ASCII art
"""

from __future__ import annotations

import base64
import io
import json
import subprocess

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

from ascii_view.image import to_grayscale
from ascii_view.linalg import combine_matrices, project_matrix, sobel_edges, svd_compress
from ascii_view.web_render import ASCII_CHARS, render_html_fast

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------

def _image_array_to_ascii(
    img_array: np.ndarray,
    width: int = 220,
    height: int = 80,
    rank: int = 50,
    edge_weight: float = 0.4,
) -> str:
    """Run the full linalg pipeline and return an HTML <pre> block.

    Parameters
    ----------
    img_array : ndarray, shape (H, W, 3), uint8
        RGB image.
    width, height : int
        Target character grid dimensions.
    rank : int
        SVD rank for compression.
    edge_weight : float
        Blend weight for edge contribution (0–1).
    """
    # ── Resize ──────────────────────────────────────────────────────────────
    pil = Image.fromarray(img_array).convert("RGB")
    # Correct for character aspect ratio (chars are ~2× taller than wide)
    char_ratio = 2.0
    orig_w, orig_h = pil.size
    scale_w = width / orig_w
    scale_h = height / (orig_h / char_ratio)
    scale = min(scale_w, scale_h, 1.0)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale / char_ratio))
    # Allow upscaling so small uploads still fill the grid
    new_w = min(width, max(new_w, 40))
    new_h = min(height, max(new_h, 20))
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    image_f = np.array(pil, dtype=np.float64)

    # ── Linalg pipeline (unchanged math) ────────────────────────────────────
    grayscale = to_grayscale(image_f)                   # luminance
    compressed, U, S, Vt = svd_compress(grayscale, rank)
    projected, _error = project_matrix(grayscale, U, S, Vt)
    edges, edge_dirs = sobel_edges(projected, return_dirs=True)                      # Sobel magnitudes
    combined = combine_matrices(projected, edges, edge_weight)

    # ── Render ──────────────────────────────────────────────────────────────
    return render_html_fast(image_f, combined, ASCII_CHARS, edges, edge_dirs)


def _decode_base64_image(data_url: str) -> np.ndarray:
    """Decode a base64 data-URL into an RGB uint8 numpy array."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(pil, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/process-image")
def process_image():
    """Accept a file upload and return HTML ASCII art."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    try:
        pil = Image.open(file.stream).convert("RGB")
        img_array = np.array(pil, dtype=np.uint8)
    except Exception as exc:
        return jsonify({"error": f"Cannot open image: {exc}"}), 400

    width  = int(request.form.get("width",  220))
    height = int(request.form.get("height",  80))
    rank   = int(request.form.get("rank",    50))
    ew     = float(request.form.get("edge_weight", 0.4))

    # Clamp to sane values
    width  = max(40, min(width,  400))
    height = max(20, min(height, 200))
    rank   = max(1,  min(rank,   200))
    ew     = max(0.0, min(ew,   1.0))

    html = _image_array_to_ascii(img_array, width, height, rank, ew)
    return jsonify({"html": html, "cols": width, "rows": height})


@app.post("/api/webcam-frame")
def webcam_frame():
    """Accept a base64 frame and return HTML ASCII art (for live feed)."""
    body = request.get_json(force=True, silent=True) or {}
    data_url = body.get("frame", "")
    if not data_url:
        return jsonify({"error": "No frame data"}), 400

    width  = int(body.get("width",  180))
    height = int(body.get("height",  60))
    rank   = int(body.get("rank",    30))
    ew     = float(body.get("edge_weight", 0.4))

    width  = max(40, min(width,  400))
    height = max(20, min(height, 200))
    rank   = max(1,  min(rank,   200))
    ew     = max(0.0, min(ew,   1.0))

    try:
        img_array = _decode_base64_image(data_url)
    except Exception as exc:
        return jsonify({"error": f"Bad frame: {exc}"}), 400

    html = _image_array_to_ascii(img_array, width, height, rank, ew)
    return jsonify({"html": html, "cols": width, "rows": height})


@app.post("/api/spawn-desktop")
def spawn_desktop():
    """Launch the native OpenCV desktop app from the web UI."""
    try:
        # Fire and forget
        subprocess.Popen(["python", "desktop_app.py"])
        return jsonify({"status": "success"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
