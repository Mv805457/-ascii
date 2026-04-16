import cv2
import numpy as np

# Use an extended detailed ramp for better visuals
RAMPS = [
    " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@", # 0: Detailed
    " .:-=+*#%@", # 1: Standard
    " ░▒▓█", # 2: Blocks
    " 01", # 3: Binary
    "  _.,-+=*%#@" # 4: Matrix/Terminal
]
EDGE_CHARS = ['|', '\\', '_', '/']

# Create mapping dictionary and arrays globally so they run once
unique_chars = set()
for r in RAMPS:
    unique_chars.update(list(r))
unique_chars.update(EDGE_CHARS)

ALL_CHARS = list(unique_chars)
CHAR_MAP = {c: i for i, c in enumerate(ALL_CHARS)}

def create_char_bank(char_list, font_scale=0.35, thickness=1, char_w=6, char_h=10):
    """Pre-render character masks to a tensor for extremely fast numpy tiling."""
    bank = np.zeros((len(char_list), char_h, char_w, 3), dtype=np.float32)
    
    for i, c in enumerate(char_list):
        img = np.zeros((char_h, char_w, 3), dtype=np.uint8)
        (text_w, text_h), baseline = cv2.getTextSize(c, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Center horizontally, text_h+baseline vertically
        x = (char_w - text_w) // 2
        y = char_h - 2
        
        cv2.putText(img, c, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Store as float matrix [0, 1] for later color multiplication
        bank[i] = img.astype(np.float32) / 255.0
        
    return bank

CHAR_BANK = create_char_bank(ALL_CHARS)

def ensure_custom_chars(custom_str):
    """Dynamically add unrendered characters to the master bank."""
    global ALL_CHARS, CHAR_MAP, CHAR_BANK
    missing = [c for c in custom_str if c not in CHAR_MAP]
    if missing:
        start_idx = len(ALL_CHARS)
        ALL_CHARS.extend(missing)
        for i, c in enumerate(missing):
            CHAR_MAP[c] = start_idx + i
        new_bank = create_char_bank(missing)
        CHAR_BANK = np.concatenate([CHAR_BANK, new_bank], axis=0)

def render_cv_fast(image, combined, edges=None, edge_dirs=None, chars=RAMPS[0], use_edges=True, filter_idx=0):
    height, width = combined.shape
    
    n_chars_ramp = len(chars)
    cmin = float(combined.min())
    cmax = float(combined.max())
    span = cmax - cmin if cmax - cmin > 1e-10 else 1.0

    # 1. Base intensity mapping (0 to len(ramp)-1)
    # Ensure chars are in our global map
    base_indices = np.clip(
        ((combined - cmin) / span * (n_chars_ramp - 1)).astype(int),
        0, n_chars_ramp - 1,
    )
    
    # Map ramp indices to global ALL_CHARS indices
    ramp_to_global = np.array([CHAR_MAP[c] for c in chars])
    global_indices = ramp_to_global[base_indices]

    # 2. Overwrite with edges if strong enough
    if use_edges and edges is not None and edge_dirs is not None:
        edge_min = float(edges.min())
        edge_max = float(edges.max())
        edge_span = edge_max - edge_min + 1e-10
        edge_norm = (edges - edge_min) / edge_span
        
        dir_to_global = np.array([CHAR_MAP[c] for c in EDGE_CHARS])
        dir_global_grid = dir_to_global[edge_dirs]
        
        # Threshold for edges
        global_indices = np.where(edge_norm > 0.35, dir_global_grid, global_indices)

    # 3. FAST Numpy Tiling
    # CHAR_BANK is shape (N, char_h, char_w, 3)
    # global_indices is shape (H, W)
    # block_grid becomes (H, W, char_h, char_w, 3) representing the mask
    block_grid = CHAR_BANK[global_indices]
    
    # rgb is (H, W, 3)
    # We expand it to (H, W, 1, 1, 3) so it broadcasts across char_h, char_w
    rgb = np.clip(image, 0, 255).astype(np.float32)
    rgb_expanded = rgb[:, :, np.newaxis, np.newaxis, :]
    
    # Apply Filters (OpenCV uses BGR channel order)
    if filter_idx == 1:   # Red
        rgb_expanded[:, :, :, :, 0] = 0   # B
        rgb_expanded[:, :, :, :, 1] = 0   # G
    elif filter_idx == 2: # Orange
        rgb_expanded[:, :, :, :, 0] = 0   # B
        rgb_expanded[:, :, :, :, 1] *= 0.5 # G
    elif filter_idx == 3: # Green
        rgb_expanded[:, :, :, :, 0] = 0   # B
        rgb_expanded[:, :, :, :, 2] = 0   # R
    
    # Multiply color * text mask
    colored_blocks = block_grid * rgb_expanded
    
    # Reshape from (H, W, char_h, char_w, 3) -> (H*char_h, W*char_w, 3)
    # To do this correctly, we swap axes: (H, char_h, W, char_w, 3)
    # Then reshape.
    _, char_h, char_w, _ = CHAR_BANK.shape
    final_image = colored_blocks.transpose(0, 2, 1, 3, 4).reshape((height * char_h, width * char_w, 3))
    
    return final_image.astype(np.uint8)
