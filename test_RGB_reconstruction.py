from PIL import Image
import os
import glob

def build_snake_grid_top_left(cols=4, rows=4):
    mapping = {}
    tile_idx = 0
    for row in range(rows):
        col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
        for col in col_range:
            mapping[tile_idx] = (row, col)
            tile_idx += 1
    return mapping

def find_tiles(tile_dir, num_tiles):
    patterns = [
        "snapshot_{}.png", "snapshot_{:02d}.png",
        "tile_{}.png",     "tile_{:02d}.png",
        "{}.png",          "{:02d}.png",
    ]
    for pattern in patterns:
        found = {
            idx: os.path.join(tile_dir, pattern.format(idx))
            for idx in range(num_tiles)
            if os.path.exists(os.path.join(tile_dir, pattern.format(idx)))
        }
        if len(found) == num_tiles:
            return found
    all_pngs = sorted(glob.glob(os.path.join(tile_dir, "*.png")))
    if len(all_pngs) >= num_tiles:
        return {idx: all_pngs[idx] for idx in range(num_tiles)}
    raise FileNotFoundError(f"Could not find {num_tiles} tiles in '{tile_dir}'.")

def reconstruct_snake_image(tile_dir, output_path=None, cols=4, rows=4, rotation_deg=0.0):
    tile_dir = os.path.abspath(tile_dir)
    if output_path is None:
        output_path = os.path.join(os.path.dirname(tile_dir), "reconstructed.png")

    num_tiles = cols * rows
    mapping   = build_snake_grid_top_left(cols, rows)
    tile_paths = find_tiles(tile_dir, num_tiles)

    tiles = {}
    tile_w = tile_h = None
    for idx in range(num_tiles):
        try:
            img = Image.open(tile_paths[idx]).convert("RGB")
            if tile_w is None:
                tile_w, tile_h = img.size
            tiles[idx] = img
        except Exception:
            tiles[idx] = None

    if tile_w is None:
        raise RuntimeError("No tiles could be loaded.")

    def make_placeholder(w, h):
        ph = Image.new("RGB", (w, h), (40, 40, 40))
        block = 20
        for y in range(0, h, block):
            for x in range(0, w, block):
                if (x // block + y // block) % 2 == 0:
                    for py in range(y, min(y + block, h)):
                        for px in range(x, min(x + block, w)):
                            ph.putpixel((px, py), (80, 80, 80))
        return ph

    canvas = Image.new("RGB", (tile_w * cols, tile_h * rows))
    for idx, (row, col) in mapping.items():
        tile = tiles[idx] if tiles[idx] is not None else make_placeholder(tile_w, tile_h)
        canvas.paste(tile, (col * tile_w, row * tile_h))

    if rotation_deg != 0.0:
        print(f"Applying rotation: {rotation_deg}°")
        canvas = canvas.rotate(rotation_deg, expand=True, resample=Image.BICUBIC)

    canvas.save(output_path)
    print(f"aved → {output_path}  ({canvas.width}×{canvas.height} px)")
    return canvas


# ── Edit these ──────────────────────────────────────────────
TILE_DIR     = r"E:\Lennon_Camera_Project\re\test_FOV_secondary_camera\delta_y\y_is_0.656\re\re_re\secondary_camera"
OUTPUT_PATH  = os.path.join(os.path.dirname(TILE_DIR), "RGB_CAMERA_reconstructed.png")
COLS         = 4
ROWS         = 3
ROTATION_DEG = 5.2
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    reconstruct_snake_image(TILE_DIR, OUTPUT_PATH, COLS, ROWS, ROTATION_DEG)