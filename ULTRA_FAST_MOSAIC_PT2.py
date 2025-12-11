
import time, re
from pathlib import Path
import numpy as np
from natsort import natsorted
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from collections.abc import Iterable
import os

# =========================================================
#                     GLOBAL SETTINGS
# =========================================================

# Root input folder
#FOLDER = Path(r'E:\Users\Ben\Programs\HSI Programs\z\Data\Chicago_5x5_test\Chicago_Slides_11_19_25\Real\Hyper005_I_stained\Bright_With_Fluoro_Filters')
FOLDER = Path(r'E:\Users\Ben\Programs\HSI Programs\z\Data\Chicago_Cancer\Chicago_5x5_test\Chicago_Slides_11_19_25\Real\Hyper005_II_unstained\Fluoro')
FOLDER = Path(r'\\DEEPSPEC\datapipeline\Test_if_savable_from_other_computer\debug_issue_with_getposition_func_nonetype\re\REAL_20_rows_20_columns')
#FOLDER = Path(r'E:\Users\Ben\Programs\HSI Programs\z\Data\Chicago_5x5_test\Chicago_Slides_11_19_25\Real\Hyper005_II_unstained\Bright_With_Fluoro_Filters')
#FOLDER = Path(r'E:\Users\Ben\Programs\HSI Programs\z\Data\Chicago_5x5_test\Chicago_Slides_11_19_25\Real\Hyper005_II_unstained\Fluoro')
FOLDER = Path(r'\\DEEPSPEC\datapipeline\Test_if_savable_from_other_computer\debug_issue_with_getposition_func_nonetype\re\REAL_20_rows_20_columns\re')
FOLDER = Path(r'\\DEEPSPEC\\datapipeline\Test_if_savable_from_other_computer\debug_issue_with_getposition_func_nonetype\re\REAL_20_rows_20_columns\Hyper001_stained\Brightfield_re_with_kohler')

# ultra_fast_mosaic_pt2.py



#FOLDERS = {'E:\Users\Ben\Programs\HSI Programs\z\Data\Chicago_5x5_test\Chicago_Slides_11_19_25\Real\Hyper005_I_stained\Bright_With_Fluoro_Filters',
    
#}

# File pattern
INPUT_PREFIX = "*_cube_y*_x*.npy"
output_suffix = lambda Pattern, String: re.findall(pattern=Pattern + r'$', string=String)
OUT_PREFIX   = "mosaic_fast"

JPEG_QUALITY = 75
DOWNSAMPLE   = 1

# Color tuning
P_LOW, P_HIGH = 1, 100
BLUE_SCALE = 0.2
GAMMA      = 1.0

# Band selection
BAND_PICK_MODE = "fixed"   # "spread", "one_tile_var", "fixed"
FIXED_BANDS    = (31, 34, 37)
MIN_BAND_GAP   = 3

# Percentile sampling
PERC_MODE        = "multi"  # "first_tile", "multi"
PERC_SAMPLE_MAX  = 16
VAR_STRIDE       = 16

# Output: "rgb" or "grayscale"
OUTPUT_MODE = "rgb"


# ---------------------------------------------------------
# User-selectable grid size
# ---------------------------------------------------------
GRID_MODE = "auto"      # "auto" = detect from filenames
                          # "manual" = use GRID_ROWS × GRID_COLS

GRID_ROWS = 20          # Used only when GRID_MODE="manual"
GRID_COLS = 20            # Used only when GRID_MODE="manual"

# Behavior when manual grid is larger than available tiles
FILL_MODE = "repeat"      # "repeat" = nearest tile
                          # "blank"  = empty tile

# =========================================================



# ──────────────────────────────────────────────────────────
# Helpers: file discovery
# ──────────────────────────────────────────────────────────
def find_tiles(folder: Path):
    paths = natsorted(folder.glob(INPUT_PREFIX))
    if not paths:
        raise FileNotFoundError(f"No {INPUT_PREFIX} found in {folder}")
    return paths


def parse_xy(name: str):
    m = re.search(r"_y(\d+)_x(\d+)\.npy$", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def ceil_div(n, d):
    return (int(n) + int(d) - 1) // int(d)


# ──────────────────────────────────────────────────────────
# Band picking
# ──────────────────────────────────────────────────────────
def simple_pick_bands(paths, min_gap=3):
    first = paths[0]

    if BAND_PICK_MODE == "fixed":
        picks = list(FIXED_BANDS)
        nb = int(np.load(first, mmap_mode='r').shape[0])
        return picks

    arr0 = np.load(first, mmap_mode='r')
    nb = int(arr0.shape[0])

    if BAND_PICK_MODE == "spread":
        i0 = max(0, min(nb-1, int(round(nb * 0.10))))
        i1 = max(0, min(nb-1, int(round(nb * 0.50))))
        i2 = max(0, min(nb-1, int(round(nb * 0.90))))
        picks = [i0, i1, i2]

    elif BAND_PICK_MODE == "one_tile_var":
        ch = arr0[:, ::VAR_STRIDE, ::VAR_STRIDE]
        var = ch.reshape(nb, -1).var(axis=1)
        thirds = [
            (0, max(1, nb//3)-1),
            (nb//3, max(nb//3, (2*nb)//3)-1),
            ((2*nb)//3, nb-1),
        ]
        picks = []
        for lo, hi in thirds:
            if hi < lo:
                continue
            seg = var[lo:hi+1]
            picks.append(lo + int(seg.argmax()))
    else:
        picks = [0, nb//2, nb-1]

    # spacing
    picks = sorted(set(picks))
    if len(picks) < 3:
        for cand in [0, nb-1, nb//2]:
            if all(abs(cand - k) >= min_gap for k in picks):
                picks.append(cand)
            if len(picks) == 3:
                break
    picks = sorted(picks[:3])

    if picks[1] - picks[0] < min_gap:
        picks[1] = min(nb-1, picks[0] + min_gap)
    if picks[2] - picks[1] < min_gap:
        picks[2] = min(nb-1, picks[1] + min_gap)

    picks = tuple(sorted(picks))
    print(f"[ULTRA] Picked RGB bands (mode={BAND_PICK_MODE}): {picks} (of {nb})")
    return picks


# ──────────────────────────────────────────────────────────
# Percentile sampling
# ──────────────────────────────────────────────────────────
def quick_percentiles(paths, bands_idx):
    if PERC_MODE == "first_tile":
        cube = np.load(paths[0], mmap_mode='r')
        p1 = np.zeros(3, dtype=np.float32)
        p9 = np.zeros(3, dtype=np.float32)
        for c, b in enumerate(bands_idx):
            ch = cube[b][::VAR_STRIDE, ::VAR_STRIDE].ravel()
            p1[c] = np.percentile(ch, P_LOW)
            p9[c] = np.percentile(ch, P_HIGH)
        print(f"[ULTRA] p{P_LOW}/{P_HIGH} (first tile): {p1.round(3)} / {p9.round(3)}")
        return p1, p9

    # Multi-tile
    p1 = np.zeros(3, dtype=np.float32)
    p9 = np.zeros(3, dtype=np.float32)
    cnt = 0

    for p in paths[:PERC_SAMPLE_MAX]:
        try:
            cube = np.load(p, mmap_mode='r')
            for c, b in enumerate(bands_idx):
                chs = cube[b][::VAR_STRIDE, ::VAR_STRIDE].ravel()
                p1[c] += np.percentile(chs, P_LOW)
                p9[c] += np.percentile(chs, P_HIGH)
            cnt += 1
        except Exception:
            continue

    if cnt == 0:
        print("[ULTRA] No percentile samples; using defaults.")
        return np.array([0, 0, 0], np.float32), np.array([1, 1, 1], np.float32)

    p1 /= cnt
    p9 /= cnt
    print(f"[ULTRA] p{P_LOW}/{P_HIGH} (multi): {p1.round(3)} / {p9.round(3)}")
    return p1, p9


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


# ──────────────────────────────────────────────────────────
# Grid + tile sizing (UPDATED FOR MANUAL GRID)
# ──────────────────────────────────────────────────────────
def load_grid_info(paths):
    coords, by_xy = [], {}
    ref_h = ref_w = None

    for p in paths:
        xy = parse_xy(p.name)
        if xy is None:
            continue

        by_xy[xy] = p
        coords.append(xy)

        if ref_h is None:
            arr = np.load(p, mmap_mode='r')
            ref_h = arr.shape[1] // DOWNSAMPLE
            ref_w = arr.shape[2] // DOWNSAMPLE

    if not coords:
        raise RuntimeError("Could not find y/x indices in filenames.")

    # Real dimensions
    real_rows = max(y for y, _ in coords) + 1
    real_cols = max(x for _, x in coords) + 1

    # Manual override
    if GRID_MODE == "manual":
        rows = GRID_ROWS
        cols = GRID_COLS
        print(f"[FAST] Grid override: using {rows}x{cols} (real is {real_rows}x{real_cols})")
    else:
        rows = real_rows
        cols = real_cols
        print(f"[FAST] Grid auto-detected: {rows}x{cols}")

    print(f"[FAST] Tile size: {ref_w}x{ref_h} (downsample={DOWNSAMPLE})")
    return coords, by_xy, rows, cols, ref_h, ref_w


# ──────────────────────────────────────────────────────────
# Folder-caption helper
# ──────────────────────────────────────────────────────────
def last_three_folders(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 3:
        return " / ".join(parts[-3:])
    elif len(parts) >= 1:
        return " / ".join(parts)
    else:
        return ""


# ──────────────────────────────────────────────────────────
# Label draw (with red folder text)
# ──────────────────────────────────────────────────────────
def add_label(img: np.ndarray, label: str, folder_info: str, font_size: int = 72) -> np.ndarray:
    img = img.astype(np.uint8, copy=False)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Red header: last 3 folders
    red = (255, 0, 0)
    draw.text((20, 20), folder_info, fill=red, font=font)

    # Secondary label in contrasting color near top-left but lower
    if img.ndim == 3:
        pr, pg, pb = map(int, img[-1, -1])
    else:
        v = int(img[-1, -1])
        pr = pg = pb = v
    contrast = (255 - pr, 255 - pg, 255 - pb)
    red = (255, 80, 80)       # softer red
    draw.text((20, 60), label, fill=contrast, font=font)

    return np.array(pil)


# ──────────────────────────────────────────────────────────
# Hyperspectral → RGB / grayscale
# ──────────────────────────────────────────────────────────
def hyperspectral_to_rgb(cube, bands_idx, p1, p9):
    chans = []
    for c, b in enumerate(bands_idx):
        ch = cube[b][::DOWNSAMPLE, ::DOWNSAMPLE]
        norm = (ch - p1[c]) / (p9[c] - p1[c] + 1e-8)
        chans.append(clamp01(norm))

    if OUTPUT_MODE == "grayscale":
        arr = np.mean(chans, axis=0)
        return (arr * 255).astype(np.uint8)

    arr = np.stack(chans, axis=2)
    return (arr * 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────
# Build mosaic (supports manual grid + S-pattern variants)
# ──────────────────────────────────────────────────────────
def build_mosaic_variant(
    by_xy,
    rows,
    cols,
    ref_h,
    ref_w,
    bands_idx,
    p1,
    p9,
    *,
    snake_rows: bool = False,
    flip_tiles: bool = False
):
    """
    snake_rows: if True, every odd row uses reversed cube order:
                visual x -> src_x = cols-1-x
    flip_tiles: if True, every odd row's tiles are horizontally flipped
                (np.fliplr) before being placed.
    """
    if OUTPUT_MODE == "grayscale":
        mosaic = np.zeros((rows * ref_h, cols * ref_w), dtype=np.uint8)
    else:
        mosaic = np.zeros((rows * ref_h, cols * ref_w, 3), dtype=np.uint8)

    real_coords = list(by_xy.keys())

    for y in range(rows):
        row_reversed = (y % 2 == 1) and snake_rows

        for x in range(cols):
            # visual column index in the output
            if row_reversed:
                # e.g. for cols=5: 0→4, 1→3, 2→2, ...
                src_x = cols - 1 - x
            else:
                src_x = x

            key = (y, src_x)
            if key in by_xy:
                path = by_xy[key]
            else:
                if FILL_MODE == "blank":
                    continue

                # Nearest available tile (for manual larger grids)
                nearest = min(
                    real_coords,
                    key=lambda p: abs(p[0] - y) + abs(p[1] - src_x)
                )
                path = by_xy[nearest]

            cube = np.load(path, mmap_mode='r')
            tile_img = hyperspectral_to_rgb(cube, bands_idx, p1, p9)

            # resize if misaligned
            if tile_img.shape[0] != ref_h or tile_img.shape[1] != ref_w:
                tile_img = np.array(
                    Image.fromarray(tile_img).resize((ref_w, ref_h), Image.LANCZOS)
                )

            # For odd rows, optionally flip tiles horizontally
            if (y % 2 == 1) and flip_tiles:
                tile_img = np.fliplr(tile_img)

            y0, x0 = y * ref_h, x * ref_w

            if OUTPUT_MODE == "grayscale":
                mosaic[y0:y0 + ref_h, x0:x0 + ref_w] = tile_img
            else:
                mosaic[y0:y0 + ref_h, x0:x0 + ref_w, :] = tile_img

    return mosaic


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    paths = find_tiles(FOLDER)
    coords, by_xy, rows, cols, ref_h, ref_w = load_grid_info(paths)

    bands_idx = simple_pick_bands(paths)
    p1, p9 = quick_percentiles(paths, bands_idx)

    # All permutations you asked for on every other row:
    #   - snake_rows ∈ {False, True}
    #   - flip_tiles ∈ {False, True}
    variants = [
        (True,  True,  "snake_and_flip"),
        (False, False, "normal"),
        (True,  False, "snake_only"),
        (False, True,  "flip_only"),
        
    ]

    folder_text = last_three_folders(FOLDER)

    for snake_rows, flip_tiles, tag in variants:
        print(f"[FAST] Building variant: {tag} (snake_rows={snake_rows}, flip_tiles={flip_tiles})")

        mosaic = build_mosaic_variant(
            by_xy, rows, cols, ref_h, ref_w,
            bands_idx, p1, p9,
            snake_rows=snake_rows,
            flip_tiles=flip_tiles,
        )

        # always make RGB for labeling, then optionally drop back to single channel
        if OUTPUT_MODE == "grayscale":
            label_img = np.stack([mosaic] * 3, axis=2)
        else:
            label_img = mosaic

        label_text = (
            f"Ultra-fast mosaic {rows}x{cols} "
            f"(mode={OUTPUT_MODE}, snake_rows={snake_rows}, flip_tiles={flip_tiles})"
        )
        label_img = add_label(label_img, label_text, folder_text)

        if OUTPUT_MODE == "grayscale":
            label_img = label_img[:, :, 0]

        out_file = FOLDER / f"{OUT_PREFIX}_{OUTPUT_MODE}_{tag}.png"
        Image.fromarray(label_img).save(out_file, quality=JPEG_QUALITY)
        print(f"[FAST] Saved variant '{tag}' -> {out_file}")

    print(f"[FAST] Done all variants in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
