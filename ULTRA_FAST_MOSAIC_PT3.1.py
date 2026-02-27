#ULTRA_FAST_MOSAIC_PT3.1
import time
from pathlib import Path
import numpy as np
from natsort import natsorted
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import os


#######!~~~~~~~~~ ITEMS THAT CAN REALISTICALLY CHANGE~~~~~~~~~~~~ !#########
FOLDER = Path(r'\\DEEPSPEC\\datapipeline\Test_if_savable_from_other_computer\debug_issue_with_getposition_func_nonetype\re\Re_re_test_liao_methylene_blue_660_nm_and_PL17\Real\675_nm_exicitation_re_re_re_re_jan_14')
FOLDER = Path(r'\\DEEPSPEC\\datapipeline\Data\Chicago_Cancer\Chicago_Slides_1_20_26\placeholder\placeholder\brightfield')
FOLDER = Path(r'Z:\Data\Chicago_Cancer\Chicago_Slides_1_20_26\placeholder\placeholder\1x_brightfield')
FOLDER = Path(r'Z:\Data\Chicago_Cancer\Chicago_Slides_1_20_26\placeholder\placeholder\1x_brightfield\20x')
FOLDER = Path(r'Z:\Data\Chicago_Cancer\Chicago_Slides_1_20_26\placeholder\placeholder\1x_brightfield\20x\with_1.02_delta_y\10x')
FOLDER = Path(r'Z:\Data\Chicago_Cancer\Chicago_Slides_1_20_26\placeholder\placeholder\1x_brightfield\20x\with_1.02_delta_y\10x\with_0.8_delta_y_10x\with_0.85_delta_y_10x\with_0.825_delta_y_10x')
FOLDER = Path(r'Z:\Data\Chicago_Cancer\Chicago_Slides_1_20_26\placeholder\placeholder\1x_brightfield\20x\with_1.02_delta_y\10x\with_0.8_delta_y_10x\with_0.85_delta_y_10x\with_0.825_delta_y_10x\with_0.835_delta_y_10x\0.84_final\0.8375\0.8325')
FOLDER = Path (r'Z:\Data\Chicago_Cancer\Chicago_Slides_1_20_26\placeholder\placeholder\1x_brightfield\20x\with_1.02_delta_y\0.8425')
FOLDER = Path(r'Z:\Data\Chicago_Cancer\Chiacgo_Slides_1_22_26\Brightfield\10x_mag\Hyper1_1_Stained')
FOLDER = Path(r'Z:\Test_if_savable_from_other_computer\debug_issue_with_getposition_func_nonetype\re\Re_re_test_liao_methylene_blue_660_nm_and_PL17\Real\745_nm_excitation_jan_23\re_hit_wrong_button')
FOLDER = Path(r'Z:\Test_if_savable_from_other_computer\debug_issue_with_getposition_func_nonetype\re\Re_re_test_liao_methylene_blue_660_nm_and_PL17\Real\745_nm_excitation_jan_23\re_hit_wrong_button\looks_like_zeros_for_data_retry_higher_gain\real_last_750_ms_inte')
FOLDER = Path(r'Z:\Data\Maglov\Test\Fluoro')
FOLDER = Path(r'Z:\Data\Chicago_Cancer\Chiacgo_Slides_1_22_26\Brightfield\10x_mag\Hyper1_2_Unstained')
FOLDER = Path(r'Z:\Data\Chicago_Cancer\Chiacgo_Slides_1_22_26\Brightfield\10x_mag\Hyper1_2_Unstained\fixed_with_kohler')
FOLDER = Path(r'Z:\Data\Chicago_Cancer\Chiacgo_Slides_1_22_26\Brightfield\10x_mag\Hyper1_2_Unstained\fixed_kohler_but_no_filters')
#FOLDER = Path(r'Z:\Data\Chicago_Cancer\Chiacgo_Slides_1_22_26\Brightfield\10x_mag\Hyper1_1_Stained')
FOLDER = Path(r'Z:\Data\Chicago_Cancer\Chiacgo_Slides_1_22_26\Fluoro\Hyper1_1_Stained')
class Mode:
    BRIGHTFIELD = False
    FLUORO = False

M = Mode()

# short alias names:
b = "BRIGHTFIELD"
f = "FLUORO"

use_attribute = f
setattr(M, use_attribute, True)

if   M.FLUORO:
    #FIXED_BANDS = (85, 95, 105) #output based 660 nm input
    FIXED_BANDS = (103, 106, 109) #output based 660 nm input
    FIXED_BANDS =  (40, 50, 60)

elif M.BRIGHTFIELD:
    FIXED_BANDS =  (40, 50, 60)#(20, 25, 30)#(30, 35, 40)
else:
    FIXED_BANDS = None
print(f"Using scan type: {use_attribute} = {getattr(M, use_attribute)}")
SUPERGRID_ENABLED = False

LABEL_MODE = False

GRID_MODE = "auto" #"manual"
GRID_ROWS = 20
GRID_COLS = 20


#######!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ !#########

TEST_MIN = 0
TEST_MAX = 30
TEST_MANUAL_CONTRAST: bool = True

if TEST_MANUAL_CONTRAST:
    from typing import TypeAlias, Final, Literal, TypeVar
    #np_8,np_16, np_64: TypeAlias = np.uint8, np.uint16, np.uint64
    #np_8: TypeAlias = np.uint8
    #UnsignedInts: Final = tuple[type[np.uint8], type[np.uint16], type[np.uint64]]
    #UNSIGNED_INT_SCHEMAS: TypeAlias = Literal["8", "16", "64"]
    #intTSchema = TypeVar("TSchema", bound=UNSIGNED_INT_SCHEMAS)
    Schema: TypeAlias = Literal["8", "16", "64"]

    SchemaToDType = {
        "8": np.uint8,
        "16": np.uint16,
        "64": np.uint64,
    }



GET_EVERY_CUBE_RAW_MIN_MAX_VALUES: bool = False

OUT_PREFIX = "mosaic_fast_v_31"
PNG_QUALITY = 100
DOWNSAMPLE = 1

P_LOW, P_HIGH = 1, 99        #percentile; probably want to throw out top 10 or 20 % of values for brightfield if theres no sample being imaged
GAMMA = 1.0
BAND_PICK_MODE = "fixed"

MIN_BAND_GAP = 5
PERC_MODE = "multi"
RATIO_TOTAL_CUBES_FOR_CONTRAST = 0.9 #i.e. if = 0.75; then first 75% of cubes in sorted path list used to sample percentile for contrast
VAR_STRIDE = 5 #length of each step inbetween each contrast sample per cube's band

OUTPUT_MODE = "grayscale" #"RGB" #; anything but <"grayscale>"" is RGB

FILL_MODE = "blank"

# ---------------------------------------------------------
# NEW: S-pattern (snake) support
# ---------------------------------------------------------
# If True, every other row is assumed to have been scanned in
# the opposite X direction. We:
#   • Reverse the tile order in those rows (0→N-1, 1→N-2, etc.)
#   • Flip each tile horizontally so spatial orientation matches.
S_PATTERN = True


def find_tiles(folder: Path):
    paths = natsorted(folder.glob("*.npy"))
    print(f"[FAST] Found {len(paths)} files in {folder}")
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {folder}")
    return paths


def _leading_int(s: str) -> int | None:
    i = 0
    start_dig = 0
    n = len(s)
    if n == 0 or (n==1 and not(s[0].isdigit())):
        return None
    elif n ==1 and (s[0].isdigit()):
        return int(s[0])
    skip_initial_text = True
    while i < n:
        if s[i].isdigit() and skip_initial_text is True:
            skip_initial_text = False
            start_dig = i
            i+=1
        elif not(s[i].isdigit()) and skip_initial_text is False: #signifies new numeric or special character after the first num sequence
            break
        else:
            i+=1

    if skip_initial_text:
        return None

    return int(s[start_dig:i]) #return item right numeric (i-1) right before the non-numeric at index i; slicing cuts off last digit -- alongwith start digit 'a'


def _prefix_sort_key(prefix: str):
    n = _leading_int(prefix)
    if n is not None:
        return (0, n, prefix.lower())
    return (1, prefix.lower())


def _read_int_forward(stem: str, i: int):
    n = len(stem)
    if i >= n or not stem[i].isdigit():
        return None, i
    j = i
    while j < n and stem[j].isdigit():
        j += 1
    return int(stem[i:j]), j


def parse_expected_subgrid_rowcol(name: str) -> tuple[int, int] | None:
    """
    Expected filename example:
      actual_coords_x-0.0001_y-0.0002_expected_none_sub_grid_row_0_col_0_data.npy

    IMPORTANT: We ONLY parse the expected subgrid numbers: sub_grid_row_<int>_col_<int>
    The actual_coords_x/y floats are reference and must NOT affect sorting.
    """
    stem = Path(name).stem

    key_row = "sub_grid_row_"
    key_col = "_col_"

    r0 = stem.rfind(key_row)
    if r0 == -1:
        return None

    row_val, j = _read_int_forward(stem, r0 + len(key_row))
    if row_val is None:
        return None

    c0 = stem.find(key_col, j)
    if c0 == -1:
        return None

    col_val, _ = _read_int_forward(stem, c0 + len(key_col))
    if col_val is None:
        return None

    return (row_val, col_val)


def _find_last_rowcol_token_start(stem: str) -> int | None:
    key_row = "sub_grid_row_"
    r0 = stem.rfind(key_row)
    if r0 == -1:
        return None
    # group prefix should cut before "sub_grid_row_..."
    return r0


def prefix_before_xy_tokens(path: Path) -> str:
    stem = path.stem
    k = _find_last_rowcol_token_start(stem)
    if k is None:
        return stem
    pref = stem[:k]
    while pref and pref[-1] in "_- .":
        pref = pref[:-1]
    return pref if pref else stem


def build_supergrid_tile_list(paths: list[Path]) -> list[Path]:
    groups = defaultdict(lambda: defaultdict(list))
    bad_rowcol = 0

    for p in paths:
        rowcol = parse_expected_subgrid_rowcol(p.name)
        if rowcol is None:
            bad_rowcol += 1
            continue
        pref = prefix_before_xy_tokens(p)
        groups[pref][rowcol].append(p)

    if bad_rowcol:
        print(f"[FAST] Supergrid: skipped {bad_rowcol} files with no expected subgrid row/col parse.")

    prefixes = sorted(groups.keys(), key=_prefix_sort_key)

    tile_list: list[Path] = []
    for pref in prefixes:
        per_rc = groups[pref]
        ordered_rc = sorted(per_rc.keys(), key=lambda t: (t[0], t[1]))
        for rc in ordered_rc:
            files = sorted(per_rc[rc], key=lambda pp: (pp.stem.lower(), pp.stat().st_mtime))
            tile_list.extend(files)

    print(f"[FAST] Supergrid: prefix groups={len(prefixes)}, tiles in list={len(tile_list)}")
    return tile_list


def build_suffix_xy_map(paths: list[Path]):
    by_xy = defaultdict(list)
    coords = set()
    bad = 0

    for p in paths:
        rowcol = parse_expected_subgrid_rowcol(p.name)
        if rowcol is None:
            bad += 1
            continue
        by_xy[rowcol].append(p)
        coords.add(rowcol)

    if not coords:
        raise RuntimeError("No valid expected subgrid row/col suffix coordinates found.")
    if bad:
        print(f"[FAST] Suffix mode: skipped {bad} files with no expected subgrid row/col suffix parse.")

    for k in list(by_xy.keys()):
        by_xy[k] = sorted(by_xy[k], key=lambda pp: (pp.stem.lower(), pp.stat().st_mtime))

    return by_xy, coords


def simple_pick_bands(paths, min_gap=3):
    first = paths[0]
    arr0 = np.load(first, mmap_mode='r')
    nb = int(arr0.shape[0])

    if BAND_PICK_MODE == "fixed":
        assert FIXED_BANDS is not None
        return list(FIXED_BANDS)

    i0 = max(0, min(nb - 1, int(round(nb * 0.10))))
    i1 = max(0, min(nb - 1, int(round(nb * 0.50))))
    i2 = max(0, min(nb - 1, int(round(nb * 0.90))))
    picks = [i0, i1, i2]

    picks = sorted(set(picks))
    while len(picks) < 3:
        for cand in [0, nb // 2, nb - 1]:
            if cand not in picks:
                picks.append(cand)
                if len(picks) == 3:
                    break
    picks = tuple(sorted(picks[:3]))
    print(f"[ULTRA] Picked RGB bands (mode={BAND_PICK_MODE}): {picks} (of {nb})")
    return picks


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def quick_percentiles(paths, bands_idx):
    p1 = np.zeros(3, dtype=np.float32)
    p9 = np.zeros(3, dtype=np.float32)
    cnt = 0
    num_cubes = int((len(paths) * RATIO_TOTAL_CUBES_FOR_CONTRAST)//1)
    
    for p in paths[:num_cubes]:
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


def last_three_folders(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 3:
        return " / ".join(parts[-3:])
    if len(parts) >= 1:
        return " / ".join(parts)
    return ""


def _get_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def add_label(img: np.ndarray, label: str, folder_info: str, font_size: int = 36) -> np.ndarray:
    img = img.astype(np.uint8, copy=False)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = _get_font(font_size)
    red = (255, 0, 0)
    draw.text((10, 10), folder_info, fill=red, font=font)

    if img.ndim == 3:
        pr, pg, pb = map(int, img[-1, -1])
    else:
        v = int(img[-1, -1])
        pr = pg = pb = v
    contrast = (255 - pr, 255 - pg, 255 - pb)
    draw.text((10, 30), label, fill=contrast, font=font)
    return np.array(pil)


def draw_tile_name(tile_img: np.ndarray, name: str, *, font_size: int = 12) -> np.ndarray:
    if tile_img.ndim == 2:
        rgb = np.stack([tile_img] * 3, axis=2)
    else:
        rgb = tile_img
    pil = Image.fromarray(rgb.astype(np.uint8, copy=False))
    draw = ImageDraw.Draw(pil)
    font = _get_font(font_size)
    draw.text((8, 6), name, fill=(255, 0, 0), font=font)
    out = np.array(pil)
    if tile_img.ndim == 2:
        return out[:, :, 0]
    return out


def hyperspectral_to_rgb(cube, bands_idx, p1, p9):
    chans = []
    dtype = np.uint8  # default

    for c, b in enumerate(bands_idx):
        ch = cube[b][::DOWNSAMPLE, ::DOWNSAMPLE].astype(np.float32, copy=False)

        if TEST_MANUAL_CONTRAST:
            bits = int(np.ceil(np.log2(TEST_MAX + 1)))

            if bits <= 8:
                max_data_size = 8
            elif bits <= 16:
                max_data_size = 16
            else:
                max_data_size = 64

            dtype = SchemaToDType[str(max_data_size)]
            # dtype is now np.uint8 / np.uint16 / np.uint64

            denom = (TEST_MAX - TEST_MIN) + 1e-8
            norm = (ch - TEST_MIN) / denom
            norm = clamp01(norm)

        else:
            if GET_EVERY_CUBE_RAW_MIN_MAX_VALUES:
                print("2D slice min/max:", float(ch.min()), float(ch.max()))

            denom = (p9[c] - p1[c]) + 1e-8
            norm = (ch - p1[c]) / denom
            norm = clamp01(norm)

            if GAMMA != 1.0:
                norm = norm ** (1.0 / GAMMA)

        chans.append(norm)

    arr = np.stack(chans, axis=2)

    # --- determine scale based on dtype ---
    if dtype is np.uint8:
        scale = 255.0
        out_dtype = np.uint8
    else:
        # normalize all uint16/uint64 to 12-bit range
        scale = 4095.0
        out_dtype = np.uint16

    arr = (arr * scale).round().astype(out_dtype)

    if OUTPUT_MODE == "grayscale":
        arr = arr.mean(axis=2).round().astype(out_dtype)


    return arr

def get_tile_size(first_path: Path):
    arr = np.load(first_path, mmap_mode="r")
    ref_h = arr.shape[1] // DOWNSAMPLE
    ref_w = arr.shape[2] // DOWNSAMPLE
    return ref_h, ref_w


def build_mosaic_from_tile_list_snake_bottom_left(
    tile_list: list[Path],
    rows: int,
    cols: int,
    ref_h: int,
    ref_w: int,
    bands_idx,
    p1,
    p9,
):
    first_tile = hyperspectral_to_rgb(np.load(tile_list[0], mmap_mode='r'), bands_idx, p1, p9)
    mosaic_dtype = first_tile.dtype
    if OUTPUT_MODE == "grayscale":
        mosaic = np.zeros((rows * ref_h, cols * ref_w), dtype=mosaic_dtype)
    else:
        mosaic = np.zeros((rows * ref_h, cols * ref_w, 3), dtype=mosaic_dtype)

    for grid_row in range(rows):
        y = rows - 1 - grid_row
        row_reversed = (grid_row % 2 == 1)

        # NEW: flip tiles on reversed rows (S-pattern)
        do_flip = S_PATTERN and row_reversed

        for x in range(cols):
            src_x = (cols - 1 - x) if row_reversed else x
            idx = grid_row * cols + src_x

            if idx >= len(tile_list):
                if FILL_MODE == "blank":
                    continue
                idx = len(tile_list) - 1

            path = tile_list[idx]
            cube = np.load(path, mmap_mode='r')
            tile_img = hyperspectral_to_rgb(cube, bands_idx, p1, p9)
            if LABEL_MODE is True:
                tile_img = draw_tile_name(tile_img, path.stem)

            if tile_img.shape[0] != ref_h or tile_img.shape[1] != ref_w:
                tile_img = np.array(Image.fromarray(tile_img).resize((ref_w, ref_h), Image.LANCZOS))

            if do_flip:
                tile_img = np.fliplr(tile_img)

            y0, x0 = y * ref_h, x * ref_w
            if OUTPUT_MODE == "grayscale":
                mosaic[y0:y0 + ref_h, x0:x0 + ref_w] = tile_img
            else:
                mosaic[y0:y0 + ref_h, x0:x0 + ref_w, :] = tile_img

    return mosaic


def build_mosaic_from_suffix_xy_snake_bottom_left(
    by_xy,
    rows: int,
    cols: int,
    ref_h: int,
    ref_w: int,
    bands_idx,
    p1,
    p9,
):
    sample_path = next(iter(by_xy.values()))[-1]
    sample_tile = hyperspectral_to_rgb(np.load(sample_path, mmap_mode='r'), bands_idx, p1, p9)
    mosaic_dtype = sample_tile.dtype

    if OUTPUT_MODE == "grayscale":
        mosaic = np.zeros((rows * ref_h, cols * ref_w), dtype=mosaic_dtype)
    else:
        mosaic = np.zeros((rows * ref_h, cols * ref_w, 3), dtype=mosaic_dtype)

    coords = list(by_xy.keys())
    for grid_row in range(rows):
        y = rows - 1 - grid_row
        row_reversed = (grid_row % 2 == 1)

        # NEW: flip tiles on reversed rows (S-pattern)
        do_flip = S_PATTERN and row_reversed

        for x in range(cols):
            src_x = (cols - 1 - x) if row_reversed else x
            want = (y, src_x)

            if want in by_xy:
                path = by_xy[want][-1]
            else:
                if FILL_MODE == "blank":
                    continue
                nearest = min(coords, key=lambda p: abs(p[0] - y) + abs(p[1] - src_x))
                path = by_xy[nearest][-1]

            cube = np.load(path, mmap_mode='r')
            tile_img = hyperspectral_to_rgb(cube, bands_idx, p1, p9)
            if LABEL_MODE is True:
                tile_img = draw_tile_name(tile_img, path.stem)

            if tile_img.shape[0] != ref_h or tile_img.shape[1] != ref_w:
                tile_img = np.array(Image.fromarray(tile_img).resize((ref_w, ref_h), Image.LANCZOS))

            if do_flip:
                tile_img = np.fliplr(tile_img)

            y0, x0 = y * ref_h, x * ref_w
            if OUTPUT_MODE == "grayscale":
                mosaic[y0:y0 + ref_h, x0:x0 + ref_w] = tile_img
            else:
                mosaic[y0:y0 + ref_h, x0:x0 + ref_w, :] = tile_img

    return mosaic


def main():
    t0 = time.time()
    paths = find_tiles(FOLDER)

    if SUPERGRID_ENABLED:
        tile_list = build_supergrid_tile_list(paths)
        if not tile_list:
            raise RuntimeError("SUPERGRID_ENABLED=True but no tiles were produced (check filename format).")

        rows = GRID_ROWS if GRID_MODE == "manual" else 1
        cols = GRID_COLS if GRID_MODE == "manual" else len(tile_list)

        ref_h, ref_w = get_tile_size(tile_list[0])

        print(f"[FAST] Supergrid ON: tile_list={len(tile_list)}")
        print(f"[FAST] Grid: using {rows}x{cols}")
        print(f"[FAST] Tile size: {ref_w}x{ref_h} (downsample={DOWNSAMPLE})")

        bands_idx = simple_pick_bands(tile_list, MIN_BAND_GAP)
        p1, p9 = quick_percentiles(tile_list, bands_idx)

        mosaic = build_mosaic_from_tile_list_snake_bottom_left(
            tile_list, rows, cols, ref_h, ref_w,
            bands_idx, p1, p9,
        )
        mode_tag = "supergrid"
    else:
        by_xy, coords = build_suffix_xy_map(paths)
        real_rows = max(y for y, _ in coords) + 1
        real_cols = max(x for _, x in coords) + 1

        if GRID_MODE == "manual":
            rows = max(GRID_ROWS, real_rows)
            cols = max(GRID_COLS, real_cols)
            print(f"[FAST] Grid override: using {rows}x{cols} (real is {real_rows}x{real_cols})")
        else:
            rows = real_rows
            cols = real_cols
            print(f"[FAST] Grid auto-detected: {rows}x{cols}")

        any_path = next(iter(by_xy.values()))[-1]
        ref_h, ref_w = get_tile_size(any_path)
        print(f"[FAST] Tile size: {ref_w}x{ref_h} (downsample={DOWNSAMPLE})")

        seeds = [v[-1] for v in by_xy.values()]
        bands_idx = simple_pick_bands(seeds, MIN_BAND_GAP)
        p1, p9 = quick_percentiles(seeds, bands_idx)

        mosaic = build_mosaic_from_suffix_xy_snake_bottom_left(
            by_xy, rows, cols, ref_h, ref_w,
            bands_idx, p1, p9,
        )
        mode_tag = "suffix"

    folder_text = last_three_folders(FOLDER)
    if OUTPUT_MODE == "grayscale":
        label_img = np.stack([mosaic] * 3, axis=2)
    else:
        label_img = mosaic

    label_text = f"Ultra-fast mosaic {mode_tag} {rows}x{cols} (mode={OUTPUT_MODE}, snake={S_PATTERN})"
    if LABEL_MODE == True:
        label_img = add_label(label_img, label_text, folder_text)

    if OUTPUT_MODE == "grayscale":
        label_img = label_img[:, :, 0]


    out_file = FOLDER / f"{OUT_PREFIX}_{OUTPUT_MODE}_{mode_tag}.png"
    import os

    if os.path.exists(out_file):
        folder, filename = os.path.split(out_file)
        stem, ext = os.path.splitext(filename)

        k = 1
        while True:
            candidate = os.path.join(folder, f"{stem}_copy{k}{ext}")
            if not os.path.exists(candidate):
                out_file = candidate
                break
            k += 1

    
    label_img = np.asarray(label_img)
    if np.issubdtype(label_img.dtype, np.floating):
        label_img = np.clip(label_img, 0, 4095).round().astype(np.uint16)
    if label_img.dtype == np.uint16:
        label_img = (label_img << 4)  # 0..4095 -> 0..65520 (fills 16-bit)


    Image.fromarray(label_img).save(out_file, quality=PNG_QUALITY)
    print(f"[FAST] Saved -> {out_file}")
    print(f"[FAST] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
