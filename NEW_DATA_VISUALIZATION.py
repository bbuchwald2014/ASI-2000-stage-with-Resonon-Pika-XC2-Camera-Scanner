import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import itertools
import glob as glob
import regex as re
from typing import TypeAlias, Final

#Path obj on other server is \\server\share\folder
# new_data_visualization_all_permutations.py

CUBE_ROOT_DIR = r'F:\Autofocus_data_set\Mock_initial_z_stack_set\Hyper001_Chicago_Cancer_BOT_SITE_HPV_NEG\Brightfield\Stained'
CUBE_ROOT_DIR = r'F:\Autofocus_data_set\Mock_initial_z_stack_set\Hyper001_Chicago_Cancer_BOT_SITE_HPV_NEG\Brightfield\Unstained'
FILE_PATTERN = "*.npy"

contrast_type_per_cube: TypeAlias = bool
t, f = True, False

USER_CONTRAST_CHOICE = f
del t, f


class Data_Settings():
    PERMUTATIONS: bool = False
    USE_WAVELENGTHS_INSTEAD_OF_BANDS: bool = False
    
    def __init__(self, contrast_truthy: bool) -> None:
        self.per_cube: Final[contrast_type_per_cube] = contrast_truthy
        self.per_band: Final[contrast_type_per_cube] = not self.per_cube


ds = Data_Settings(contrast_truthy=USER_CONTRAST_CHOICE)

np.set_printoptions(threshold=sys.maxsize)


def wrapper_for_HSI_Scanner_meth(num_bands: np.ndarray = np.empty((0)),
                                 spectral_bin_factor: int = 6,
                                 places_past_decimal=None) -> np.ndarray:
    from FORK_V2_COMBINED_FILE_HSI_14_POINT_5 import HSI_Scanner
    s = HSI_Scanner()
    m = s._solve_pixel_from_band(
        band_nums=num_bands,
        bin_factor=spectral_bin_factor,
        reversed_bc_pikaXC2=False,
        median_on=True,
        decimals=places_past_decimal
    )
    wavelengths = HSI_Scanner.lambda_from_bands(m)
    return wavelengths


def load_cube(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    cube = np.load(path)
    print("Original shape:", cube.shape)

    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape: {cube.shape}")

    print(f"Loaded cube: {path}")
    print(f"Shape: {cube.shape}, dtype: {cube.dtype}")
    return cube


def show_all_permutations(cube,permutations=Data_Settings.PERMUTATIONS,cur_cube_base_name_for_slider: str = ""):
    if not permutations:
        perms = [(0, 1, 2)]
    else:
        perms = list(itertools.permutations([0, 1, 2]))

    print(f"Trying all {len(perms)} axis permutations...")

    for i, perm in enumerate(perms):
        permuted_cube = np.transpose(cube, perm)
        bands, rows, cols = permuted_cube.shape
        bands = np.arange(bands)

        wavelengths = (
            wrapper_for_HSI_Scanner_meth(bands)
            if Data_Settings.USE_WAVELENGTHS_INSTEAD_OF_BANDS
            else None
        )

        print(f"\n[{i}] Permutation {perm} → shape {permuted_cube.shape}")

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)

        data = permuted_cube[0, :, :]
        vmin, vmax = np.percentile(data, [1, 99])

        im = ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax)

        if wavelengths is None:
            spectral_values = np.arange(len(bands))
            spectral_unit = "Band"
        else:
            spectral_values = wavelengths
            spectral_unit = "nm"

        perm_str = f"Permutation {perm} of" if tuple(perm) != (0, 1, 2) else ""

        rnd = lambda x: format(
            float(np.format_float_positional(
                x, precision=4, unique=False,
                fractional=False, trim='k'
            )),
            ".1f"
        )

        title_fn = lambda idx: (
            f"{perm_str} file: <{cur_cube_base_name_for_slider}> "
            f"\n —— at {rnd(spectral_values[int(idx)])} {spectral_unit}"
        )

        ax.set_title(title_fn(0))
        ax.set_xlabel("Lines")
        ax.set_ylabel("Spatial")
        cbar.set_label(f"Intensity ({permuted_cube.dtype})")

        ax_slider = plt.axes((0.3, 0.1, 0.6, 0.03))
        slider = Slider(
            ax=ax_slider,
            label="2D Slice Number:",
            valmin=0,
            valmax=len(spectral_values) - 1,
            valinit=0,
            valstep=1
        )

        def _get_total_v_min_v_max():
            return permuted_cube.min(), permuted_cube.max()

        def set_up_data(two_d_slice):
            two_d_slice_num = int(two_d_slice)

            if ds.per_cube:
                vmin, vmax = _get_total_v_min_v_max()
                data = permuted_cube[two_d_slice_num, :, :]
            else:
                data = permuted_cube[two_d_slice_num, :, :]
                vmin, vmax = np.percentile(data, [1, 99])

            im.set_data(data)
            im.set_clim(vmin, vmax)
            ax.set_title(title_fn(two_d_slice_num))
            fig.canvas.draw_idle()

        slider.on_changed(set_up_data)
        plt.show()


if __name__ == "__main__":
    if not os.path.exists(CUBE_ROOT_DIR):
        raise FileNotFoundError(f"Folder not found: {CUBE_ROOT_DIR}")

    CUBE_PATHS = glob.glob(os.path.join(CUBE_ROOT_DIR, FILE_PATTERN))

    for i, cube_path in enumerate(CUBE_PATHS):
        filename = os.path.basename(cube_path)
        print(f'Cube # {i}', filename, sep='\t----\t')

        cube = load_cube(cube_path)
        show_all_permutations(cube,
                              cur_cube_base_name_for_slider=filename)