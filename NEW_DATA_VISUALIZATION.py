# new_data_visualization_all_permutations.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import itertools
import glob as glob
import regex as re

#Path obj on other server is \\server\share\folder
#Need both items before folder which is then the path relative to the server's drive on windows

#CUBE_ROOT_DIR = r'\\DEEPSPEC\datapipeline\Test_if_savable_from_other_computer\debug_issue_with_getposition_func_nonetype\re\REAL_20_rows_20_columns\re'
CUBE_ROOT_DIR = r'\\DEEPSPEC\\datapipeline\Test_if_savable_from_other_computer\debug_issue_with_getposition_func_nonetype\re\REAL_20_rows_20_columns\Hyper001_stained\Brightfield_re'
#CUBE_ROOT_DIR = r'Z:\Test_if_savable_from_other_computer\debug_issue_with_getposition_func_nonetype\re\REAL_20_rows_20_columns\Hyper001_stained\Brightfield_re'
#CUBE_ROOT_DIR = r'E:\Users\Ben\Programs\HSI Programs\z\Data\Chicago_Cancer\Chicago_5x5_test\Chicago_Slides_11_19_25\Real\Hyper005_II_unstained\Fluoro'
FILE_PATTERN = "*.npy"
#FILE_PATTERN = "*actual_0.0_0.0_expected_none_cube_y5_x5_copy.npy"
#FILE_PATTERN = "24.0*_*.npy"
PERMUTATIONS = False

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

def show_all_permutations(cube, permutations = PERMUTATIONS):
    if not PERMUTATIONS:
        permutations = [(0, 1, 2)] #permutations not required here anymore
    else:
        permutations =  list(itertools.permutations([0, 1, 2]))    
    print(f"Trying all {len(permutations)} axis permutations...")

    for i, perm in enumerate(permutations):
        permuted = np.transpose(cube, perm)  # Rearrange axes
        bands, rows, cols = permuted.shape

        # Heuristic: axis with highest count likely to be spectral (we show that as bands)
        print(f"\n[{i}] Permutation {perm} → shape {permuted.shape} (bands, rows, cols assumed)")

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)
        data = permuted[0, :, :]
        vmin, vmax = np.percentile(data, [1, 99])
        im = ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax) #documentation says lines 84,85 are using linear normalization (linear mapping)
        ax.set_title(f"Permutation {perm} — Band 1 / {bands}")
        cbar = plt.colorbar(im, ax=ax)

        # Slider setup
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03]) # <-- relating to sizing of the axes only not values
        slider = Slider(ax_slider, 'Band', 1, bands, valinit=1, valstep=1)

        def update(val, permuted=permuted):  # Capture permuted in default args
            band = int(slider.val) - 1
            data = permuted[band, :, :]
            vmin, vmax = np.percentile(data, [1, 99])
            im.set_data(data)
            im.set_clim(vmin, vmax)
            ax.set_title(f"Permutation {perm} — Band {band + 1} / {bands}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

if __name__ == "__main__":
    if not os.path.exists(CUBE_ROOT_DIR):
        raise IsADirectoryError (f"Folder not found: {CUBE_ROOT_DIR}")
    CUBE_PATHS = glob.glob(os.path.join(CUBE_ROOT_DIR, FILE_PATTERN))

    for i, cube_path in enumerate(CUBE_PATHS):
        filename = os.path.basename(cube_path)
        print(f'Cube # {i}', filename, sep='\t----\t')
        
        cube = load_cube(cube_path)
        show_all_permutations(cube)

