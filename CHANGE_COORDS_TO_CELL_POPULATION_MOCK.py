import os
import unittest
from CHANGE_COORDINATES_TO_CELL_POPULATION import Hold_Meta_Data, Post_Data_Process

# file name should be in mock folder:
#   actual_coords_x-0.845_y-51.6906_expected_none_cur_z_pos_-0.0_expected_z_position_0.0_um_sub_grid_row_2_col_1_data.npy
# expected rename:
#   100%_MCF_0%_NIH_actual_coords_x-0.845_y-51.6906_expected_none_cur_z_pos_-0.0_expected_z_position_0.0_um_sub_grid_row_2_col_1_data.npy

MOCK_FOLDER = r'E:\MCF_7_Breast_Cancer\Real\Mock_Folder'
#ORIGINAL_FILE = "actual_coords_x-0.845_y-51.6906_expected_none_cur_z_pos_-0.0_expected_z_position_0.0_um_sub_grid_row_2_col_1_data.npy"
#EXPECTED_PREFIX = "100%_MCF_0%_NIH_"
ORIGINAL_FILE = "actual_coords_x-76.6901_y-52.5354_expected_none_cur_z_pos_-0.0_expected_z_position_0.0_um_sub_grid_row_3_col_2_meta"
EXPECTED_PREFIX = "50%_MCF_50%_NIH_"
EXPECTED_FILE = f"{EXPECTED_PREFIX}_{ORIGINAL_FILE}"

class TestPostDataProcess(unittest.TestCase):

    def setUp(self):
        Hold_Meta_Data.FOLDER = MOCK_FOLDER
        self.hmd = Hold_Meta_Data
        self.process = Post_Data_Process(
            folder_name=self.hmd.FOLDER,
            actual_coordinates=self.hmd.ACTUAL_COORDINATES_BASE,
            cell_populations=self.hmd.CELL_POPULATIONS
        )

    def test_populations_coords_length_match(self):
        self.assertEqual(
            len(self.hmd.CELL_POPULATIONS),
            len(self.hmd.ACTUAL_COORDINATES_BASE)
        )

    def test_all_states_changed(self):
        result = self.process.psuedo_main()
        self.assertTrue(all(result), msg=f'States didnt change for: {result}')

    def test_file_renamed_with_correct_prefix(self):
        self.process.psuedo_main()
        renamed_path = os.path.join(MOCK_FOLDER, EXPECTED_FILE)
        self.assertTrue(
            os.path.exists(renamed_path),
            msg=f'Expected renamed file not found: {renamed_path}\n '
        )

if __name__ == "__main__":
    unittest.main()