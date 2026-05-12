import os
import unittest
from unittest.mock import patch
from CHANGE_COORDINATES_TO_CELL_POPULATION import Hold_Meta_Data, Post_Data_Process
from abc import ABC, abstractmethod

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
ROGUE_FILE_TO_DELETE ="0%_MCF_100%_NIH_actual_coords_x-26.69_y-25.0003_expected_none_cur_z_pos_-0.0_expected_z_position_0.0_um_sub_grid_row_0_col_2_data.npy"
ROGUE_PATH_TO_DELETE = os.path.join(MOCK_FOLDER, ROGUE_FILE_TO_DELETE)

print(ROGUE_PATH_TO_DELETE)

class PostProcessTestMixin(ABC):
    def __init__(self, *args, **kwargs) -> None:
    
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def setUp(self):
        Hold_Meta_Data.FOLDER= MOCK_FOLDER
        self.hmd = Hold_Meta_Data
        with patch ("os.remove") as mock_remove:
            self.process = Post_Data_Process(
            folder_name=self.hmd.FOLDER,
            actual_coordinates=self.hmd.ACTUAL_COORDINATES_BASE,
            cell_populations=self.hmd.CELL_POPULATIONS
        )
    
class TestPostProcessNameModificationsPostProcessTestMixin(PostProcessTestMixin, unittest.TestCase, ):

    def setUp(self):
        
        Hold_Meta_Data.FOLDER= MOCK_FOLDER
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

class TestPostProcessDataSizeModifications(PostProcessTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.file_to_delete = ROGUE_PATH_TO_DELETE

        #return super().setUp()(self)
    
    def test_mock_deletion(self):
        #return is like: return {"files_to_delete": files_to_delete}
        expected_Return: dict[str, list[str | os.PathLike,]] = {
            "files_to_delete": [self.file_to_delete,]
        }

        delete_bind = self.process._delete_files(
            self.file_to_delete,
            save_user_input=True,
            live_file_delete=True
        )

        #self.assertTrue(all(delete_bind), msg = f'File deleted is {delete_bind}')
        self.assertDictEqual(
            expected_Return,
            delete_bind,
            msg=f'expected_Return is {expected_Return}\n while delete_bind is {delete_bind}')
        
    
if __name__ == "__main__":
    '''
    print("path:", ROGUE_PATH_TO_DELETE)
    print("repr:", repr(ROGUE_PATH_TO_DELETE))
    print("exists:", os.path.exists(ROGUE_PATH_TO_DELETE))
    print("isfile:", os.path.isfile(ROGUE_PATH_TO_DELETE))
    print("isdir:", os.path.isdir(ROGUE_PATH_TO_DELETE))
    print("parent exists:", os.path.exists(os.path.dirname(ROGUE_PATH_TO_DELETE)))

    try:
        os.remove(ROGUE_PATH_TO_DELETE)
        print("REMOVED")
    except Exception as e:
        print("FAILED because:", type(e).__name__, repr(e))
    '''
    #unittest.main()
    #unittest.main(defaultTest = [callable(getattr(DataSizeMods, atr)) for atr in iter(DataSizeMods.__dict__())])
    
    DataSizeMods = TestPostProcessDataSizeModifications
    unittest.main(defaultTest="DataSizeMods.test_mock_deletion")
    unittest.TestCase(methodName = 'test_mock_deletion')
    #unittest.TestCase(methodName= 'runTest')
    