#CHANGE_COORDINATES_TO_CELL_POPULATION

import numpy as np
import os
import glob
import regex as re
import json
from typing import Literal, TypeAlias, Any, TypedDict, Tuple, TypeVar
from collections.abc import Iterable
import tkinter as tk
import gc
import warnings

globbed_objs: TypeAlias = os.PathLike | str
tup_of_tup: TypeAlias = tuple[tuple[Any, ...], ...]

T = TypeVar('T')

class Final_File_to_Final_Coords(TypedDict):
    files_to_be_modified: list[str | globbed_objs] | None
    coords_to_be_modified: list[tuple[float, float]] | None
    payload: Any


c_dict = Final_File_to_Final_Coords

MODIFY_FILE_NAMES = False

MODIFY_MEMORY_SIZE= True

DELETE_FILES = True

class  Hold_Meta_Data():
    FOLDER: str = r'C:\MCF_7_Breast_Cancer\Real\With_RGB_camera\pt_2'
                                        #dict[str, (x_coord, y_coord)]

    ACTUAL_COORDINATES_BASE =(
        (0,0), (25, 0), (50, 0), (75, 0), #order of pair in outer tuple is based on s-shape pattern. @ element [0] = (0,0); first item
        (75, 25), (50, 25), (25, 25), (0, 25), 
        (0, 50), (25, 50), (50, 50), (75, 50))
    #scanned and [1] = second item scannd, etc, etc.
    
    CELL_POPULATIONS = (    #MCF: NIH
        (25, 75), (25, 75), (0,0), (0,0),  #(0,0) <=> Blinded populations for test post validation 
        (75, 25), (75, 25), (0, 100), (0, 100),
        (100, 0), (100, 0), (50, 50), (50, 50), 
    )
#coords_to_population_dict: dict[str, tuple[float,float]]
# Base populations pattern : <num>%_<MCF>_><num>%_<NIH>
#coords: tuple[float, float] = 

class Post_Data_Process():
    def __init__(self, folder_name: globbed_objs, actual_coordinates: tup_of_tup, cell_populations: tup_of_tup) -> None:
        
        self.population_split = cell_populations
        self.actual_cords = actual_coordinates
        self.folder = folder_name
        
        self.extensions = "*.npy", "*.json" ,  "_data.npy", "_meta.json"

        self.globbed_files =  None
        self.gcd = None
        
        self._pre_vars = self.__dict__.copy() 
        self._post_vars = None
        files = []
        for ext in self.extensions:
            files.extend(glob.glob(f"*{ext}", root_dir=self.folder))
        print(f'len of files {len(files)}')
        files = files#[100:200]
        print(f'len of files {len(files)}')
        self.globbed_files = files
        del files
        self.tk_root= tk.Tk()

    def psuedo_main(self) -> list[Any]: #return instances where fields didn't change; means state change failed
        

        #print(self.globbed_folders[1]) #prints something like actual_coords_x-0.0001_y-51.6906_expected_none_cur_z_pos_0.0023_expected_z_position_0.0_um_sub_grid_row_2_col_0_data.npy

        self.gcd = self._find_gcd_within_iterables(self.actual_cords)

        self.population_split = self._make_populations_name(self.population_split)  

        # MUST RUN PARSER
        self.files_computed, self.files_not_computed, = self._parse_files_for_coords()
        print(f'Files computed: \t {self.files_computed} \n Files not computed: \t {self.files_not_computed}')
        self._post_vars = self.__dict__.copy()

        states = [
            self._post_vars[item] != self._pre_vars[item]
            for item in self._pre_vars
        ] 

        self.modify_file_name_via_files(self.files_computed)
        return states
    
    def modify_files_memory(self, delete_outlier_files_and_modify_others: bool):
        '''Match file size(effectively numpy dimensions) so that any NN has a consistent input params
            Delete the outliers, round down to lowest not outlier npy value. Our .npy files are
            of (spectral, spatial, line) but that type wasn't explicitly set with <np.dtype()>
                e.g. <dt = np.dtype([('time', [('min', np.int64), ('sec', np.int64)]),('temp', float)])>'''
        
        outlierConstant = 1.5
        assert self.globbed_files is not None
        files_not_opened = []
        files_opened = []
        file_dims = []
        memory_all_files = {"files_opened:": files_opened, "file_dims:" : file_dims, "temp": None}
        #self.__save_user_input: bool = False
        
        to_delete_wrapper = lambda _, override_delete: self._delete_files(
                            main_file_to_delete=_,
                            live_file_delete= override_delete if override_delete is not None else commands.get("should_delete", lambda: True)()
                        )
        
        for npy_file in [npy_ext for npy_ext in self.globbed_files if npy_ext.endswith('.npy')]:
            try:
                npy_file = str(os.path.join(self.folder, npy_file))
                arr = np.load(npy_file, mmap_mode="r")
                file_dims.append(arr.shape)
                files_opened.append(npy_file)
                del arr

            except Exception:
                files_not_opened.append(npy_file)
                print(f'Could not get memory of file: \t {npy_file}')
                try:
                    to_delete_wrapper(npy_file, override_delete= True)
                except:
                    RuntimeWarning(f"Couldn't delete file {npy_file}")
                
                    
        line_dims = np.array([d[2] for d in file_dims if len(d) >= 3], dtype=np.float64)
        print(f'line dims is: {line_dims} with len: {len(line_dims)}')

        temp = np.percentile(a=line_dims, q=(25, 75), method="closest_observation")
        lower_quartile, upper_quartile = temp[0], temp[1]
        
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        print(f'IQR is ({lower_quartile - IQR, upper_quartile + IQR})')
        ui_state = None
        files_truthy = None
        save_user_input_truthy = True
        commands = {}
        window = None

        if self.tk_root:
            files_truthy, save_user_input_truthy, ui_state = self._instaniate_tk_window()
            commands = ui_state["commands"]
            window = ui_state.get("window", None)
            print(f'went into {self.tk_root}')
            self.tk_root.mainloop()
        
        
        def __update_tk(file=None, **kwargs):
            window = kwargs.get("window", None)

            if not window or not window.winfo_exists():
                return
            
            cmds = kwargs.get("commands", {})

            for cmd in cmds.values():
                try:
                    cmd(file)
                except:
                    pass

            return

        print(f'lower quartile: {lower_quartile};\t upper quartile: {upper_quartile}')

        valid_lines = [l for l in line_dims if quartileSet[0] <= l <= quartileSet[1]]
        if not valid_lines:
            valid_lines = line_dims.tolist()

        target_line_dim = int(min(valid_lines))
        print(f'target_line_dim to round to is {target_line_dim}')
        meta_log = []

        target_line_dim = int(min(valid_lines))
        print(f'target_line_dim to round to is {target_line_dim}')
        meta_log = []

        for i, npy_file in enumerate(files_opened):
            try:
                # --- Phase 1: Read header only ---
                with open(npy_file, "rb") as f:
                    version = np.lib.format.read_magic(f)
                    shape, _, _ = np.lib.format._read_array_header(f, version)

                #print(f'{shape} is shape')

                # --- Phase 2: Branch on shape ---
                if shape[2] < quartileSet[0] and save_user_input_truthy:
                    print(f"DELETING at {i}, {npy_file}")
                    __update_tk(file=npy_file, **ui_state)
                    deleted = to_delete_wrapper(None, override_delete=False)
                    meta_log.append({
                        "files": deleted["files_to_delete"],
                        "action": "deleted" if delete_outlier_files_and_modify_others else "flagged_for_deletion",
                        "line_dim": int(shape[2])
                    })

                elif shape[2] != target_line_dim:
                    #print("MODIFY LOOP")

                    # Read fully (file is closed from phase 1)
                    arr = np.load(npy_file)
                    arr = arr[:, :, :target_line_dim]

                    print(f'arr.shape is {arr.shape}')
                    print(f'ENTERED: new arr will be {arr.shape}')

                    if delete_outlier_files_and_modify_others:
                        # Write to temp file first, then replace — avoids partial writes
                        tmp_path = npy_file + ".tmp"
                        with open(tmp_path, "wb") as fw:
                            np.save(fw, arr)
                        os.replace(tmp_path, npy_file)  # atomic on most OSes

                        meta_log.append({
                            "file": npy_file,
                            "action": "trimmed",
                            "new_dim": target_line_dim
                        })

                    del arr

            except Exception as e:
                print(f"FAILED FILE {npy_file}: {e}")
                to_delete_wrapper(npy_file, override_delete=True)

            gc.collect()
            
        meta_path = os.path.join(os.path.dirname(files_opened[0]), "modification_log.json")
        with open(meta_path, "w") as f:
            json.dump(meta_log, f, indent=2)

        memory_all_files["temp"] = {
            "target_line_dim": target_line_dim,
            "quartiles": (lower_quartile, upper_quartile),
            "bounds": quartileSet,
            "log_file": meta_path
        }

        def callback():
            if window and window.winfo_exists():
                window.destroy()
            if self.tk_root and self.tk_root.winfo_exists():
                self.tk_root.quit()

        self.tk_root.after(1000, callback)
        print(f'Finished modifiying files returning <memory_all_files> with type: {type(memory_all_files)}')
        return memory_all_files

    def _instaniate_tk_window(self) -> tuple[bool, bool, dict[str, tk.Widget]]:
        radio_like = tk.Toplevel(self.tk_root)

        files_truthy = tk.BooleanVar(value=False)
        save_user_input = tk.StringVar(value="SKIP")
        file_to_scroll_bar = tk.StringVar(value = "")
        
        text = 'long ' * 20
        text += 'line\n' * 20
        
        tk.Scrollbar(radio_like, orient=tk.HORIZONTAL).pack()
        tk.Scrollbar(radio_like, orient=tk.VERTICAL).pack()
        
        files_deleted = tk.Label(radio_like, text='Files Deleted', font=('Arial', 24))
        files_deleted.pack()

        file_listbox = tk.Listbox(radio_like, width=80, height=10)
        file_listbox.pack()
        
        result = {"SKIP": False, "YES": True, "NO": False, False: False, True: True}

        def close_and_store():
            radio_like.destroy()

        tk.Radiobutton(radio_like, text="YES", variable=files_truthy, value=True).pack()
        tk.Radiobutton(radio_like, text="NO", variable=files_truthy, value=False).pack()

        tk.Button(radio_like, text="OK", command=close_and_store).pack()

        commands = {
            "add_file": lambda file=None: file_listbox.insert(tk.END, file) if file is not None else None,
            "refresh": lambda: self.tk_root.update(),
            "should_delete": lambda: files_truthy.get(),
            "should_skip": lambda: save_user_input.get()
        }

        return files_truthy.get(), result[save_user_input.get()], {
            "window": radio_like,
            "commands": commands
        }
        
    def _delete_files(self, main_file_to_delete: os.PathLike | str,
                        default_file_types_to_check: tuple[str, ...] = ("_data.npy", "_meta.json"),
                        live_file_delete: bool = False,
                        debug: bool = False) -> dict[str, list[str | os.PathLike,]]:
            
        '''Pass in a .npy file or .json expected but allows others if it self.extensions; assumes not mutation'''
        
        default = default_file_types_to_check

        extension_warning = lambda ext: warnings.warn(
            f"<ext>: {ext} not in extensions; <self.extensions> = {self.extensions}; possible attribute mutation or bad string parse",
            category=RuntimeWarning,
            stacklevel=2,
        )

        b: str = str(main_file_to_delete)

        # use regex to grab extension at end (literal dot + non-dots until end)
        ending_match = re.search(pattern=r"(?:_data|_meta)\.[^.]+$", string=b)
        ending = ending_match.group(0) if ending_match else ""
        print(f"Ending is : {ending}")

        if not ending:
            extension_warning(b)
            return {"files_to_delete": []}

        if debug:
            a = b.replace(ending, default[1])

            print("path:", a)
            print("exists:", os.path.exists(a))
            print("isfile:", os.path.isfile(a))
            print("isdir:", os.path.isdir(a))
            print("parent exists:", os.path.exists(os.path.dirname(a)))

        file_exts_to_delete = [
            b.replace(ending, ext)
            for ext in default
        ]

        # make sure delete is subset/ are elements of self.extension no mutations
        print(f"files_to_delete: {file_exts_to_delete}")

        if file_exts_to_delete and live_file_delete != False:  # use parameter, not global 
            try:
                for file in file_exts_to_delete:
                    print(f"{file} in files to del loop")
                    if file and os.path.exists(file):  # should not crash here as this <_delete_files> func will be called from elsewhere
                        print(f"{file} in os remove loop")
                        os.remove(file)

            except:
                pass

        return {"files_to_delete": file_exts_to_delete}
    
    def _make_populations_name(self,
        all_population_splits: tuple[tuple[float, float], ...],
        naming_pattern: str = "num%_MCF_num%_NIH",
        naming_pattern_substition: str = "num"
    ) -> tuple[str, ...]:

        num = naming_pattern_substition
        string_names_to_return = ()
        
        #for num_coordinates in coordinates:
            #for coord in num_coordinates:
        for i, population in enumerate(all_population_splits):
            print(all_population_splits)
            print(population)
            a, b = population

            values = [str(a), str(b)]  # order of replacement

            def replacer(match):
                return values.pop(0) if values else match.group(0)

            result = re.sub(num, replacer, naming_pattern)
            string_names_to_return += (result,)

        assert string_names_to_return is not None
        
        return string_names_to_return


    def _parse_files_for_coords(self, **kwargs) -> tuple[list[str | globbed_objs], ...] | None :
        
        #example "actual_coords_x-0.0_y-0.0004_" 
        unpack = kwargs
        
        coord_sub = r'\d+\.?\d*'  #search for digits; ASI console cant produce more than 7 in a row; decimal in between possible
        cur_prefix_to_search: str = f"x-({coord_sub})_y-({coord_sub})_"
        print(f'cur_prefix_to_search: {cur_prefix_to_search}')

        files_not_computed: list[str | globbed_objs] = []
        files_computed: c_dict = {"files_to_be_modified": [],"coords_to_be_modified": [],"payload": None}
        
        def _check_coordinates_slop(
            coords_to_check: tuple[float, float] | tuple[Literal[-1, -1]] = (-1, -1),
            maximum_space_in_between_wells: float = 25,
            greatest_common_denom: float = 5
        ):

            '''Parse tuple of (x,y) of actual coordinates then return closest match based upon current coordinates'''
            print(f'coords to check: {coords_to_check}')

            if coords_to_check == (-1, -1):
                return (-1, -1)

            # --- STEP 1: coarse quantization via floor (stabilizes drift / noise) ---
            coarse_coords = []
            for coord in coords_to_check: 
                print(f"\n[COORD] original: {coord}")

                modulo = coord // greatest_common_denom
                print(f"[STEP] modulo = {coord} // {greatest_common_denom} -> {modulo}")

                remainderless_modulo = int(modulo * greatest_common_denom)
                print(f"[STEP] remainderless = {modulo} * {greatest_common_denom} -> {remainderless_modulo}")

                coarse_coords.append(remainderless_modulo)
                print(f"[APPEND] coarse_coords now: {coarse_coords}")
            coarse_coords = tuple(coarse_coords)
            print(f'coarse_coords are: {coarse_coords}')

            # --- STEP 2: nearest-center correction using known real coordinate grid ---
            def _dist(a, b):
                return (a[0] - b[0])**2 + (a[1] - b[1])**2

            nearest = min(
                self.actual_cords,
                key=lambda c: _dist(coarse_coords, c)
            )

            snapped_coords = nearest

            # --- STEP 3: assert validity in real coordinate system ---
            assert snapped_coords in self.actual_cords

            return snapped_coords
        
        for file in (self.globbed_files or []):
            a = []  # ensure defined
            
            try:
                
                a = re.findall(pattern=cur_prefix_to_search, string=file)  # max digits the asi console can read is 7
                print(f'initially: a[0] is {a[0]}\t a is {a}')
                if len(a) != 1:
                    raise ValueError(f"[BAD MATCH COUNT] {len(a)}")
                x_str, y_str = a[0]
                coords = (float(x_str), float(y_str))
            except Exception:
                try:
                    coords = tuple(map(float, re.search(pattern = coord_sub, string = a[0]))) #checks if entire obj is str return then parses as above
                except:
                    RuntimeWarning(f'Could not match {cur_prefix_to_search} digits trying alternative parsing')
                    a = re.split(pattern="None.$", string=file)
                
                print(f'could not find find structure for file appending to <files_not_computed>')
                files_not_computed.append(file)
            

            finally:
                if coords:  # safe guard

                    print(f'coords is {coords} of type: {type(coords)}')
                    new_coords = _check_coordinates_slop(coords)
                    files_computed["files_to_be_modified"].append(file)
                    files_computed["coords_to_be_modified"].append(new_coords)
                    
                    #f, c = files_computed.get("files_to_be_modified"), files_computed.get("coords_to_be_modified")
                    #f.append(file), c.append(new_coords)
                    #files_computed.update({files_to_be_modified: files_to_be_modified.append(file)},"coords_to_be_modified".append(coords))
                    
        return files_computed, files_not_computed #should return some sort of k,v pair or new typed dict to confine this maybe in payload


    def _find_gcd_within_iterables(self, obj): #used to generate common factor between all coords
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            values = [
                self._find_gcd_within_iterables(x)
                for x in obj
                if x is not None
            ]
            values = [v for v in values if v is not None]
            return np.gcd.reduce(values) if values else None
        else:
            return int(obj)


    def search_coordinates_via_files(self,
        *okay: tuple[tuple[float, float], ...],
        globbed_files: globbed_objs = None,
        naming_pattern: str = "num%_MCF_num%_NIH",
        naming_pattern_substition: str = "num"
    ):
        return None
    
    def modify_file_name_via_files(
        self,
        d: c_dict,
        naming_pattern: str = "num%_MCF_num%_NIH"
    ) -> list[str]:

        file_names = d["files_to_be_modified"]
        coords = d["coords_to_be_modified"]

        assert file_names is not None and coords is not None
        assert len(file_names) == len(coords)

        #data_dir = os.path.dirname(self.folder) 
        data_dir = self.folder
        renamed_files = []

        print(f'len of file name is {len(file_names)}; {range(len(file_names))}')
        for i in range(len(file_names)):
            print(f"i is {i}")

            file = file_names[i]
            coord = coords[i]
            coord_index = self.actual_cords.index(coord)
            mcf_pop_nih_pop = self.population_split[coord_index] #IMPORTANT TO NOT USE COORDINATES HERE WE ALREADY HAVE 1:1 MAPPING FROM EARLIER
            
            print(file)
            
            prefix =  mcf_pop_nih_pop

            old_path = os.path.join(data_dir, file)
            
            new_file = f"{prefix}_{file}"
            #new_file = None

    
            new_path = os.path.join(data_dir, new_file)
            if os.path.exists(new_file) or str(file).startswith(prefix): #checks to see if file already exist in dir
                continue
            else:
                print(f'old_path:\t {old_path}, \n new_path:\t {new_path}, ')# new_file{new_file}')

                os.rename(old_path, new_path)
                renamed_files.append(new_path)

        return renamed_files

if __name__ == "__main__": 
    Hmd = Hold_Meta_Data
    assert len(Hmd.CELL_POPULATIONS) == len(Hmd.ACTUAL_COORDINATES_BASE)
    
    Process_Data = Post_Data_Process(
        folder_name=Hmd.FOLDER, 
        actual_coordinates=Hmd.ACTUAL_COORDINATES_BASE, 
        cell_populations=Hmd.CELL_POPULATIONS
    )

    del Hmd.CELL_POPULATIONS, Hmd.ACTUAL_COORDINATES_BASE
    
    if MODIFY_FILE_NAMES:
        states = Process_Data.psuedo_main()
        print(f'All states changed for {type(Process_Data)}') if all(states) else print(f'States didnt change for: {states} in {type(Process_Data)}')
    
    elif MODIFY_MEMORY_SIZE:
        Process_Data.modify_files_memory(delete_outlier_files_and_modify_others= DELETE_FILES)