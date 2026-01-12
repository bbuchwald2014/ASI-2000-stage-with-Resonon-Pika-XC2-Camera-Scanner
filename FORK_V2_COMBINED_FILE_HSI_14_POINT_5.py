#fork_v2_combined_file_HSI_14_point_5.py ; 12/11/25 --> post git upload

#SECOND CAMERA SETTINGS
#27k exposure, 0 db gain, led 15% < --? 1 filter and red mirror?
#500k exposure, 23.112, led = 99% <-- 2nd camera with brightfield and filters and mirror
#7k exposure? 3db, 20% <-- 2nd camera with brightfield and NO filters and NO mirror

#C:\Users\Robert Usselman\microscope\odmrm <-- py path for default kernel ; default set local not by vs code profile; cc https://code.visualstudio.com/docs/python/environments
#    ------> Changed to  .\HSI_env\Scripts\activate if .../z/ file structure wasn't changed because PySimpleGUI subscription ran out so using github clone of PySimpleGUI #4\
                #located in the Git_Extra_Packages folder
#import pypylon.genicam as genicam
#print(dir(genicam))

# path = 'C:\Users\Robert Usselman\Documents\HyperSpectralimaging\Resonson Rochi Testing\BrightField\10x for old Windows Batch (.wbt) programs 
#    --> contains fluoro parameters
    
#    *Looks like 490 ms integration, 0.00204 mm/s, 0.0762 distance 20 gain, predicted line count ~ 200
#
                
import sys
import os
from pathlib import Path

import time
import numpy as np
import regex as re
import keyboard # pip install keyboard
from typing import Tuple, List, Iterable
from typing import Callable, TypeVar, Any, Optional, Dict, Union
from itertools import product  # --- Build permutation tree of (debug_type, strategy_type) ---
import gc
import types
from decimal import Decimal, InvalidOperation

import math
import json
import psutil, os
from queue import Queue
#####
import threading, time
from multiprocessing import Process, Event
from multiprocessing.synchronize import Event as MpEvent
from math import floor
from dataclasses import dataclass, field
from pypylon import pylon
from functools import singledispatchmethod, partial
# -----------COMPUTER VISION-------------- #
import cv2

'''        self.scan_type_baseline_speed = self.ask_user_for_scan_type_fluoro_or_bright()
        self.is_fluorescence = self.scan_type_baseline_speed == 0.000762 
        if self.scan_type_baseline_speed == 0.0625:'''
# Try Sobel + Variance Combination
# -----------GUI EXTRAS-------------- #
# main.py

verbose = lambda *args: False
if verbose:
    for i, pathobj in enumerate(sys.path):print(f"Path index {i}:{pathobj} -- {pathobj}\n"
                                                f"\t from memory-- {pathobj.__dir__}\n" f"\t\t and file -- {os.path.abspath(str(pathobj))}")
del verbose
    
sys.path.append(r"E:\Users\Ben\Programs\HSI Programs\z")
from AUX_IMPORTS import ensure_packages, import_pysimplegui, safe_import
import CAMERA_LIVE_CONTROL2 as focus
from typing import TypedDict
import SCAN_METADATA_DICTIONARY as SMF #Custom class-module to hold metadata for GUI and .JSON files efficiently
import SCAN_METADATA_HEADERS as SMH 
# ------------------------- #
# Safe imports Check
# ------------------------- #
#default_callable = lambda *args, **kwargs: None


# (Optional) make sure these are present in THIS interpreter/venv
ensure_packages(["pyserial", "pypylon"], quiet=False)

# Import your libs
serial = safe_import("", "serial")                 # pyserial
pylon = safe_import("", "pypylon" )          # pypylon exposes 'pylon' submodule
#from pypylon import pylon
# Import PySimpleGUI:
# 1) Try normal; 2) else search recursively under project root (one level up from this file)
import os
project_root = os.path.dirname(os.path.abspath(__file__))  # or any root you want 

sg = import_pysimplegui(root_dir=project_root)

assert isinstance(sg, types.ModuleType), "Expected a module object, got something else" 
print("Loaded PySimpleGUI from:", getattr(sg, "__file__", "<no __file__>"))
import pathlib
#import PySimpleGUI as sg

# -----Debugging class-------- #
from abc import ABC, abstractmethod
from typing import Optional
import inspect
import warnings

'''TEMPORARY GLOBALS FOR CONVENIENCE'''
##HSI SCANNER INIT##
ENABLE_LIVE_KEYBOARD = True
ENABLE_MINIMUM_CAMERA_SETTINGS = True
USER_CONTINOUS_SECONDARY_CAMERA_SNAPSHOT = False
USE_SECONDARY_CAMERA = True
##MAKE_GUI CLASS##
DEFAULT_FOLDER = Path(r"E:\Users\Ben\Programs\HSI Programs\z\Data") #string arg not os.path arg --> changed to Path obj not for flexibility
#DEFAULT_FOLDER = pathlib.Path(r"E:\Users\Ben\Programs\HSI Programs\z\Data\working_camera_2")
SAVE_DIR_CONST = r"E:\Users\Ben\Programs\HSI Programs\z\Data\working_camera_8_with_wells_option"
# ------------------------- #


# If DEFAULT_FOLDER isn't defined elsewhere, fall back to a local "meta" folder.
try:
    DEFAULT_FOLDER  # type: ignore[name-defined]
except NameError:
    DEFAULT_FOLDER = Path.cwd() / "meta"

##SCAN_TILE_FUNC##
MAKE_DEBUG_FILE = True 
DEBUG_FRAMES = True
OVERRIDE_DEBUG_FILE = True
TIME_TO_MANUALLY_FOCUS = False
##MS2000 CLASS##
BACKLASH: float = 0.004  #in mm
ACCEL_TIME: float = 50 #in ms     Constants for class (global reference currently); **MINIMUM POSSIBLE ACCEL/DECEL TIME FOR ~6.86? mm ENCODERS
#WITHOUT SCREWING UP THE ACTUAL VELOCITY & DISTANCE PART OF MOVEMENT FUNC ITSELF; useful because it minimizes our error 
#associated with line-scanning. We don't hard code lines or trigger here; ping for ASI firmware for deceleration bits so need fast
#stage response speed. 



'''https://resonon.gitlab.io/programming-docs/content/overview.html#pika-l-l-gige-lf-xc2-and-uv x 447 channels possible?
--> Resonon Pika XC2 ROI PART OF SENSOR... ROI width = 1600' ROI Height = 924 //// ROI Width = Within sample spatial  && ROI Height = Spectral  
          
          Total = https://resonon.gitlab.io/programming-docs/content/overview.html#pika-l-l-gige-lf-xc2-and-uv                                           
        *Double-checked that this is true it's "(1200, 1920, 694)" --> https://www.baslerweb.com/en-us/shop/aca1920-155um/?variant=variant-b
                                                                   &&& https://resonon.com/Pika-XC2   --> says 447 where as basler says 694
                                                            Stagetop:  https://www.asiimaging.com/downloads/datasheets/MS-2000XY-Datasheet-Web.pdf
                                                            Stagetop 2: https://www.asiimaging.com/downloads/manuals/Operations_and_Programming_Manual.pdf
        ==> bin 1 sample, 1 spatial, 6 spectral = 200 x 1920 x 694 --> not expected dimensions should change binning to median over average
        ==> 
        
        --> Initially sampled images in get_dimensions_of_cubes.py and sliced along axis(0, 1, 2) and found axis[1] to have best graph 
        HOWEVER it has 924 width when spectral binning was 9 which means we were sampling correctly but not recreating properly?
            --> https://sharpsight.ai/blog/numpy-axes-explained/ 
            axis 0 = Going down Y-axis
            axis 1 = Going along X-axis
            axis 2= Going along Z axis (assumed)
    ** we are moving in correct axis delta X on ASI corresponds to delta X on spectronon **
            
             also useful for checking what resonon's settings/production for back-camera
            https://resonon.com/content/files/Resonon---Camera-Data-Sheets-Pika-XC2.pdf

Basic code intuition:
    Dynamic vars/methods/etc.  = while scan is live
    Static vars/methods/classes/etc. = while not scanning
           hence we separate the logic (mostly) with when real movement is happening, so no 
           real life or code-breaking bugs happen
           
    *1 space for sequence (e.g. main) or reading purposes (backlash method)
    *Many spaces for different logical object
            '''
            

# PySimpleGUI as sg # Replace import PySimpleGUI as sg with import FreeSimpleGUI as sg  --> https://github.com/spyoungtech/FreeSimpleGUI 
 #Free alternative available unsure if this messes with current syntax as of RN
'''
    Hobbyists can continue to use PySimpleGUI 5 until their keys expire. After that you'll need to switch to version 4, which you'll find 1,000s of copies on GitHub 
    with at least 1 being community supported.If you wish to use PySimpleGUI without the key expiring or want support, then you can buy a Commercial License which is 
    good perpetually.'''

import serial
from serial import Serial, SerialException, EIGHTBITS, PARITY_NONE, STOPBITS_ONE # pyright: ignore[reportGeneralTypeIssues]
from serial.tools import list_ports
from pypylon import pylon
import PySimpleGUI as sg  # type: ignore ##SUPRESSION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


import os
import sys
import importlib.util
from typing import Optional
    
# ------------------------- #
# SerialPort class
# ------------------------- #

class SerialPort:
    def __init__(self, com_port: str, baud_rate: int, report: bool = True):
        self.serial_port = Serial() 
        self.com_port = com_port
        self.baud_rate = baud_rate
        self.report = report
        self.print = self.report_to_console

    @staticmethod
    def scan_ports() -> list[str]:
        ports = [port.device for port in list_ports.comports()]
        ports.sort(key=lambda v: int(v[3:]))
        return ports

    @staticmethod
    def send_command_direct(
        stage_com_port: str = "COM4",
        stage_baud_rate: int = 115200,
        command: str = "",
        timeout: float = 0.1
    ) -> str:
        """
         ******ASI CONSOLE ONLY DEFAULTS BACK TO IT'S WHEN IT'S **PHYSICALLY** TURNED OFF NOT OPENING/CLOSING COM PORTS********
         
        Send one raw command directly to a COM port without instantiating SerialPort.
        Opens -> writes -> reads -> closes automatically.

        :param stage_com_port: COM port string, e.g. "COM4"
        :param stage_baud_rate: Baud rate (valid: 9600, 19200, 28800, 115200). Default = 115200.
        :param command: The ASI/MS2000 command string to send (without carriage return).
        :param timeout: Read timeout in seconds (default=1).
        :return: First line of response from the controller.
        """
        VALID_BAUD_RATES = (9600, 19200, 28800, 115200)
        if stage_baud_rate not in VALID_BAUD_RATES:
            raise ValueError(
                f"Invalid baud rate {stage_baud_rate}. "
                f"Valid options: {VALID_BAUD_RATES}"
            )

        response = ""
        # Context manager ensures the port is *always closed* when done
        with Serial(port=stage_com_port, baudrate=stage_baud_rate, timeout=timeout) as ser:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            ser.write(bytes(f"{command}\r", encoding="ascii"))
            response = ser.readline().decode("ascii").strip()

        # At this point `ser.close()` has already been called
        return response

    def connect_to_serial(self, rx_size=12800, tx_size=12800, read_timeout=1, write_timeout=1):
        self.serial_port.port = self.com_port
        self.serial_port.baudrate = self.baud_rate
        self.serial_port.parity = PARITY_NONE
        self.serial_port.bytesize = EIGHTBITS
        self.serial_port.stopbits = STOPBITS_ONE
        self.serial_port.xonoff = False
        self.serial_port.rtscts = False
        self.serial_port.dsrdtr = False
        self.serial_port.write_timeout = write_timeout
        self.serial_port.timeout = read_timeout
        self.serial_port.set_buffer_size(rx_size, tx_size)
        try:
            self.serial_port.open()
        except SerialException:
            self.print(f"Cannot connect to {self.com_port} at {self.baud_rate}!")

        if self.is_open():
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            self.print(f"Connected to serial port: {self.com_port} at {self.baud_rate}")

    def disconnect_from_serial(self):
        if self.is_open():
            self.serial_port.close()
            self.print("Disconnected from serial port")

    def is_open(self) -> bool:
        return self.serial_port and self.serial_port.is_open

    def report_to_console(self, message: str):
        if self.report:
            print(message)

    @dataclass
    class GuaranteedDecode:
        command: bytes
        printer: Callable[[str], None]
        decode: Callable[[], str] = field(init=False)

        def __post_init__(self):
            self.decode = lambda: self.command.decode("ascii").strip()

        def run(self) -> str:
            s = self.decode()
            self.printer(f"Send: {s}")
            return s

    def send_command_normal(
        self,
        cmd: str,
        *,
        return_encoded: bool = False,   # just decides if we return the sent text
    ) -> Optional[str] | Optional[int] | None:
        
        self.serial_port.reset_input_buffer() #assumption is no cached firmware settings like set x-axis speed are cleared per use
        self.serial_port.reset_output_buffer() #assumption is no cached firmware settings like set x-axis speed are cleared per use
        command = bytes(f"{cmd}\r", encoding= ("ascii"))
        # fast path: minimal overhead, let exceptions propagate
        sent_text = self.serial_port.write(command)
        if (command is None) or (not return_encoded):
            return None
        elif sent_text is None:
            print(f'sent_text var on line not used {DebugProgram.get_line_number()}')
            return None
        else:
            sent_text = self.serial_port.write(command)
            self.print(f"Send: {sent_text}")
            return sent_text
        
    def send_command_special1(self, axis="X"):
        DEC = re.compile(r"\b(\d{1,3})\b")

        def _read_line_crlf(ser):
            buf = bytearray()
            while True:
                b = ser.read(1)
                if not b:  # blocking mode recommended
                    continue
                buf += b
                if b in (b"\r", b"\n"):
                    break
            return buf.decode("ascii", errors="ignore").strip()

        self.serial_port.reset_input_buffer()

        # RDSTAT first (decimal)
        self.serial_port.write((f"RDSTAT {axis}\r").encode("ascii"))
        rd = _read_line_crlf(self.serial_port)
        print("RDSTAT raw:", repr(rd))

        m = DEC.search(rd)
        if m:
            val = int(m.group(1))
            print("RDSTAT dec:", val)
            return val

        # If no decimal, try RDSBYTE (hex)
        self.serial_port.reset_input_buffer()
        self.serial_port.write((f"RDSBYTE {axis}\r").encode("ascii"))
        rb = _read_line_crlf(self.serial_port)
        print("RDSBYTE raw:", repr(rb))
    
    
    def send_command_special2(self, axis="X", verbose: bool = False) -> int:
        """
        Ultra-fast RDSTAT reader (0–255 integer).
        Prints only the parsed decimal value.
        """
        ser = self.serial_port
        ser.reset_input_buffer()
        ser.write((f"RDSTAT {axis}\r").encode("ascii"))

        val = 0
        in_num = False

        while True:
            b = ser.read(1)
            if not b:
                continue
            c = b[0]

            # digits '0'–'9'
            if 48 <= c <= 57:
                if not in_num:
                    in_num = True
                    val = 0
                val = val * 10 + (c - 48)
                if val > 255:
                    val = 255
                    break
            else:
                if in_num:
                    break
                if c in (13, 10):
                    continue
        if verbose:
            print(f"RDSTAT dec: {val}")
        return val

    def read_response(self) -> str:
        return self.serial_port.readline().decode('ascii').strip()
    '''
        try:
            response = self.serial_port.readline().decode('ascii').strip()
            print(response)
            return response
        
        except:
            print(self.serial_port.readline())
            
        finally:
            #print(self)
        #self.print(f"Recv: {response}")
        
            return "None"
    '''


# ------------------------- #
# MS2000 stage controller
# ------------------------- #
class MS2000(SerialPort):
    BAUD_RATES = [9600, 19200, 28800, 115200]

    def __init__(self, stage_com_port: str, stage_baud_rate=115200, report=True):
        
        super().__init__(stage_com_port, stage_baud_rate, report)
        if stage_baud_rate not in self.BAUD_RATES:
            raise ValueError("Invalid baud rate for MS2000")


    def close(self):
        try:
            if self.is_open():
                self.disconnect_from_serial()
                print("Disconnected from serial port")
        except Exception as e:
            print(f"Error while closing MS2000: {e}")
        finally:
            self.serial_port = None
            print("Closing MS2000/SerialPort object")


 ##### STAGE COMMANDS USING ASI'S MS2000 STAGE COMMAND API #####
    def move_stage(self, x=None, y=None, z=None, relative=False):
        """
        Move the stage. By default moves absolute, unless relative=True.
        Uses ASI 'MOVE' (absolute) or 'MOVREL' (relative).
        All inputs should be in ASI units (tenths of microns).
        """
        cmd_parts = []
        if x is not None:
            cmd_parts.append(f"X={int(x)}")
        if y is not None:
            cmd_parts.append(f"Y={int(y)}")
        if z is not None:
            cmd_parts.append(f"Z={int(z)}")
        if not cmd_parts:
            return  # nothing to move

        cmd_str = " ".join(cmd_parts)
        command = "MOVREL" if relative else "MOVE"

        self.send_command_normal(f"{command} {cmd_str}")
        self.read_response()
        
    
        
    def custom_starting_home_points(
        self,
        home_points: Optional[List[Tuple[int, int, int]]] = None
    ) -> List[Tuple[int, int, int]]:
        """
        # eg for a messupplies 12 well ; # delta x = +16.7 mm, delta y = +24.8 mm = Home well #1
        
        Moves the stage to the first custom home point and returns the remaining points.

        Parameters:
            a (float): Optional parameter, default is 0.
            home_points (List[Tuple[int, int, int]], optional): List of (x, y, z) coordinates.
                Defaults to [(0, 0, 0)] if None or empty.

        Returns:
            List[Tuple[int, int, int]]: The remaining list of points after popping the first.
                This list can be empty (`[]`) if all points have been used.
        """
        assert (home_points is not None and len(home_points) > 0 and type(home_points is Tuple[int, int, int])) # was causing crash --> <and type(home_points) is Tuple) >
        
        point = home_points.pop(0)  # Remove and retrieve the first (x, y, z) tuple
        custom_x, custom_y, custom_z = point
        print(f"MOVING TO HOME POSITION: future home; {custom_x}, {custom_y}, {custom_z}")

        self.move_stage(x=custom_x, y=custom_y, z=custom_z)

        return home_points



    def get_position(self, axis: str) -> Optional[Union[int, float]]:
        """
        Query the stage for its position along a given axis.
        """

        def _read_position(axis: str, _attempt_num: int = 1) -> Optional[Union[int, float]]:
            # --- First attempt: normal WHERE ---
            resp = self.send_command_normal(f"WHERE {axis}", return_encoded=True)
            resp = self.read_response()

            if (resp is None) and (_attempt_num <= 3):
                _attempt_num += 1
                print(f"[Fallback] Raw response: {resp!r}, type={type(resp)}")
                time.sleep(2)
                print(
                    f"Attempt num: {_attempt_num},\tCalling _read_position func to reread current stage position"
                )
                print(f"On ~line num: {DebugTimer.get_line_number()}")
                return _read_position(axis=axis, _attempt_num= _attempt_num)

            print(f"[Fast] Raw response: {resp!r}, type={type(resp)}")

            pos = _parse_position(resp)
            if pos is not None:
                return pos

            # --- Second attempt: fallback W ---
            print("[Fast] Failed. Trying fallback 'W' command...")
            resp = self.send_command_normal(f"W {axis}", return_encoded=True)

            return _parse_position(resp)

        def _parse_position(value_str: Optional[str]) -> Optional[Union[int, float]]:
            """Parse position from a response string."""
            if not value_str:
                return None

            if isinstance(value_str, str) and len(value_str) > 1:
                parts = value_str.split()
            else:
                return None

            print(f"Parsed parts: {parts}")

            if len(parts) < 2:
                return None

            raw = parts[1]

            # Try float first
            try:
                raw = float(raw)
                try:
                    raw *= -1.0
                    return raw
                except:
                    pass
                return raw
            except ValueError as e:
                raise ValueError(
                    f"{e} can't cast raw response in _parse_position"
                ) from e

        try:
            return _read_position(axis)
        except (IndexError, ValueError, TypeError) as e:
            print(f"Error while reading position: {e}. Retrying once...")
            try:
                return _read_position(axis)
            except Exception as e2:
                print(f"Retry also failed: {e2}")
                return None



    def get_position_trycasters(self, axis: str, types_to_check: Iterable[type] = (int, float, str)) -> Optional[object]:
        try:
            self.send_command_normal(f"WHERE {axis.upper()}", return_encoded=True)
            parts = self.read_response().strip().split()
            if len(parts) >= 2:
                token = parts[1]
            for i, caster in enumerate(types_to_check, start=1):
                print(f"{i}th attempt: trying {caster.__name__} on {token}") #can return the name of the type, method, func, class, etc. 
                try:
                    return caster(token)
                except Exception as e:
                        f'{e}:Cannot return caster obj: {caster} at line {DebugProgram.get_line_number()}'
            return None
        except Exception:
            return None

    #def move(self, x=0, y=0, z=0):
        #self.send_command(f"MOVE X={x} Y={y} Z={z}")
        #self.read_response()     #OUTDATED

    def __move_axis(self, axis: str, distance: float):
        self.send_command_normal(f"MOVE {axis}={distance}")
        self.read_response()

    def set_max_speed(self, axis: str, speed: float) -> float:
        if speed == 0:
            return speed
        else:
            """Speed in mm/s (direct, no conversion).""" #maximum speed is = 7.5 mm/s for standard 6.5 mm pitch leadscrews
            print(f'set_max_speed func which axis {axis} and speed {speed} mm/s and type {type(axis)}')
            self.send_command_normal(f"SPEED {axis}={speed}")
            self.read_response()
            return speed

    def is_axis_busy(self, axis: str) -> bool:
        self.send_command_normal(f"RS {axis}?")
        return "B" in self.read_response()
 
    def get_RDSTAT(self, axis: str) -> bool:
        rd_stat = self.send_command_special1(axis = axis)
        return False if rd_stat == 63 else True
        #return true if we are deccelearating or false if not based on ms 2000 "RDSTAT" command
        '''
        #144 < x <  128 + 64 + 16 
        #143 < x < 
        eight_byte_int: int = self.send_command(f"RS {axis}")
        if eight_byte_int < 143:
            return False 
        #must be greater than or equal to 144 here then
        if eight_byte_int > 203:
            return False
        #number now must be less than or equal to 203
        else: return True
        #print((self.send_command1(axis = axis)))
        '''

    '''
    def get_RDSTAT(self, axis: str) -> bool:
        """Return True if the stage is decelerating (bits 4 & 5 set)."""
        val = self.send_command1(axis)  # returns 0–255
        return (val & 0x30) != 0x30
    '''

    def wait_for_device(self, axis=None):
        
        print("Waiting for device...")
        if axis is None:
            while self.is_axis_busy('X') or self.is_axis_busy('Y'):
                time.sleep(0.005)
        else:
            while self.is_axis_busy(axis):
                time.sleep(0.005)
    
    def _turn_on_light(self, actual_percentage_power: int):
        self._change_light_power(actual_percentage_power)  # POWER IS A [0, 99] NOT 100 :^)
        #self.change_light_power(percentage_power=  99)  # POWER IS A [0, 99] NOT 100 :^)
        print("Light turned ON")
    
    def _turn_off_light(self):
        self._change_light_power(actual_percentage_power =  0)  # Change to real command
        print("Light turned OFF")
    
    def _change_light_power(self, actual_percentage_power: int):
        '''  
        **NOT CASE SENSITIVE**
        command: LED (LED DIMMER Firmware Required)
        Format: LED [X= 0 to 99]
        LED X?'''
        
        self.send_command_normal(f"LED X = {actual_percentage_power}")
        print(f"Changed LED power to --> {actual_percentage_power}%")
       
    @staticmethod
    def get_and_set_backlash(
        stage_com_port: str = "COM4",
        stage_baud_rate: int = 115200,
        axis: str = "X",
        backlash_distance: float = BACKLASH
    ):
        """
        Set the backlash distance on a given axis using the provided MS2000 controller instance.
        Basically; the MS2000 accelerates during our {set_distance} and goes to the end of {set_distance} at {set_speed}
        hence it decelerates while traveling still after meeting target. Hence the backlash is one of the correction factors
        to make sure the stage actually ends where it says it will; travel distance is ~ 3 microns extra when ms = 100 at 0.0623 mm/s.
        
        Need to make backlash_distance + error > deceleration_distance
        
        -Online manual says it shouldn't go under 50 ms for our gear type; other errors including drift and finish error factor in here
        :param controller: MS2000 instance (already connected)
        :param axis: Axis name (e.g., 'X')
        :param backlash_distance: Distance in mm (must be >= 0.000022)
        """
        
        '''
        CC supplementary_infomation_only.ipynb on how backlash releates to deceleration
        '''
        if axis is None or backlash_distance is None or backlash_distance < 0.000022:
            print("Invalid backlash distance or axis.")
            return 0
        
        #Below comment is outdated command?:
        #backlash_counts = float(backlash_distance * 45397.60)  # encoder counts/mm 
        #controller.send_command(f"BACKLASH {axis}={backlash_counts}")
        
        SerialPort.send_command_direct(
            stage_com_port=stage_com_port,
            stage_baud_rate=stage_baud_rate,
            command=f"BACKLASH {axis}={backlash_distance}"
        )
        return MS2000.get_ASI_stagetop_info(stage_com_port, stage_baud_rate, [axis])

    @staticmethod
    def get_and_set_acceleration(
        stage_com_port: str = "COM4",
        stage_baud_rate: int = 115200,
        axis: str = "X",
        time_taken_to_accelerate: int = ACCEL_TIME
    ):
        """
        Set time it takes to accelerate, in (mm/s^2), from 0 mm/s --> {set_speed} mm/s. 
        MS2000 doesn't decelerate until after reaching {set_distance}. We scan in X direction only with spectronon camera
        so default for now is X-axis.
        
        :param controller: MS2000 instance (already connected)
        :param axis: Axis name (e.g., 'X')
        :param backlash_distance: Distance in mm (must be >= 0.000022)
        """
    
        '''
         CC online ASI manual for more info. 
        '''
        SerialPort.send_command_direct(
            stage_com_port=stage_com_port,
            stage_baud_rate=stage_baud_rate,
            command=f"ACCEL {axis}={time_taken_to_accelerate}"
        )
        return MS2000.get_ASI_stagetop_info(stage_com_port, stage_baud_rate, [axis])
        

    @staticmethod
    def get_ASI_stagetop_info(
        stage_com_port: str = "COM4",
        stage_baud_rate: int = 115200,
        axes: list[str] | None = None
    ) -> list[str]:
        """
        Get detailed info for specified axes.

        :param controller: MS2000 instance
        :param axes: List of axis names (e.g., ['X', 'Y'])
        :return: List of response strings
        """
        if axes is None:
            axes = ["X", "Y", "Z"]

        results: list[str] = []
        for axis in axes:
            response = SerialPort.send_command_direct(
                stage_com_port=stage_com_port,
                stage_baud_rate=stage_baud_rate,
                command=f"I {axis}"
            )
            results.append(response)
        print(f"Current stage info is: {results}")
        return results

# ------------------------- #
# HSI Scanner
# ------------------------- #

class HSI_Scanner:
    '''Make dictionary of all the class variables so they can be referenced by strings regardless if value changes or not'''
    A = 1.37999e-5   # class variable: quadratic calibration coefficient (Resonon Pika XC2)
    B = 0.6555092    # class variable: linear calibration coefficient
    C = 322.6319     # class variable: constant calibration coefficient
    SENSOR_DIMS = (1936, 1216)  # class variable: expected full Basler sensor dimensions (width, height)
    T = TypeVar("T")  # generic type variable used in callable precision functions (for IDE typing)
    default_camera: Optional[pylon.InstantCamera] = None #Should be hardcoded to 1 camera type; pika xc2 because methods make not work for others; class variable bc happens before scan
    _dispatch_installed: bool = False  # guard   

    # ------------------------- #
    # Init
    # ------------------------- #
    def __init__(self, camera_index=None, alias_MS2k_stage: Optional[MS2000] = None, *camera_args):
        # ---- Stage setup ----
        if alias_MS2k_stage is not None and not isinstance(alias_MS2k_stage, MS2000):
            raise TypeError(f"alias_MS2k_stage must be an MS2000 instance or None, got {type(alias_MS2k_stage)}")
        self.Stage = alias_MS2k_stage
        if self.Stage is not None:
            self.Stage.connect_to_serial()
            print("Stage connected.")
        else:
            print("[INFO] No stage alias provided; skipping stage connect.")

        self.enable_keyboard_control = ENABLE_LIVE_KEYBOARD
        self.debug_frames = None

        # ---- Camera discovery ----
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No cameras found")

        # Resolve primary device
        if camera_index is None:
            device_info = devices[0]
            print(f"default camera behavior at {device_info}")
        elif isinstance(camera_index, int):
            if camera_index < 0 or camera_index >= len(devices):
                raise RuntimeError(f"camera_index out of range: {camera_index}")
            device_info = devices[camera_index]
        elif isinstance(camera_index, str):
            matches = [
                d for d in devices
                if d.GetModelName() == camera_index
                or d.GetSerialNumber() == camera_index
                or d.GetFriendlyName() == camera_index
            ]
            if not matches:
                raise RuntimeError(f"No camera found matching '{camera_index}' (model/serial/friendly).")
            device_info = matches[0]
        else:
            raise RuntimeError("Invalid camera_index type")

        # Print list
        print("Available cameras:")
        for i, dev in enumerate(devices):
            marker = "<-- SELECTED (primary)" if dev is device_info else ""
            print(f"Index {i}: Model={dev.GetModelName()}, Serial={dev.GetSerialNumber()}, Friendly={dev.GetFriendlyName()} {marker}")
        print(f"Using camera with device_info: {device_info}")

        # ---- Open primary camera ----
        self.camera = pylon.InstantCamera(tl_factory.CreateDevice(device_info))
        self.camera.Open()

        type(self)._install_dispatchers()

        # ---- Initial temperature (assuming this static/classmethod exists) ----
        self.initial_temperature = HSI_Scanner.get_temperature(which_subdevice=self.camera)
        
        # Handle initial camera settings if enabled
        if ENABLE_MINIMUM_CAMERA_SETTINGS:
            initial_exposure, initial_gain = self._initialize_camera_settings()
            initial_set_exposure_var = self.set_exposure_from_gui(initial_exposure_set_ms=initial_exposure / 1000)
            initial_set_gain_var = self.increase_gain(current_gain=initial_gain)

            print(f"Variable values of (initial_set_exposure, initial_gain) = ({initial_set_exposure_var}, {initial_set_gain_var})")
        else:
            pass  # Change logic later if needed
        self._set_ROI()
        
        print(f"Using camera: {self.camera.GetDeviceInfo().GetModelName()}")
        
        # ---- Secondary camera logic ----
        if USE_SECONDARY_CAMERA:
            _secondary_device_info = next((d for d in devices if d is not device_info), None)
            self.set_up_secondary_camera(secondary_info=_secondary_device_info, save_dir_override = Make_GUI.cur_dir)

    # ------------------------- #
    # Secondary camera helpers
    # ------------------------- #
    # ------------------------- #            self._snapshot_event.set()
    @staticmethod
    def _launch_focus_gui(serial: str, second_camera_save_dir: str = "", stop_event: MpEvent | None = None, snapshot_event: MpEvent | None = None):
        focus.CameraApp.receive_serial(serial)
        focus.CameraApp.main(second_camera_save_dir, stop_event, snapshot_event)

    def set_up_secondary_camera(self, secondary_info, save_dir_override: str = ""):
        if secondary_info is not None:
            focus_serial = secondary_info.GetSerialNumber()

            # Stop event to signal the child process to quit
            stop_event = Event()
            
            
            continous_snapshot = USER_CONTINOUS_SECONDARY_CAMERA_SNAPSHOT 
            # Event to toggle continuous snapshots
            if continous_snapshot:
                self._snapshot_event = Event()  # keep a reference
            else:
                self._snapshot_event = None
            self._close_secondary_camera_event = stop_event

            # pick save_dir: either override, else use constant, else empty
            save_dir = save_dir_override or SAVE_DIR_CONST or ""

            # Launch focus GUI in its own process
            p = Process(
                target=HSI_Scanner._launch_focus_gui,
                args=(focus_serial, save_dir, stop_event, self._snapshot_event,),
                daemon=True
            )
            self.secondary_process = p
            p.start()
            print(f"[INFO] Focus GUI launched in child process PID={p.pid}")

        else:
            print("[INFO] No secondary camera detected; setting primary pixel format to Mono12 by default.")
            self._set_camera_pixel_format("Mono12")


    def close_secondary_camera(self):
        """Request the focus GUI to shut down."""
        if hasattr(self, "_close_secondary_camera_event") and self.secondary_process.is_alive():
            print("[INFO] Closing secondary camera...")
            self._close_secondary_camera_event.set()  # signal child process
            self.secondary_process.join(timeout=10)  # give it more time

            if self.secondary_process.is_alive():
                print("[WARN] Secondary camera process did not exit; terminating forcibly...")
                self.secondary_process.terminate()  # force exit
                self.secondary_process.join()
            print("[INFO] Secondary camera process closed.")
        else:
            print("No secondary camera is currently open to close")
        
    def close(self):
        """NOT INHERITED SELF-MADE Public cleanup method for scanner. Closes camera by default"""

        # Nested helper so it's invisible outside close()
        def _do_cleanup(camera_to_close):
            # Check if a camera object exists
            if camera_to_close is None:
                print("No camera object to clean up.")
                return

            # Try to stop grabbing safely
            try:
                camera_to_close.StopGrabbing()
                print("Stopped grabbing frames.")
            except Exception as e:
                print(f"StopGrabbing skipped or failed: {e}")

            # Try to close safely
            try:
                camera_to_close.Close()
                print("Camera closed successfully.")
            except Exception as e:
                print(f"Close skipped or failed: {e}")
                
            
        self.close_secondary_camera()
        # ---- Run cleanup on this instance's camera ----
        _do_cleanup(self.camera)
        self.camera = None   # Remove reference so GC can collect


        print("HSI_Scanner closed gracefully.")

    def __del__(self):
        """Destructor safety net (runs if user forgot close())."""
        try:
            self.close()
        except Exception:
            # Never let errors bubble out of __del__
            pass
        print("HSI_Scanner object destroyed.")

        #should be dispatched!!
    def _set_camera_pixel_format(self, pixel_format):
        """Attempt to set the primary camera's pixel format with fallback to Mono8."""
        try:
            supported_formats = self.camera.PixelFormat.GetSymbolics()
            if pixel_format in supported_formats:
                self.camera.PixelFormat.SetValue(pixel_format)
                print(f"Primary camera pixel format set to: {pixel_format}")
            else:
                print(f"Primary camera does not support {pixel_format}, falling back to Mono8.")
                self.camera.PixelFormat.SetValue("Mono12")
        except Exception as e:
            print(f"Error setting pixel format: {e}. Falling back to Mono8.")
            self.camera.PixelFormat.SetValue("Mono8")

    #should be dispatched!!
    def _match_pixel_format_to_secondary(self): 
        """Match primary camera's pixel format to the secondary camera's format if possible."""
        try:
            secondary_format = self.secondary_camera.PixelFormat.GetValue()
            print(f"Secondary camera pixel format: {secondary_format}")

            # Prefer Mono12 if supported by primary camera (good default for your data)
            if "Mono8" in self.camera.PixelFormat.GetSymbolics():
                self._set_camera_pixel_format("Mono8")
            else:
                # Otherwise, try to match the secondary exactly if supported
                if secondary_format in self.camera.PixelFormat.GetSymbolics():
                    self._set_camera_pixel_format(secondary_format)
                else:
                    self._set_camera_pixel_format("Mono12")
        except Exception as e:
            print(f"Failed to match pixel format: {e}")
            self._set_camera_pixel_format("Mono8 or Mono12")

    def _initialize_camera_settings(self) -> Tuple[float, float]:
            # Disable auto gain/exposure
        self.camera.GainAuto.SetValue('Off')
        self.camera.ExposureAuto.SetValue('Off')

        # Set default values
        self.gain_min = self.camera.Gain.Min
        self.gain_max = self.camera.Gain.Max
        self.exp_min = self.camera.ExposureTime.Min
        self.exp_max = self.camera.ExposureTime.Max
        
        #We set class variables here for min/max gain/exposure

        gain = self.gain_min 
        exposure = self.exp_min
        
        return exposure, gain

    
    def turn_on_light(self, percentage_power: int):
        '''Default is for brightfield as of right now; potentially dangerous for camera saturation reasons but options such as:
            1. Covering sample plane with opaque cover
            2. Being able to abort program mid-scan
            3. Able to change exposure/gain mid-scan makes this safer'''
            #careful of self.Stage and self. ; aliases or wrappers because implementation can be completely different
            #managling to avoid namespace conflict
        if percentage_power is None:
            percentage_power = -1
        self.Stage._turn_on_light(actual_percentage_power= percentage_power if percentage_power != -1 else self.Stage._turn_on_light(actual_percentage_power = 99))


    def turn_off_light(self):
        self.Stage._turn_off_light()
        
        # ------------------------- #
    # Helper function
    # ------------------------- #


    class Utils:
        
        stage = None
        
        @staticmethod
        def safe_num(x: float, min_val: float = 0, max_val: float = 2001) -> float: 
            if not isinstance(x, float):
                raise TypeError(f"x must be float, goat {type(x)}")
            x *= -1 if x < 0 else 1
            return max(min_val, min(x, max_val))
                #limits inputs from 0-128 similar to a short from Java
        
        @staticmethod
        def metric_to_asi(
            single_mm: Optional[Union[int, float]] = None,
            mm_iter: Optional[Tuple[Union[int, float], Union[int, float], Union[int, float]]] = None,
            asi_to_mm: bool = False
        ) -> Union[int, Tuple[int, int, int]]:
            """Convert mm to ASI units (tenths of microns) or back if `asi_to_mm` is True."""
            asi_scalar = 10000 if not asi_to_mm else 1 / 10000

            if mm_iter is not None:
                return tuple(int(x * asi_scalar) for x in mm_iter)
            elif single_mm is not None:
                return int(single_mm * asi_scalar)
            else:
                raise ValueError("Either 'single_mm' or 'mm_iter' must be provided.")

# ------------------------- #
# AUTOFOCUS METHODS ; unsure if need new class
# ------------------------- #
        class Autofocus:
            '''Temporary container for future brightfield autofocus code'''
            # === Add these to HSI_Scanner.Utils ===
            
            @classmethod
            def acquire_and_process_once_with(cls,
                *,
                camera: pylon.InstantCamera,
                acquire_fn,                           # bound: self.acquire_frame()
                process_fn,                           # bound: self.process_frame(img, cube_frames)
                cube_frames: list[np.ndarray],
                on_frame=None,                        # optional debug hook: (i, img)
                index: int = 0,
            ) -> tuple[bool, int]:
                """
                Acquire + process exactly one frame using caller's dynamic methods.
                Ensures the camera is open & grabbing (LatestImages) so acquire_fn() has data.
                Returns (got_frame, next_index).
                """
                # --- ensure camera is actually grabbing ---
                if not camera.IsOpen():
                    camera.Open()
                if not camera.IsGrabbing():
                    # don't fight your later strategy; this is just to get a frame now
                    cls._ensure_latest_images(camera, n_buffers=8, force=False)

                # --- call the user's dynamic acquire/process methods ---
                _, img = acquire_fn()
                if img is None:
                    return False, index

                if callable(on_frame):
                    try:
                        on_frame(index, img)
                    except Exception:
                        pass

                process_fn(img, cube_frames)
                return True, index + 1


            @classmethod
            def show_live_preview_with(cls,
                *,
                camera: pylon.InstantCamera,
                acquire_fn,
                process_fn,
                cube_frames: list[np.ndarray],
                max_frames: int | None = None,
                ensure_latest: bool = True,
                n_buffers: int = 8,
                on_frame: Optional[Callable[[int, np.ndarray], None]] = None,  # debug hook
                start_index: int = 0,
            ) -> int:
                """
                Drive a live preview loop using the caller's dynamic acquire/process methods.
                Returns the total number of frames processed.
                """
                assert camera is not None, "Camera is None"
                if ensure_latest:
                    cls._ensure_latest_images(camera, n_buffers=max(n_buffers, 4), force=False)

                i = start_index
                processed = 0
                try:
                    while True:
                        got, i = cls.acquire_and_process_once_with(
                            camera=camera,
                            acquire_fn=acquire_fn,
                            process_fn=process_fn,
                            cube_frames=cube_frames,
                            on_frame=on_frame,
                            index=i,
                        )
                        if got:
                            processed += 1

                        if max_frames is not None and processed >= max_frames:
                            break

                        # allow quick exit if your process_fn window has focus
                        k = cv2.waitKey(1) & 0xFF
                        if k in (27, ord('q')):  # ESC / 'q'
                            break
                finally:
                    # leave window cleanup to your process_fn if desired
                    pass

                return processed

            @classmethod    
            def tenengrad_score_mono(cls, mono: np.ndarray, ksize: int = 3) -> float:
                """Tenengrad focus metric for mono uint8/uint16 images."""
                # DEBUG prints
                print(f"[DEBUG] Utils.tenengrad_score_mono: dtype={mono.dtype}, shape={mono.shape}")
                if mono.dtype == np.uint8:
                    f = mono.astype(np.float32) / 255.0
                    print("[DEBUG] normalize uint8 /255")
                elif mono.dtype == np.uint16:
                    f = mono.astype(np.float32) / 65535.0
                    print("[DEBUG] normalize uint16 /65535")
                else:
                    f = mono.astype(np.float32)
                    f = (f - f.min()) / (f.max() - f.min() + 1e-12)
                    print("[DEBUG] normalize dynamic range")
                sx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=ksize)
                sy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=ksize)
                score = float(np.hypot(sx, sy).mean())
                print(f"[DEBUG] tenengrad mean={score:.6f}")
                return score
        
        
            # 1) _ensure_latest_images: only stop/start if we really need to
            @classmethod
            def _ensure_latest_images(cls, camera: pylon.InstantCamera, n_buffers: int, *, force: bool = False) -> bool:
                """
                Ensure camera is grabbing with LatestImages and enough buffers.
                Returns True if we changed state here.
                """
                print(f"[DEBUG] Ensure LatestImages: n_buffers>={n_buffers}, force={force}")
                started_or_restarted = False

                if not camera.IsOpen():
                    print("[DEBUG] Camera not open. Opening now...")
                    camera.Open()

                # Try to bump buffer count ONLY when not grabbing (some transports lack node)
                if not camera.IsGrabbing():
                    try:
                        node = camera.GetNodeMap().GetNode("MaxNumBuffer")
                        if node and pylon.IsWritable(node):
                            current = camera.MaxNumBuffer.GetValue()
                            target = max(current, n_buffers)
                            if target != current:
                                camera.MaxNumBuffer.SetValue(int(target))
                                print(f"[DEBUG] MaxNumBuffer: {current} → {target}")
                        else:
                            print("[DEBUG] MaxNumBuffer not available/writable; skipping.")
                    except Exception as e:
                        print(f"[WARN] Could not set MaxNumBuffer: {e}")

                # If already grabbing and not forcing, keep going as-is
                if camera.IsGrabbing() and not force:
                    print("[DEBUG] Already grabbing; keep current strategy.")
                    return False

                # Start (or restart if force=True)
                if camera.IsGrabbing():
                    print("[DEBUG] Restart grabbing due to force=True.")
                    camera.StopGrabbing()

                camera.StartGrabbing(pylon.GrabStrategy_LatestImages)
                print("[DEBUG] Camera grabbing with GrabStrategy_LatestImages.")
                started_or_restarted = True
                return started_or_restarted


            # 2) grab_latest_n_mono: assume strategy already set; do NOT stop/start here
            @classmethod
            def grab_latest_n_mono(cls, camera: pylon.InstantCamera, n: int, flush_first: bool = True) -> list[np.ndarray]:
                print(f"[DEBUG] Grabbing latest {n} frames (flush_first={flush_first})…")

                # sanity: ensure camera is open & grabbing, but DO NOT restart if it already is
                if not camera.IsOpen():
                    print("[DEBUG] Camera not open. Opening now…")
                    camera.Open()
                if not camera.IsGrabbing():
                    print("[DEBUG] Camera not grabbing. Starting LatestImages…")
                    camera.StartGrabbing(pylon.GrabStrategy_LatestImages)

                # drain at most, say, 50 buffers to avoid pathological loops
                if flush_first:
                    drained = 0
                    for _ in range(50):
                        res = camera.RetrieveResult(0, pylon.TimeoutHandling_Return)
                        if res is None:
                            break
                        drained += 1
                        res.Release()
                    print(f"[DEBUG] Flushed {drained} old frames.")

                frames: list[np.ndarray] = []
                for i in range(n):
                    res = camera.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
                    try:
                        if not res.GrabSucceeded():
                            raise RuntimeError(f"Grab failed: {res.GetErrorCode()} {res.GetErrorDescription()}")
                        arr = res.Array
                        if arr.ndim == 3 and arr.shape[-1] == 1:
                            arr = arr[..., 0]
                        frames.append(np.copy(arr))  # clone
                        print(f"[DEBUG] Frame {i+1}/{n} shape={arr.shape} dtype={arr.dtype}")
                    finally:
                        res.Release()
                return frames


            # 3) score_latest_n_frames: now just read/score (no reconfigure)
            @classmethod
            def score_latest_n_frames(cls, camera: pylon.InstantCamera, n: int = 5) -> tuple[float, float]:
                print(f"[DEBUG] Scoring latest {n} frames…")
                frames = cls.grab_latest_n_mono(camera, n=n, flush_first=True)
                scores = np.array([cls.tenengrad_score_mono(f) for f in frames], dtype=np.float64)
                mean = float(scores.mean())
                std  = float(scores.std(ddof=1) if len(scores) > 1 else 0.0)
                print(f"[DEBUG] Scores={np.round(scores,5)} mean={mean:.5f} std={std:.5f}")
                return mean, std


            # 4) autofocus: ensure strategy ONCE, then loop without restarting
            @classmethod
            def autofocus_simple_latest_few(cls,
                camera: pylon.InstantCamera,
                Stage,
                *,
                settle_s: float = 5.0,
                n_latest: int = 5,
                step_asi: float = 50,
                span_steps: int = 2,
                verbose: bool = True
            ):
                print(f"[DEBUG] AF start: step_asi={step_asi}, span_steps={span_steps}, n_latest={n_latest}")
                if not camera.IsOpen():
                    print("[DEBUG] Camera not open. Opening now…")
                    camera.Open()

                # set LatestImages once (no force restarts during loop)
                cls._ensure_latest_images(camera, n_buffers=max(n_latest, 4), force=False)

                def move_rel_units(delta_units: int):
                    mm = delta_units
                    print(f"[DEBUG] Move stage: {delta_units:+d} ASI units → {mm:+.6f} mm (relative)")
                    Stage.move_stage(z=mm, relative=True)

                ks = list(range(-span_steps, span_steps + 1))
                offsets_units = [int(k * step_asi) for k in ks]  # ensure int for ASI units
                print(f"[DEBUG] Offsets (ASI units): {offsets_units}")

                history = []
                moved_units = 0

                try:
                    for off in offsets_units:
                        delta = off - moved_units
                        if delta:
                            move_rel_units(delta)
                            moved_units = off
                            print(f"[DEBUG] Settling {settle_s}s…")
                            time.sleep(settle_s)
                            cur_z = Stage.get_position('Z')/10
                            print(f"Starting z-stack (z={off}) at (z={cur_z:.4f}, um")
                            
                        mean, std = cls.score_latest_n_frames(camera, n=n_latest)
                        history.append({"offset_units": off, "mean": mean, "std": std})
                        if verbose:
                            print(f"[AF] z={off/10.0:+.1f} µm (ASI {off:+d}) mean={mean:.4f} std={std:.4f}")

                    best = max(history, key=lambda d: d["mean"])
                    best_off = int(best["offset_units"])
                    print(f"[DEBUG] Best offset: {best_off} ASI (mean={best['mean']:.4f})")

                    back = best_off - moved_units
                    if back:
                        move_rel_units(back)
                        moved_units = best_off
                        time.sleep(settle_s)

                    if verbose:
                        print(f"[AF] BEST z={best_off/10.0:+.1f} µm (ASI {best_off:+d}) mean={best['mean']:.4f} std={best['std']:.4f}")

                    return {
                        "history": history,
                        "best_offset_units": best_off,
                        "best_mean": best["mean"],
                        "best_std": best["std"],
                    }
                finally:
                    print("[DEBUG] AF done (camera left grabbing in LatestImages).")



            '''
                /
            def autofocus(self, camera: pylon.InstanceCamera):
                
                camera.
                score = compute_tenengrad()
                
            def compute_tenengrad(np.array = image):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Sobel filter in X direction
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Sobel filter in Y direction
                tenengrad = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
                return np.mean(tenengrad)  # Return mean gradient magnitude as focus score
            '''
        
        @staticmethod
        def bin_cube_axes(cube: np.ndarray, bin_factors=(1, 1, 1)) -> tuple[np.ndarray, int]:
            """
            Bin a 3D hyperspectral cube along all three axes.

            Parameters
            ----------
            cube : np.ndarray
                Hyperspectral cube with shape (spectral, spatial, lines).
            bin_factors : tuple[int, int, int], optional
                Binning factors as (spectral_bin, spatial_bin, line_bin).
                Defaults to (1, 1, 1) = no binning.

            Returns
            -------
            np.ndarray
                Binned cube with shape:
                (spectral // spectral_bin, spatial // spatial_bin, lines // line_bin)
            int
                spectral_bin (returned for compatibility)
            """
            if cube.ndim != 3:
                raise ValueError("Expected cube with shape (spectral, spatial, lines)")

            spectral, spatial, lines = cube.shape
            spectral_bin, spatial_bin, line_bin = bin_factors

            # If no binning requested, return unchanged
            assert cube is not None
            if spectral_bin == 1 and spatial_bin == 1 and line_bin == 1:
                return cube, spectral_bin

            # New sizes after binning
            S_new = spectral // spectral_bin
            Y_new = spatial // spatial_bin
            X_new = lines // line_bin

            # Truncate so dimensions divide evenly
            cube = cube[:S_new * spectral_bin, :Y_new * spatial_bin, :X_new * line_bin]

            # Reshape and average
            cube = cube.reshape(S_new, spectral_bin, Y_new, spatial_bin, X_new, line_bin)
            cube_binned = cube.mean(axis=(1, 3, 5))

            return cube_binned, spectral_bin

    # ------------------------- #
    # Refactored frame/tile helpers 
    # ------------------------- #
    def acquire_frame(self):
        """Grab a single frame from camera and return (grab, img)."""
        assert self.camera is not None
        grab = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

        img = None
        # --- new guard to avoid NULL pointer crash ---
        if grab is None:
            print("[WARN] RetrieveResult returned None (camera not grabbing yet?)")
            return None, None

        if grab.GrabSucceeded():
            img = grab.Array
        else:
            print("grab.GrabSucceeded was not successful")
        return grab, img


    def process_frame(self, img, cube_frames):
        """Append to cube, show live preview, handle keyboard if enabled."""
        cube_frames.append(img)
        cv2.imshow("Live Preview", img)
        #print("processing_frame")
        if self.enable_keyboard_control:
            self.handle_keyboard_controls()

        cv2.waitKey(1)

    def move_camera_home(self, custom_home_spot: bool = False, *args):
        if (not custom_home_spot) or (not args) : #or not args or args[0] == (0, 0, 0): this part of code may be unnecessary because if the start for custom home is (0, 0, 0) we won't pop (0,0,0 out I think)
            # Default home position
            x, y, z = 0.0, 0.0, 0.0
            self.Stage.move_stage(x=x, y=y, z=z)
            print(f"MOVING TO HOME POSITION: {x}, {y}, {z}")
            self.Stage.wait_for_device()

            return None
        else:
            # Use custom home position
            future_homes = self.Stage.custom_starting_home_points(home_points=list(args))
            self.Stage.wait_for_device()

            return future_homes

    # ------------------------- #
    # Main scanning loop
    # ------------------------- #

    def pre_scan_check(
        self,
        rows,
        cols,
        x_distance_mm,
        y_distance_mm,
        stage_speed,
        save_folder,
        LED_power,
        bin_factors,
        _pause_between_grids: float = 1,
        custom_home_spot: bool = True,
        home_points: list | None = None,
    ) -> bool:
        """
        Pre-scan wrapper around the pushbroom scan.

        This version mirrors the behavior of the old recursive run_pushbroom_scan:
        - If custom_home_spot=True and multiple home points are provided, we perform
        one full scan per home point.
        - If custom_home_spot=False, we treat (0,0,0) as the only home and scan once.
        """

        self.turn_on_light(percentage_power = LED_power)   # <--- Add this at start

        # If home_points not passed, initialize to a single default tuple
        # (same semantics as the old recursive version)
        home_points_current = home_points if home_points else [(0, 0, 0)]  # List with a single tuple

        # Baseline speeds
        self.Stage.set_max_speed('X', 1.5)
        self.Stage.set_max_speed('Y', 2)

        os.makedirs(save_folder, exist_ok=True)

        make_debug_file = MAKE_DEBUG_FILE
        if make_debug_file:
            # should factor this into general debug call eventually like
            # seen in the scan_tile section
            debug_file_callable = DebugTimer.make_local_debug_txt_file_json


        def _run_push_broomscan(home_points_for_naming) -> None:
            """
            Pushbroom scan: one cube per grid tile, with X-only motion per tile,
            and Y-only stepping per new row. Uses snake (S) pattern.

            Notation:
            (a, b) = grid space indices = (row, col) in matrix convention
            (x, y, z) = stage real-space coordinates in mm
            """

            # We assume the stage is already at the correct home position here.

            #self.stage.get_position()
            '''
            for item in x:
            x0_asi, y0_asi, z0_asi  = (
                self.Stage.get_position('X'), #('X'#,
                self.Stage.get_position('Y'),
                self.Stage.get_position('Z')
                )
        
            f = lambda v: math.floor(v)
            v = 50 #5 microns in asi terms
            if (f(x0_asi) < v) or (f(y0_asi) < v) or (f(z0_asi) < v): #works as long as the true machine zero isn't changed
                pass
            else:
                #should add args to get_position
            '''

            # Anchor x0_asi, y0_asi at the *current* home
            time.sleep(1)
            x0_asi = self.Stage.get_position_trycasters('X')
            time.sleep(0.2)
            y0_asi = self.Stage.get_position_trycasters('Y')

            time.sleep(1)
            print(f'type of x0_asi, y0_asi are {type(x0_asi), type(y0_asi)} with vals {y0_asi, x0_asi}')

            # Guard against None positions (prevents TypeError later)
            if x0_asi is None or y0_asi is None:
                raise RuntimeError(f"Stage home position read failed: x0_asi={x0_asi}, y0_asi={y0_asi}")

            if TIME_TO_MANUALLY_FOCUS == True:
                time.sleep(10)
            print("time given to manually focus")

            # Set X and Y stage speed and convert mm to ASI units
            self.Stage.set_max_speed('X', stage_speed)
            #self.Stage.set_max_speed('Y', stage_speed)
            
            x_step: float = HSI_Scanner.Utils.metric_to_asi(single_mm=x_distance_mm)
            y_step: float = HSI_Scanner.Utils.metric_to_asi(single_mm=y_distance_mm)

            # also use the current X as the subscan anchor

            # Loop over rows (a); logic for the scan pattern not the actual recording of data itself
            #try:
            for a in range(rows):
                # Move Y to correct position for this row in real space (a → y)
                y_start = (a * y_step) + y0_asi
                self.Stage.set_max_speed('Y', 2) #set a second time for both modularity and seems to be issues if program is running for >24 hours - 36 hours with set speed?

                self.Stage.move_stage(y=y_start)
                self.Stage.wait_for_device()
                time.sleep(1)
                #y_real_pos = self.Stage.get_position('Y')
                #print(f"Moved to Y row {a} at {y_real_pos / 10000:.4f} mm") <-- can cause crashes so will remove

                # Determine snake pattern
                is_even = a % 2 == 0
                direction = +1 if is_even else -1

                # choose row start based on snake direction (anchor to subscan home)
                if is_even:
                    x_start = x0_asi
                else:
                    x_start = x0_asi + (cols) * x_step
                
                debug_var = self.Stage.set_max_speed('X', 0)
                #self.Stage.set_max_speed('X', 1.5) #speed should implicitly be 
                self.Stage.move_stage(x=x_start)
                self.Stage.wait_for_device()
                print("CHECKING IF THE CHANGE IN X SUPERGRID GOES AT PROPER SPEED")
                self.Stage.set_max_speed('X', stage_speed)

            #This should be logic for within subgrid; delta x movement 
                for b in range(cols):
                    #try:
                        delta = direction * x_step
                        try:
                            # Attempt 1: real ASI-2000 position
                            x_real = self.Stage.get_position('X') / 10000.0
                            y_real = self.Stage.get_position('Y') / 10000.0

                        except Exception as asi_err:
                            try:
                                # Attempt 2: calculated fallback from outer nested loop
                                x_real = x_start 
                                y_real = y_start

                            except Exception as calc_err:
                                raise RuntimeError(
                                    "Neither ASI-2000 position query nor calculated fallback succeeded"
                                ) from calc_err

                            
                        print(f"Starting tile (a={a}, b={b}) at (x={x_real:.4f}, y={y_real:.4f}) mm")

                        self.scan_tile(
                            a, b, bin_factors, save_folder,
                            delta_x=delta,
                            actual_super_grid_coords=(x_real, y_real)
                        )
                    #except Exception as e:
                        #print(f"[WARN] tile (a={a}, b={b}) failed: {e}")
                        #continue
                # optional per-row cleanup
                gc.collect(0)  # young-generation GC; cheaper
                time.sleep(_pause_between_grids)

            #finally:
                # Ensures cleanup even if an exception interrupts the row loop
                print("[CLEANUP] Releasing resources and flushing buffers...")
                try:
                    self.camera.StopGrabbing() #real method but 
                except Exception:
                    pass
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
                gc.collect(2)  # full GC to flush frames/buffers

        # ---- Iterative version of the old recursive home-point loop ----
        while True:
            # Move to the next home point (or to (0,0,0) if custom_home_spot is False)
            self.Stage.set_max_speed('X', 1.5)
            self.Stage.set_max_speed('Y', 2)
            
            if custom_home_spot:
                home_points_left = self.move_camera_home(custom_home_spot, *home_points_current)
            else:
                # ignore home_points when not using custom homes; stage home is always (0,0,0)
                home_points_left = self.move_camera_home(custom_home_spot)

            # Run one full pushbroom scan at this home
            _run_push_broomscan(home_points_for_naming=home_points_current)

            # Decide whether to continue like the recursive version did
            # In the original:
            #   if home_points_left and len(home_points_left) > 0:
            #       recurse with home_points=home_points_left
            #   else:
            #       cleanup and stop
            if custom_home_spot and home_points_left and len(home_points_left) > 0:
                # prepare for next iteration with remaining home points
                home_points_current = home_points_left
                continue
            else:
                # No more custom homes (or custom_home_spot=False): exit loop
                break

        # ---- Post-scan debug + shutdown (base case of old recursion) ----
        DebugTimer.summarize_events()
        if make_debug_file:
            debug_file_callable(override=OVERRIDE_DEBUG_FILE)
        self.close()
        self.close_secondary_camera()
        self.Stage.set_max_speed('X', 1.5)
        self.Stage.set_max_speed('Y', 2)
        self.move_camera_home(custom_home_spot=False)
        self.turn_off_light()
        print("Scan complete. Light turned off.")

      
    def scan_tile(self, row, col, bin_factors, save_folder, delta_x: float,  home_points_for_naming=None, actual_super_grid_coords: tuple = ()):

        """Scan a single tile at (row, col), grab frames, build cube, and save.


        Notation:
        (a, b) = grid space indices = (row, col)
        (x, y, z) = stage real-space coordinates in mm
        
        'Supergrid' from <actual_super_grid_coords> is refering to if you want to, within a single usage of this program with no re-scan, use the row x col S-snake pattern multiple times;
        As of V8 automatically defaults to true if you want to scan a dish/well item in <Make_GUI()>. Was made like this that way there is exactly
        only 1 save folder and expected per <Scanner> instaniation. Harder for user to mess up previous files considering we also don't overwrite old files by default.
        """
        
        frame_debug_type = "500"          # Placeholder variable for GUI-User or Private Logic logic
        grab_strategy_type = "OneByOne"   # Default grab strategy
        cube_frames = []
        make_debug_file = True
        
    
        # NOTE:
        # Currently using static debug/strategy values (hard-coded for testing).
        # In the future these will be set via GUI inputs (e.g., dropdowns).
        # The combo_map approach lets us support all (debug_type, strategy_type) pairs
        # without writing dozens of case statements.

        # --- Debug function builder ---
        def make_debug_fn(a: int):
            def debug_fn(i: int, img):
                if i % a == 0:
                    print(f"Frame stats - min: {np.min(img)}, max: {np.max(img)}")
                    print(img.shape)
            return debug_fn
            
        # --- Debug map ---
        debug_map = {
            "0":   make_debug_fn(1),
            "100": make_debug_fn(100),
            "500": make_debug_fn(500),
        }

        # --- Grab strategy map ---
        strategy_map = {
            "OneByOne": pylon.GrabStrategy_OneByOne,
            "LatestImageOnly": pylon.GrabStrategy_LatestImageOnly,
            "LatestImages": pylon.GrabStrategy_LatestImages,
            "UpcomingImage": pylon.GrabStrategy_UpcomingImage,
        }


        combo_map = {
            (d, s): (debug_map[d], strategy_map[s])
            for d, s in product(debug_map.keys(), strategy_map.keys())
        }

        # --- Select debug_fn + grab_strategy ---
        if self.debug_frames:
            frame_debug_fn, grab_strategy = combo_map.get(
                (frame_debug_type, grab_strategy_type),
                (lambda i, img: None, pylon.GrabStrategy_OneByOne),
            )
        else:
            frame_debug_fn, grab_strategy = (lambda i, img: None, pylon.GrabStrategy_OneByOne)

        custom_timer.start()
    
        # micro-caches (cheaper attribute lookups)
        Stage_send    = self.Stage.send_command_special2
        Stage_send_buffered = partial(Stage_send, axis="X", verbose=False)
        acquire_frame = self.acquire_frame
        process_frame = self.process_frame
        frame_fn      = frame_debug_fn

        # bind _snapshot_event.set if available
        try:
            getattr(self, "_snapshot_event")  # just check existence
            _check_snapshot = self._snapshot_event.set
        except AttributeError:
            DebugExceptions.MyAttributeError.custom_warning("_snapshot_event", o=self)
            _check_snapshot = lambda: None  # no-op if missing

        # call safely
        _check_snapshot()
        
        i = 0
        print(f"MOVING STAGE NOW (row={row}, col={col})")

        # Throttle status polls a bit (serial is slow). Tune 2..8; 4 is a safe start. Found worse results polling every second frame
        poll_div = 2
        
        self.Stage.move_stage(x=delta_x, relative=True)

    # --- Start grabbing with chosen strategy (unchanged) ---
        self.camera.StartGrabbing(grab_strategy)

        try:
            while True:
                # ---- stop when decelerating (your exact rule preserved) ----
                if (i % poll_div) == 0:
                    val = Stage_send_buffered()  # 0–255 status byte
                    # if (val & 0x30) == 0x30:   # safer mask (optional)
                    if ((val == 31  or val == 10)) and i > 15:
                        break

                # ---- acquire + process (unchanged semantics) ----
                grab, img = acquire_frame()
                if img is not None:
                    process_frame(img, cube_frames)
                    frame_fn(i, img)  # no-op if disabled
                    i += 1
                # else:
                #     # comment out noisy print in hot path
                #     # print("IMAGE IS NONE")
                #     pass

                grab.Release()

        finally:
            print("Closing window")
            self.camera.StopGrabbing()
            cv2.destroyAllWindows()
            
        timed_event = custom_timer.stop() #<-- for checking x_distance travel time should be around 8.2 seconds on 1/0.0625 mm/s * 0.51 mm distance
        _check_snapshot()


        if not isinstance(cube_frames, list):
            print("Cube_frames is either not a list or is uninitialized")
        
        if len(cube_frames) == 0:
            print(f"⚠ No frames within cube_frames numpy array at row={row}, col={col}")
            return

        cube = np.stack(cube_frames, axis=-1)
        print(f"[DEBUG] Stacked cube shape (after stacking frames): {cube.shape}")

        cube = np.transpose(cube, (1, 2, 0))
        print(f"[DEBUG] Cube shape after transpose (1,2,0): {cube.shape}")

        cube = np.moveaxis(cube, 2, 0)
        print(f"[DEBUG] Cube shape after moveaxis (2->0): {cube.shape}")

        cube = HSI_Scanner._reverse_band_order(cube)
        print(f"[DEBUG] Completed reversing band order on cube")
        #print(f"[DEBUG] Cube shape[0] after moveaxis (2->0): {cube.shape[0]}")
        #print(f"[DEBUG] Cube shape after moveaxis (2->0): {cube.shape}")

        binned_cube, spectral_bin_factor = HSI_Scanner.Utils.bin_cube_axes(cube, bin_factors=bin_factors)
        
        '''
        print(f"[DEBUG] Binned cube shape: {binned_cube.shape}, spectral bin factor: {spectral_bin_factor}")
        flat_values = binned_cube[:, 0, 0]
        for i, v in enumerate(flat_values):
            print(f"Band {i:03d}\t{v:.6f}")
        '''
        bands = binned_cube.shape[0]  # number of spectral bands
        print(f"[DEBUG] Total spectral bands (post-binning): {bands}")
        print(f"[DEBUG] BinnedCube shape[0] after moveaxis (2->0): {binned_cube.shape[0]}")
        print(f"[DEBUG] BinnedCube shape after moveaxis (2->0): {binned_cube.shape}")
        # show a few pixels per band — truncated display
        flat_values = binned_cube[:, 0, 0].ravel()
        preview_len = 50  # show first 18
        print("[DEBUG] Example pixel values (pre-wavelength, post-binning):")
        for i in range(0, len(flat_values), preview_len):
            print("\t" + "\t".join(f"{v:.3f}" for v in flat_values[i:i+preview_len]))
            if i > preview_len * 2:  # stop after ~3 lines
                print("\t...")
                break

        spectral_axis = np.arange(bands)
        print(f"[DEBUG] spectral_axis = {spectral_axis}")

        # ===== NEW: compute wavelengths from band indices ONLY (no intensities) =====
        # NOTE: Use your device's unbinned Y offset here (from config report). 192 is your current default.
        #device_y_offset_unbinned = 100  # <-- replace with actual config OffsetY when available

        # Build raw-sensor pixel centers m (unbinned coords) from band indices
        m = HSI_Scanner._solve_pixel_from_band(
            spectral_axis,
            bin_factor=spectral_bin_factor,
            #off_set=device_y_offset_unbinned,  # unbinned device offset (Oy)
            reversed_bc_pikaXC2=False,                    # XC2: lower λ at higher pixel index
            #total_pixels=1200,                 # XC2 sensor vertical size (unbinned)
            median_on=True,                    # use center-of-bin (0.5*B - 0.5)
            decimals=None                      # set to an int (e.g., 3) if you want m rounded
        )

        # Evaluate λ = A*m^2 + B*m + C  (no bin_factor scaling on λ)
        wavelengths = HSI_Scanner.lambda_from_bands(m)

        print(f"[DEBUG] wavelengths (nm):")
        for i in range(0, len(wavelengths), 18):
            print("\t" + "\t".join(f"{w:.2f}" for w in wavelengths[i:i+18]))
            if i > 36:
                print("\t...")
                break
        print(f"[DEBUG] Calculated wavelengths shape: {wavelengths.shape if hasattr(wavelengths, 'shape') else type(wavelengths)}")
        print(f'wavelengths nm = {wavelengths}')

        meta_filename = f"cube_y{row}_x{col}_meta.json"
        data_filename = f"cube_y{row}_x{col}.npy"

        # Convert actual_cords to ASI units string if present
        ''''''
        super_x = actual_super_grid_coords[0]
        super_y = actual_super_grid_coords[1]
        super_x = str(super_x)
        super_y = str(super_y)
        '''
        actual_super_grid_coords_string = (
            '_'.join(str(x) for x in actual_super_grid_coords)
            if actual_super_grid_coords and isinstance(actual_super_grid_coords, tuple)
            else ""
        )
     
        
        # Make sure `expected_super_grid_cord` is defined and converted to string
        expected_super_grid_coords = str(expected_super_grid_cord)
        
    
        # Compose the logic string (REMOVE extra spaces in keys like "actual_ ")
        main_file_logic = f"actual_{actual_super_grid_coords_string}_expected_{expected_super_grid_coords}_"
        '''

         # --- Derive expected super grid coords ---
        expected_super_grid_coords = None
        if home_points_for_naming and isinstance(home_points_for_naming, (list, tuple)) and home_points_for_naming:
            expected_super_grid_coords = home_points_for_naming[0]

        # --- Convert if exists ---
        if expected_super_grid_coords and any(expected_super_grid_coords):
            try:
                expected_super_grid_coords = HSI_Scanner.Utils.metric_to_asi(
                    mm_iter=expected_super_grid_coords,
                    asi_to_mm=True
                )
            except Exception as e:
                print(f"[WARN] Failed to convert expected_super_grid_coords: {e}")
                expected_super_grid_coords = None

        # --- Stringify (after conversion or fallback) ---
        if expected_super_grid_coords and any(expected_super_grid_coords):
            expected_super_grid_coords = "_".join(f"{v:.3f}" for v in expected_super_grid_coords)
        else:
            expected_super_grid_coords = "none"

        main_file_logic = f"actual_{super_x}_{super_y}_expected_{expected_super_grid_coords}_"

        # Update filenames
        meta_filename = main_file_logic + meta_filename
        data_filename = main_file_logic + data_filename

        # Debug prints
        print(f"[DEBUG] Default data file name for saving: {data_filename}")
        print(f"[DEBUG] Default meta file name for saving: {meta_filename}")

        # Set the GUI meta file
        Make_GUI.set_meta_file(
            directory=Make_GUI.cur_dir if Make_GUI.cur_dir else SAVE_DIR_CONST,
            filename=meta_filename
        )

        print(f"[DEBUG] Set meta file to: {Make_GUI.meta_file}")

        band_wavelength = True
        metafile_printout = list(zip(spectral_axis.tolist(), wavelengths.tolist())) if band_wavelength else wavelengths.tolist()
        Make_GUI._write_to_data_metafile(wavelength_range_nm = metafile_printout)

        save_path = os.path.join(save_folder, data_filename)
        save_path = os.path.normpath(save_path)
        print(f"[DEBUG] Normalized save path: {save_path}")

        if os.path.exists(save_path):
            base, ext = os.path.splitext(save_path)
            save_path = base + "_copy" + ext
            print(f"[DEBUG] Save path exists, changed to: {save_path}")

        np.save(save_path, binned_cube)
        print(f"[INFO] Saved cube to: {save_path}")

        del binned_cube, cube, cube_frames, flat_values, spectral_axis, wavelengths
        gc.collect()
        
        #to_delete: Optional[Tuple[str, ...]] = ("binned_cube", "cube", "img", "cube_frames", "flat_values")
                            
        #custom_debug_exceptions.delete_global_names(to_delete)
        #DebugExceptions.delete_global_names(*to_delete)
        
    

    ########SETTING ROI##########
            ############## '''
    #should be static and psuedo-private/protected because should be set ON SPECIFIC CAMERA OF INTEREST and should not be set by user or accessible

    @classmethod
    def handle_precision(cls, sub_call: Callable[[], T]) -> T:
        '''Need to know how large of spectral wavelengths to return after converting from: band --> wavelength'''
        # NOTE: returning the sub_call() result as-is; keep helpers below for later use.
        temp_var_for_ide_parse = sub_call()  # FIX: remove trailing colon and invalid type syntax

        def get_precision_of_coefficients():
            lowest_sig = min(
                _count_sigfigs_from_thistype(cls.A),  # FIX: qualify via cls.*
                _count_sigfigs_from_thistype(cls.B),
                _count_sigfigs_from_thistype(cls.C),
            )
            return lowest_sig

        def _count_sigfigs_from_thistype(idx: str | float | Decimal) -> int:
            """Count significant figures from a string like '0.6555092' or '1.37999e-5'."""
            # WARNING: floats don't preserve declared sig figs; prefer str/Decimal in practice.
            if isinstance(idx, (str, Decimal)):  # FIX: isinstance instead of "type(...) is str | float"
                d = Decimal(idx)
            else:
                d = Decimal(str(idx))  # best-effort if float passed
            # Normalize to remove trailing zeros while preserving intended precision
            tup = d.normalize().as_tuple()
            # If it's zero, define 1 sig fig by convention
            if not any(tup.digits):
                return 1
            # Count digits, ignoring leading zeros before decimal in subnormals
            return len([dg for dg in tup.digits if dg != 0]) if d != 0 else 1

        def round_sig(x: np.ndarray, sig: int) -> np.ndarray:
            """Round to a fixed number of significant figures (vectorized)."""
            x = np.asarray(x, dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                mags = np.floor(np.log10(np.abs(x)))  # order of magnitude
            mags[~np.isfinite(mags)] = 0  # handle zeros/nans/infs
            scale = np.power(10.0, sig - 1 - mags)
            return np.round(x * scale) / scale

        def decimals_from_delta(delta_nm: float, min_decimals: int = 0, max_decimals: int = 6) -> int:
            """
            Choose number of decimal places so that the rounding step is <= delta_nm.
            """
            if delta_nm <= 0 or not math.isfinite(delta_nm):
                return max_decimals
            d = max(0, -int(math.floor(math.log10(delta_nm))))
            return max(min_decimals, min(d, max_decimals))
        return temp_var_for_ide_parse
    
    @classmethod
    def _solve_pixel_from_lambda(cls, lmbd_nm: float) -> float:
        """Solve A*m^2 + B*m + C = λ for m (unreflected coordinate)."""
        a, b, c = cls.A, cls.B, (cls.C - lmbd_nm)
        disc = b*b - 4*a*c
        if disc < 0:
            raise ValueError(f"No real solution for λ={lmbd_nm} nm.")
        return (-b + math.sqrt(disc)) / (2*a)


    @classmethod
    def _solve_pixel_from_band(
        cls,
        band_nums: np.ndarray,
        bin_factor: int = 1,
        off_set: float = 100.0,          # device Y-offset in UNBINNED pixels (from config)
        reversed_bc_pikaXC2: bool = True,
        total_pixels: int = 1200,        # <-- use FULL SENSOR HEIGHT for XC2, not ROI height
        median_on: bool = True,
        decimals: int | None = None
    ) -> np.ndarray:
        # NOTE: Do NOT pre-flip the offset. Reflection should be applied to m, once, at the end.

        sub_band_position = (0.5 * bin_factor - 0.5) if median_on else 0.0
        band_nums = np.asarray(band_nums, dtype=np.float64)

        # Unreflected pixel center on the RAW sensor (unbinned coords)
        m = float(off_set) + band_nums * float(bin_factor) + float(sub_band_position)

        # XC2 reversal: lower λ at higher pixel index -> flip about (N - 1)
        if reversed_bc_pikaXC2:
            m = (float(total_pixels) - 1.0) - m

        if decimals is not None:
            m = np.round(m, decimals)

        return m
    
    @classmethod
    def lambda_from_bands(cls, pixel_values: np.ndarray) -> np.ndarray:
        """
        ******UNSURE WHICH WAY THE OFFSET IS EITHER +226 OR MINUS NEED TO TEST BOTH**********
        Map pixel index (spectral axis) -> wavelength λ.
        If reflected=True, physical relation is idx = (N-1) - m  (you observed larger index ⇒ shorter λ).
        Should already be done though in previous func calls
        """
        #return cls.A*(m*m) + cls.B*m + cls.C
        pixel_values = np.asarray(pixel_values, dtype=np.float64)  # FIX: ensure float input to polyval
        return (np.polyval([cls.A, cls.B, cls.C], pixel_values))   # apparently faster than #return cls.A*(m*m) + cls.B*m + cls.C

    @staticmethod
    def _reverse_band_order(cube_or_frame: np.ndarray, spectral_axis: int = 0) -> np.ndarray:
        pass
        #https://resonon.gitlab.io/programming-docs/content/baslerstorage.html

        # -------------------- START CODE --------------------
        # Keep your 'pass' above; here is a safe implementation that preserves shape and dtype.
        # If your 2D line frames are (H, W) with H = spectral (1216), set spectral_axis=0.
        # For cubes (Bands, Rows, Cols) where Bands is spectral, spectral_axis=0 also works.
        return np.flip(cube_or_frame, axis=spectral_axis)
        # -------------------- END CODE --------------------
  
    @classmethod
    def pixel_from_lambda(cls, lmbd_nm: float, total_pixels: int, reflected: bool = True) -> float:
        """
        Inverse of lambda_from_pixel for a single λ.
        """
        m = HSI_Scanner._solve_pixel_from_lambda(lmbd_nm)
        return (float(total_pixels - 1) - m) if reflected else m


    def _align_to_inc(self, val: int, inc: int, vmin: int, vmax: int) -> int:
        """Clamp to [vmin, vmax] and honor hardware increment."""
        val = max(vmin, min(val, vmax))
        if inc <= 1:
            return val
        return vmin + ((val - vmin) // inc) * inc

    def _set_ROI(self):
        #BACKGROUND
        ''' 
            ROI (Region of Interest - within sensor array) according to documentation is  1600 x 924 with; from pylon viewer you can see 1936 x 1216 sensor height (must include edges?) 
                So our starting ROI should be: pixel 605; Ending ROI = 1800 for spectral; assume that the discrepancy from: 1936 rection pylon viewer(basler) --> 1900 y resonon
                                                                                                                        1216 y direction pylon viewer(basler) --> 1200 x resonon
                                                                                                                        is most likely the difference between edge of array and inside?
                                                                                                                            ==> need to add (1936 - 1900) / 2 = 18 to the offset in spatial
                                                                                                                            ==> need to add (1216 - 1200) / 2 = 8 in the spectral dimension
        
            which is different than some of the older files of ~ 1920 x 1200 because an (+- 8? or 12?)offset was implemented on the firmware level; fair assumption it's about evenly diplaced in center.
            
            💡💡SPECTRAL💡💡
                --> how to know exactly is to use reverse of Ax^2 + Bx + C = wavelength; from https://resonon.com/content/files/Resonon---Camera-Data-Sheets-Pika-XC2.pdf... 
                we know we can theoretically sample from 400 nm to 1000 nm: --> y(x) ;
                                                            391 
                                                            400 = ((1.379999) * (10**-5)) * x^2 ((6.555092) * (10**-1))*x +((322.6319) * (10**0)) 
                                                            x = 117.81...
                                                            1000 = = ((1.379999) * (10**-5)) * x^2 ((6.555092) * (10**-1))*x +((322.6319) * (10**0)) 
                                                            x = 1012.41
                                                            Δx = 916 pixels for the entire "spectral" part of the sensor which matches documentation almost perfectly at 1200
            
            
            🌌🌌SPATIAL🌌🌌
            Spatial should be basically linear with only assumption being we pull ±1 pixel due to shannon nyquist?... so if our sensor is 1936; non-edges 1912 and ROI is 1600
            so 19
            
            '''

        # -------------------- START CODE (kept comments above verbatim) --------------------
        cam = self.camera 

        # Make sure camera is idle
        try:
            if cam.IsGrabbing():
                cam.StopGrabbing()
        except Exception:
            pass

        # Turn off centering if present
        for name in ("CenterX", "CenterY", "AutoCenterX", "AutoCenterY"):
            try:
                attr = getattr(cam, name)
                if hasattr(attr, "Value"):
                    attr.Value = False
            except Exception:
                pass

        # Hardware max
        Wmax = cam.Width.Max
        Hmax = cam.Height.Max
        print(f'our cam width max = {Wmax} and height max = {Hmax}')

        # Increments (fallback to 1)
        w_inc = getattr(cam.Width, "Inc", 1) or 1
        h_inc = getattr(cam.Height, "Inc", 1) or 1
        x_inc = getattr(cam.OffsetX, "Inc", 1) or 1
        y_inc = getattr(cam.OffsetY, "Inc", 1) or 1

        # Minimums
        x_min = getattr(cam.OffsetX, "Min", 0)
        y_min = getattr(cam.OffsetY, "Min", 0)

        # === Desired ROI (your target) ===
        #1924 x 1216
        total_spatial = 1936
        total_spectral = 1216
        desired_ox = 168 #100% correct pulled from pylonviewer after loading spectronon
        desired_oy = 192 #100% correct pulled from pylonviewer after loading spectronon
        desired_w  = 1600 #spatial
        desired_h  = 924 #spectral
        # Snap sizes to valid range/inc
        tgt_w = self._align_to_inc(max(1, min(desired_w, Wmax)),  w_inc, cam.Width.Min,  Wmax)
        tgt_h = self._align_to_inc(max(1, min(desired_h, Hmax)),  h_inc, cam.Height.Min, Hmax)

        # Apply sizes first (offset limits depend on size)
        cam.Width.Value  = int(tgt_w)
        cam.Height.Value = int(tgt_h)

        # Recompute legal offset headroom after size
        ox_max = cam.OffsetX.Max   # should be Wmax - tgt_w snapped to x_inc
        oy_max = cam.OffsetY.Max   # should be Hmax - tgt_h snapped to y_inc

        # Target offsets, clamped to legal range and snapped to inc
        tgt_x = self._align_to_inc(
            max(x_min, min(desired_ox, ox_max)),
            x_inc, x_min, ox_max
        )
        tgt_y = self._align_to_inc(
            max(y_min, min(desired_oy, oy_max)),
            y_inc, y_min, oy_max
        )

        # Apply offsets
        cam.OffsetX.Value = int(tgt_x)
        cam.OffsetY.Value = int(tgt_y)

                # --- OPTIONAL: Trim ROI from RIGHT (spatial) and BOTTOM (spectral) without moving offsets ---
        # Keep left/top anchored; shrink size so the cuts come off the right/bottom edges.
        # Set these to the Resonon margins you want to remove:
        trim_right_px  = 0   # spatial (X)
        trim_bottom_px = 0  # spectral (Y)

        # Current sizes
        cur_w = int(cam.Width.Value)
        cur_h = int(cam.Height.Value)

        # New desired sizes (offsets unchanged)
        desired_w_after_trim = max(1, cur_w - trim_right_px)
        desired_h_after_trim = max(1, cur_h - trim_bottom_px)

        # Snap to valid ranges/increments
        tgt_w2 = self._align_to_inc(min(desired_w_after_trim, Wmax), w_inc, cam.Width.Min,  Wmax)
        tgt_h2 = self._align_to_inc(min(desired_h_after_trim, Hmax), h_inc, cam.Height.Min, Hmax)

        # Apply sizes first (this updates allowed Offset*.Max)
        cam.Width.Value  = int(tgt_w2)
        cam.Height.Value = int(tgt_h2)

        # Keep offsets the same to ensure trimming occurs on right/bottom only.
        # (Re-assign via _align_to_inc to satisfy hardware bounds after size change.)
        ox_max = cam.OffsetX.Max
        oy_max = cam.OffsetY.Max
        cam.OffsetX.Value = int(self._align_to_inc(max(x_min, min(cam.OffsetX.Value, ox_max)), x_inc, x_min, ox_max))
        cam.OffsetY.Value = int(self._align_to_inc(max(y_min, min(cam.OffsetY.Value, oy_max)), y_inc, y_min, oy_max))

        print(f"[ROI] Trimmed right {trim_right_px}px and bottom {trim_bottom_px}px -> "
              f"Width={cam.Width.Value}, Height={cam.Height.Value}, "
              f"OffsetX={cam.OffsetX.Value}, OffsetY={cam.OffsetY.Value}")

    # -------------------- END CODE --------------------



        # -------------------- END CODE --------------------

    ##````AUXILIARY SETTINGS1```##
    @staticmethod
    def get_temperature(which_subdevice: pylon.InstantCamera) -> float:
        """Gets the current temperature of the camera subdevice."""
        e = which_subdevice.TemperatureState.Value
        print(e)
        current_temperature = which_subdevice.DeviceTemperature.Value
        print(f'Current Temperature is: {current_temperature}°C')
        return current_temperature


    '''!!!!!!!!FRAME CORRECTION!!!!!!!!'
     -----------------------------'''
    #BACKGROUND
    '''Camera detectors have pixel-to-pixel variations in their sensitivity to light, and some noise is present from the detector, even when it records no light. 
    To account for this, most applications should incorporate a flat field, or “white reference” calibration.This calibration requires two measurements: 
    a dark frame recorded with no light incident on the camera sensor, and a reference frame recorded with a known reference level of light uniformly incident on the sensor. 
    A common way to apply this calibration is to record a dark frame with a lens cap in place, and to record a reference (or “white”) frame with a material of known reflectance 
    spanning the entirety of the imager’s field of view. After the two reference points have been collected, each subsequent frame of camera data is corrected as:
    
    
    C = C_ref * (R-D)      <=========> to chemistry equivalent f_i = f_st *  (Ai)
               -------                                                      -----     where f_i = response factor (molar volume, mass type of basis), Ai = signal peak, Ast = Standard
                (W-D)                                                        (Ast)                            and f_st = some response value of standard
    Sources : https://resonon.gitlab.io/programming-docs/content/overview.html#pika-l-l-gige-lf-xc2-and-uv and https://en.wikipedia.org/wiki/Response_factor
              https://bioresources.cnr.ncsu.edu/wp-content/uploads/2019/04/2009.1.273.pdf
    Where is the C= corrected frame, C_ref = (scalar) is the reflectance of the reference material (often assumed to be 1), R = is the raw camera frame to be corrected, 
    W = is the white reference frame, D = and is the dark reference frame.'''
   


    @singledispatchmethod
    @classmethod
    def get_camera_var_type(cls, arg, *, set_class: bool = False, verbose: bool = False):
        raise TypeError(f"Unsupported type: {type(arg)!r}")

    '''Should use dispatcher to make clones of some current class dynamic funcs as static/class. Broad scale architecture because some instances = real scan run time; but 
        sometimes the camera (pika xc2) is required outside of that context so a super broad generator/dispatcher is required'''
    @classmethod
    def _install_dispatchers(cls) -> None:
        """Register singledispatch handlers exactly once. Kept INSIDE the class."""
        if cls._dispatch_installed:
            return

        # --- define local impls that accept (c, arg, ...); 'c' will be the class ---
        def _impl_scanner(c, arg, *, set_class=False, verbose=False):
            if verbose: print("dispatch: HSI_Scanner")
            cam = getattr(arg, "camera", None)
            if set_class and cam is not None:
                c.default_camera = cam
            return cam

        def _impl_camera(c, arg: pylon.InstantCamera, *, set_class=False, verbose=False):
            if verbose: print("dispatch: InstantCamera")
            if set_class:
                c.default_camera = arg
            return arg

        def _impl_none(c, arg, *, set_class=False, verbose=False):
            if verbose: print("dispatch: None")
            if set_class:
                c.default_camera = None
            return None

        try:
            # --- perform registrations using explicit types (no forward-ref strings) ---
            cls.default_camera = cls.get_camera_var_type.register(cls)(_impl_scanner)            # for HSI_Scanner instances
            cls.get_camera_var_type.register(pylon.InstantCamera)(_impl_camera)
            cls.get_camera_var_type.register(type(None))(_impl_none)

            cls._dispatch_installed = True
            print(f"The installer for {cls.default_camera} is {cls._dispatch_installed}")
        except RuntimeError as e:
            print(f'Failed to use @singledispatchmethod')
            print(f'code at {DebugProgram.get_line_number}')   
                
    '''!!!!!!!!SENSOR CORRECTION!!!!!!!

    '''     
    
    # -------------------- frame grabbing / averaging --------------------
    @classmethod
    def get_white_response_frame(cls, n: int) -> np.ndarray:
        if n == None or n < 0:
            n = 30
        # Harder to get... either need to empirically measure reflectance value of say white sheet of office paper (assumed homogenous)
        # https://www.researchgate.net/figure/Spectral-fluorescence-values-of-the-print-solids-on-fluorescent-paper-substrate-Light_fig2_47337184
        #       --> Then measure lux input / lux output per the entire grid and average for reflectance value 
        #       --> Or pull values from online as reflectance coefficient?
        #     --> Is it possible to reflectance by comparing two cameras of same object? Probably still need wavelength graph 
        
        # -------------------- START CODE --------------------
        # Practical implementation: average several frames for stability.
        return cls._grab_and_average_frames(n, timeout_ms=5000)
        # -------------------- END CODE --------------------

    @classmethod
    def get_dark_frame(cls) -> np.ndarray:
        if n == None or n < 0:
            n = 30
        # Easy to get ... just need to take a snapshot with no lights

        # -------------------- START CODE --------------------
        # Same averaging, but ensure the lens is capped / lights off.
        return cls._grab_and_average_frames(n=30, timeout_ms=5000)
        # -------------------- END CODE --------------------

    @classmethod
    def calculate_correction_frame(
        cls, 
        live_frame: np.ndarray = 0, 
        white_response_frame: np.ndarray = 0, 
        dark_frame: np.ndarray = 0
    ) -> np.ndarray:
        # Assuming we should clone the data and perform on cloned data from pypylon single thread OR
        # Go back after the cube is created and run calculations at end... will assume on cloned data for now
        get_numpy_dtype = np.finfo()
        
        if live_frame.size != (white_response_frame.size or dark_frame.size):
            live_frame.size
        
        R, W, D = live_frame, white_response_frame, dark_frame 
        C_ref = 1  # Assuming a perfectly white background ==> sheet of paper 
        C = C_ref * (R - D) / (W - D)  # from equation in orange
        C = C * get_numpy_dtype  # scaling to 255 if 8-bit 
        
        return C

        # -------------------- START CODE --------------------
        # Kept your comments and structure; below is a numerically safe, dtype-aware version.
        # It returns float32 in [0,1] by default; scale to 8/16-bit as needed by casting.
        if not (isinstance(live_frame, np.ndarray) and isinstance(white_response_frame, np.ndarray) and isinstance(dark_frame, np.ndarray)):
            raise ValueError("All frames must be numpy arrays.")
        if live_frame.shape != white_response_frame.shape or live_frame.shape != dark_frame.shape:
            raise ValueError("Frame shapes must match for correction.")

        R = live_frame.astype(np.float32, copy=False)
        W = white_response_frame.astype(np.float32, copy=False)
        D = dark_frame.astype(np.float32, copy=False)

        eps = 1e-6
        denom = np.maximum(W - D, eps)
        C = (R - D) / denom  # C_ref assumed 1.0
        C = np.clip(C, 0.0, 1.0)
        return C
        # -------------------- END CODE --------------------

    # ============== helpers used above ==============
    @classmethod
    def _grab_and_average_frames(cls, num_frames: int = -1, timeout_ms: int = 5000) -> np.ndarray:
        assert num_frames != -1, "Must specify number of frames to grab"
        """
        Internal utility: grab and average n frames to float32 (H, W).
        Assumes mono stream; tweak PixelType if needed.
        """
        
        if not getattr(cls, "default_camera", None):
            raise RuntimeError(
                f"camera dispatcher didn't initialize, str(cls.default_camera) not initialized yet. Cannot grab dark/correction frames"
            )
        
        cur_cam = cls.default_camera
        assert cur_cam is not None
        
        cur_exposure_mu = cur_cam.ExposureTime.Value 

        cur_cam.StartGrabbingMax(num_frames)
        acc = None
        count = 0
        converter = pylon.ImageFormatConverter()  # NOT thread safe class
        
        # Pixel type lookup for bit depth and dtype
        pixel_name_by_value = {v: k for k, v in pylon.__dict__.items() if k.startswith("PixelType_")}
        converter.OutputPixelFormat = pylon.PixelType_Mono12  # can change depending on pipeline

        pixel_name = pixel_name_by_value.get(converter.OutputPixelFormat, "Unknown")
        match = re.search(r"(\d+)$", pixel_name)
        bit_depth = int(match.group(1)) if match else 8
        print(f"{pixel_name=}, {bit_depth=}")

        if bit_depth > 8:
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Map PixelType to numpy dtype
        PIXEL_TYPE_TO_DTYPE = {
            pylon.PixelType_Mono8: np.uint8,
            pylon.PixelType_Mono10: np.uint16,
            pylon.PixelType_Mono12: np.uint16,
            pylon.PixelType_Mono16: np.uint16,
        }
        output_dtype = PIXEL_TYPE_TO_DTYPE.get(converter.OutputPixelFormat, np.uint8)

        while cur_cam.IsGrabbing():
            res = cur_cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
            if res.GrabSucceeded():
                arr = converter.Convert(res).GetArray()
                if acc is None:
                    acc = arr.astype(np.float32)  # accumulate in float32
                else:
                    acc += arr.astype(np.float32)
                count += 1
            res.Release()

        if acc is None or count == 0:
            raise RuntimeError("No frames grabbed for averaging.")
        
        # Average and cast to camera dtype
        return (acc / count).astype(output_dtype)     


    @staticmethod
    def _calculate_wavelength_from_pixel(cube: np.ndarray) -> np.ndarray:
        #Probably should be calculating this live if possible per snapshot/2D sheet per datacube if not too slow; can use cloned data on separate thread to run
        # λ = a(x)^2 + bx + c 
        
        # where x is delta x direction for both sensor and for ASI console; λ = wavelength of light; (a,b,c) are calibration coefficients of resonon setup 
        #from spectronon app, the calibration looks like:  A = 1.37999 x 10^-05, B = 6.555092 x 10 ^ -1, C = 322.631 x 10 ^ 0
        # ------> seems to imply this may or may not be autoset on firmware... will have to do using this program and a scan using spectronon
        #         software and compare the graphs... if is already set on firmware setting then we should get (nearly) same values
        np.array
        A: float;  B: float; C: float = ((1.379999) * (10**-5)),  ((6.555092) * (10**-1)), ((322.6319) * (10**0)) 
        wavelength = A*(x*x) + B*(x) + C
        
        return cube

        # -------------------- START CODE --------------------
        # Keeping your prose above; below is a concrete mapper that returns a 1D wavelength vector
        # for the spectral dimension (assumes spectral = first axis of a 2D frame (H,W) or a cube (Bands,Rows,Cols)).
        # If 'cube' is (H, W), H is spectral length; if (Bands, R, C), Bands is spectral length.
        if cube is None or not isinstance(cube, np.ndarray):
            raise ValueError("cube must be a numpy array.")
        spectral_len = cube.shape[0]
        idx = np.arange(spectral_len, dtype=np.float32)
        # reflection True per your observation: higher pixel index ⇒ shorter λ
        wl = lambda_from_pixel(idx, total_pixels=spectral_len, reflected=True)
        return wl
        # -------------------- END CODE --------------------
    
        #'''''''''''''~~~~~~~~~~~CAMERA SETTINGS~~~~~~~~~~~~~''''''''''''
        
        #________________________________________________________________
        
        
    def handle_keyboard_controls(self):
        """Respond to arrow keys for exposure/gain tweaks."""
        
        #check_for_esc()
        #def check_for_esc(self):
        if keyboard.is_pressed("esc"):
                print("CLOSING PROGRAM")
                time.sleep(2)
                self.Stage.set_max_speed('X', 1.0)
                self.Stage.set_max_speed('Y', 1.0)
                self.close_secondary_camera()
                self.move_camera_home(custom_home_spot= False)
                time.sleep(1)
                self.close()
                time.sleep(1)
                raise KeyboardInterrupt("User requested exit via ESC key")
        
        if keyboard.is_pressed("shift+left"):
            self.increase_exposure(self.camera.ExposureTime.Value)
        elif keyboard.is_pressed("shift+right"):
            self.decrease_exposure()
            
        elif keyboard.is_pressed("shift+up"):
            self.increase_gain(self.camera.Gain.Value)
        elif keyboard.is_pressed("shift+down"):
            self.decrease_gain()
        #print("handling_keyboard")  
        
    def set_exposure_from_gui(self, initial_exposure_set_ms: float) -> float:  # set by user
        """
        https://docs.baslerweb.com/exposure-time

        Set absolute exposure from GUI input (ms → µs conversion).
        min: 20/21 (8 bit / 12 bit pixel format respectively)
        max: 10,000,000 µs = 10 s
        """
        # Convert ms → µs first, then clamp to camera/driver limits
        exposure_us_requested = float(initial_exposure_set_ms) * 1000.0
        exposure_us = HSI_Scanner.Utils.safe_num(
            exposure_us_requested,
            min_val=self.exp_min,   # assumed in µs
            max_val=self.exp_max    # assumed in µs
        )
        exposure_us = round(exposure_us, 2)
        self.camera.ExposureTime.SetValue(exposure_us)

        print(f"[GUI] Exposure set to: {exposure_us:.0f} µs (requested {initial_exposure_set_ms:g} ms)")
        print(f"[Exposure set via pypylon]: {self.camera.ExposureTime.Value} µs")
        return exposure_us


    def set_gain_from_gui(self, initial_gain: float) -> float:  # set by user
        """
        Gain in dB. Basler 'Gain' node is typically floating-point dB.
        """
        gain_db = HSI_Scanner.Utils.safe_num(
            float(initial_gain),
            min_val=self.gain_min,   # dB
            max_val=self.gain_max    # dB
        )
        gain_db = round(gain_db, 2)
        self.camera.Gain.SetValue(gain_db)

        print(f"[GUI] Gain set to: {gain_db:.2f} dB")
        print(f"[Gain set via pypylon]: {self.camera.Gain.Value} dB")
        return gain_db

        
        
    def increase_exposure(self, step: int = 100):
        """
        Increase exposure by a small step in microseconds (used with keyboard).
        
        """
        #__custom_exposure_max = 490000 # µs (16.33 ms)
        #__custom_exposure_max = 16330 # µs (16.33 ms)
        __custom_exposure_max = self.camera.ExposureTime.Value
        current_exposure = self.camera.ExposureTime.Value
        new_exposure = min(current_exposure + step, self.exp_max, __custom_exposure_max)
        self.camera.ExposureTime.SetValue(new_exposure)
        self.exposure = new_exposure
        print(f"[Hotkey] Exposure set to: {self.exposure:.0f} µs")
        

        
    '''def increase_exposure(self, current_exposure: float = None, event=None):
        #ucrrent_exposurecan only be 12,800 going into func
            
            if current_exposure == None:
                current_exposure = self.camera.ExposureTime.Value
        
            __custom_exposure_max = 12800
            #converted_exposure: int = current_exposure * 100
            converted_exposure: int = current_exposure * 1
            
            if converted_exposure > __custom_exposure_max:
                self.camera.ExposureTime.SetValue(__custom_exposure_max)
            #Default limit how high the exposure can be changed my memory max should be ~128000 ms
            #Found that Exposure time = 1633, Gain = 9 gives reasonable results visually
            
            self.exposure = min(self.camera.ExposureTime.Value + 100, self.exp_max)
            self.camera.ExposureTime.SetValue(self.exposure)
            print(f"Exposure set to: {self.exposure:.0f} µs")
    '''

    def decrease_exposure(self, event=None):
        self.exposure = max(self.camera.ExposureTime.Value - 100, self.exp_min)
        self.camera.ExposureTime.SetValue(self.exposure)
        print(f"Exposure set to: {self.exposure:.0f} µs")
        
    
    def increase_gain(self, current_gain: float, event=None):
        if (current_gain >= 23) or (self.camera.Gain.GetValue() >= 15):
            self.camera.Gain.SetValue(23)
            print("Already at maximum custom_gain setting")
            return None
        new_gain = min(current_gain + 1, self.gain_max)
        self.camera.Gain.SetValue(new_gain)
        self.gain = new_gain  # Store it if you're tracking it elsewhere
        print(f"Gain set to: {self.gain:.2f}")
        return self.gain


    def decrease_gain(self, event=None):
        current_gain = self.camera.Gain.GetValue()
        new_gain = max(current_gain - 1, self.gain_min)
        self.camera.Gain.SetValue(new_gain)
        self.gain = new_gain
        print(f"Gain set to: {self.gain:.2f}")
        return self.gain




class Make_GUI:
    # class-level state for an external "data meta" file (set via set_meta_file)
    meta_file: str = ""
    cur_dir: str = ""   #class level variable that saves based on instance variable found init --> should be update to date at this point hence cur_dir
    

    def __init__(self):
        # preload the save/GUI structure before the actual scan execution
        cur_file, save_dir = self._format_meta_data_and_save_structure()
        Make_GUI.cur_dir = save_dir or ""   # keep a canonical, class-level current dir
        print(f"our Make_GUI.cur_dir var is {Make_GUI.cur_dir}")
        self.params: Dict[str, Any] = {}
                # Leave empty and it’ll be used only as a last resort.
        self.GLOBAL_FIELD_MAP: Dict[str, str] = {
            # "x_dist": "x_distance",
            # "y_dist": "y_distance",
            # ...
    }

    # ── public-ish helpers ──────────────────────────────────────────────

    @classmethod
    def set_meta_file(cls, *, directory: Union[str, Path], filename: Union[str, Path]) -> None:
        """
        Set the class-level "data meta" file path used by _write_to_data_metafile().
        Accepts Path or str for both args. Spaces are fine.
        """
        d = Path(directory)
        f = Path(filename)
        cls.meta_file = str(d / f)

    def _write_to_metafile(self, **kwargs) -> None:
        """
        Update the primary GUI meta JSON file (self.meta_file_path) with key/value pairs.
        Requires that _format_meta_data_and_save_structure has already run.
        """
        if not getattr(self, "meta_file_path", None):
            raise RuntimeError("meta_file_path not initialized yet.")
        payload = self._json_load_or_empty(self.meta_file_path)
        payload.update(kwargs)
        with open(self.meta_file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    @classmethod
    def _write_to_data_metafile(cls, **kwargs) -> None:
        """
        Append a JSON object (pretty-printed) to the class-level data meta file.
        Call set_meta_file(...) first.
        """
        if not cls.meta_file:
            raise ValueError("Meta file path not set. Call set_meta_file() first.")
        with open(cls.meta_file, "a", encoding="utf-8") as f:
            json.dump(kwargs, f, ensure_ascii=False, indent=2)
            f.write("\n")

    # ── core setup ──────────────────────────────────────────────────────

    def _format_meta_data_and_save_structure(self):
        """
        Decide where the GUI's meta JSON lives (under DEFAULT_FOLDER, named after this .py),
        ensure the folder exists, read previous settings (if any), then write a normalized
        JSON file containing at least {"previous_user_folder": "..."}.
        """
        print("[INIT] Starting _format_meta_data_and_save_structure")

        self.default_folder: Path = DEFAULT_FOLDER
        self.previous_user_folder: Optional[str] = None  # marker

        # derive meta file name from current script name
        cur_file_name = __file__
        self.meta_file_type = ".json"

        trimmed_file = os.path.basename(cur_file_name)
        m = re.fullmatch(r"(.*)\.py", trimmed_file, flags=re.IGNORECASE)
        if m:
            cur_file_name = m.group(1)
        else:
            print("[WARN] Could not parse script base name; using raw __file__")

        # Ensure folder exists
        self.default_folder.mkdir(parents=True, exist_ok=True)

        # meta file path
        txt_meta_file: Path = self.default_folder / (cur_file_name + self.meta_file_type)
        print(f"[FORMAT] txt_meta_file path: {txt_meta_file}")

        # Read any existing state (legacy TXT or JSON) to populate self.previous_user_folder
        self._change_meta_data_and_save_structure(params=None, user_txt_meta_file=txt_meta_file)

        # Normalize payload and (over)write as JSON
        payload = {"previous_user_folder": self.previous_user_folder or str(self.default_folder)}
        with open(txt_meta_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[FORMAT] Wrote GUI meta JSON: {payload!r}")

        self.meta_file_path: Path = txt_meta_file
        assert self.meta_file_path, "meta_file_path must not be empty"
        print("[INIT] Finished _format_meta_data_and_save_structure")
        return self.meta_file_path, self.previous_user_folder

    def _change_meta_data_and_save_structure(
        self,
        params: Optional[Dict[str, Any]] = None,
        user_txt_meta_file: Optional[os.PathLike] = None
    ):
        """
        If params is provided (and includes 'save_folder'), persist it to meta.
        Otherwise, try to read existing meta to populate self.previous_user_folder.
        Handles legacy plain-text meta and new JSON meta.
        """
        print(f"[CHANGE] Called with params: {params}, user_txt_meta_file: {user_txt_meta_file}")
        if user_txt_meta_file is None:
            print("[CHANGE] No meta file path provided, aborting save/read")
            return

        meta_path = Path(user_txt_meta_file)

        if params is not None:
            # SAVE branch
            save_folder = params.get("save_folder") if isinstance(params, dict) else None
            if save_folder:
                self.previous_user_folder = os.path.abspath(str(save_folder))
            else:
                self.previous_user_folder = str(self.default_folder)

            payload = {"previous_user_folder": self.previous_user_folder}
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"[CHANGE] Saved JSON meta: {payload!r}")
            return

        # READ branch
        if not meta_path.exists():
            print("[CHANGE] Meta file not found; using default folder")
            self.previous_user_folder = str(self.default_folder)
            return

        # Try JSON first, then legacy text as fallback
        payload = self._json_load_or_none(meta_path)
        if isinstance(payload, dict) and "previous_user_folder" in payload:
            user_dir = str(payload.get("previous_user_folder") or "").strip()
            if user_dir:
                self.previous_user_folder = os.path.abspath(user_dir)
                print(f"[CHANGE] Loaded JSON meta previous_user_folder={self.previous_user_folder}")
                return

        # Fallback: legacy first-line text
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                first = f.readline().strip()
            if first:
                self.previous_user_folder = os.path.abspath(first)
                print(f"[CHANGE] Loaded legacy meta previous_user_folder={self.previous_user_folder}")
            else:
                print("[CHANGE] Legacy meta empty; using default folder")
                self.previous_user_folder = str(self.default_folder)
        except Exception as e:
            print(f"[CHANGE] Error reading legacy meta: {e}")
            self.previous_user_folder = str(self.default_folder)

    # ── JSON utils ──────────────────────────────────────────────────────

    @staticmethod
    def _json_load_or_none(p: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Return dict if valid JSON dict, otherwise None."""
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    @staticmethod
    def _json_load_or_empty(p: Union[str, Path]) -> Dict[str, Any]:
        """Return dict if valid JSON dict, otherwise {}."""
        d = Make_GUI._json_load_or_none(p)
        return d if isinstance(d, dict) else {}

    
    # ── Real life scan container type utils ──────────────────────────────────────────────────────
    @staticmethod
    def _plate_measurements(which_well: str):
        """
        Return plate parameters (x_start_mm, y_start_mm, z_start_mm, dx_mm, dy_mm, rows, cols)
        for a given well type. The order should be (# of rows, # columns) like matrix notation
        """
        # Example: 12-well 3x4 plate
        if which_well == "3_by_4_wells":
            return {
                "x_start_mm": 0.0,
                "y_start_mm": 0.0,
                "z_start_mm": 0.0,
                "dx_mm": 25.0,
                "dy_mm": 25.0,
                "rows": 3,
                "cols": 4
            }
            
        if which_well == "8_by_12_wells": #make this in a new file for every corin type well this one should be corin 3603 
            #recommended to do 1x1 or 2x2 because the diameter is roughly 2 * pi for the well (area = 31 mm^2)
            return {
                "x_start_mm": 0.0,
                "y_start_mm": 0.0,
                "z_start_mm": 0.0,
                "dx_mm": 9,
                "dy_mm": 9,
                "rows": 3, #8
                "cols": 7  #12
            }
        
        if which_well == "4_by_6_wells": # data found here: https://www.tpp.ch/page/produkte/09_zellkultur_testplatte.php used TPP plate 92424
            #recommended to do 4x4 - 6x6 for fluorescence to be safe or 2x2 brightfield

            return {
                "x_start_mm": 0.0,
                "y_start_mm": 0.0,
                "z_start_mm": 0.0,
                "dx_mm": 19,
                "dy_mm": 19,
                "rows": 4, #8
                "cols": 6  #12
            }
            
        else:
            raise ValueError(f"Unknown well type: {which_well}")

    @staticmethod
    def generate_well_snake(
        which_well: str,
        to_asi: Optional[Callable] = None,
        both_asi_and_mm: bool = False
    ) -> List[Tuple[float, float, float]]:
        """
        Generate 12-well (3×4) plate home points in snake (S-pattern) order.
        Returns: list of (x, y, z) in mm or ASI units (or both if flag is set)
        """
        # Get plate measurements
        params = Make_GUI._plate_measurements(which_well)
        x_start_mm = params["x_start_mm"]
        y_start_mm = params["y_start_mm"]
        z_start_mm = params["z_start_mm"]
        dx_mm = params["dx_mm"]
        dy_mm = params["dy_mm"]
        rows = params["rows"]
        cols = params["cols"]

        homes = []
        mm_homes = []

        for row in range(rows):
            y = y_start_mm + row * dy_mm
            col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)

            for col in col_range:
                x = x_start_mm + col * dx_mm
                z = z_start_mm

                if to_asi:
                    x_asi = to_asi(x)
                    y_asi = to_asi(y)
                    z_asi = to_asi(z)
                    homes.append((x_asi, y_asi, z_asi))
                    if both_asi_and_mm:
                        mm_homes.append((x, y, z))
                else:
                    homes.append((x, y, z))

        # Print the coordinates
        if both_asi_and_mm and mm_homes:
            print("=== ASI + MM Coordinates ===")
            for index, (asi_coord, mm_coord) in enumerate(zip(homes, mm_homes)):
                print(f"{index}: ASI: {asi_coord}\t| mm: {mm_coord}")
        else:
            label = "ASI Units (1/10th micron)" if to_asi else "Metric Units (mm)"
            print(f"=== {label} ===")
            for index, item in enumerate(homes):
                print(f"{index}: {item}")

        return homes

        #scan_type: TypedDict[x_dist: str, y_dist: str, stage_speed: str] = {}
        #SPD.needs_factory(Make_Dict)   
        # Optional: your existing central map, if you have it.


    def _normalize(self, s: str) -> Optional[str] | Optional[Any]:
        if not isinstance(s, str):
            return s
        try:
            return re.sub(r"[^a-z0-9]+", "", s.lower())
        except RuntimeError as e:
            print(f"Problem with current string normalization: {s}")
            print(f"Function: {self._normalize.__name__}")
            print(f"~Line # {DebugTimer.get_line_number()}")

    def _build_binding_map(
        self,
        window,
        defaults: Dict[str, object],
        global_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Build a mapping from schema/defaults keys -> GUI element keys dynamically.
        Priority:
        1) Element metadata: if window[gui_key].metadata == <schema_key> or dict with {"schema_key": ...}
        2) Exact key name match: schema_key == gui_key
        3) Fuzzy match on normalized names
        4) Fallback to provided global_map (if any)
        Returns: {schema_key: gui_key}
        """
        if global_map is None:
            global_map = {}

        try:
            gui_keys = list(window.AllKeysDict.keys())
        except Exception:
            gui_keys = []

        # Pre-index GUI keys by normalized form for fuzzy match
        norm_to_gui = {}
        for gk in gui_keys:
            norm_to_gui.setdefault(self._normalize(gk), []).append(gk)

        binding: Dict[str, str] = {}

        # Pass 1: metadata hints (most explicit)
        for gk in gui_keys:
            elem = window[gk]
            meta = getattr(elem, "metadata", None)
            if isinstance(meta, dict) and "schema_key" in meta:
                schema_key = meta["schema_key"]
                if schema_key in defaults:
                    binding[schema_key] = gk
            elif isinstance(meta, str):
                # allow metadata to be the schema key directly
                if meta in defaults:
                    binding[meta] = gk

        # Pass 2: exact key-name matches
        for sk in defaults.keys():
            if sk in binding:
                continue
            if sk in gui_keys:
                binding[sk] = sk

        # Pass 3: fuzzy (normalized) matches
        for sk in defaults.keys():
            if sk in binding:
                continue
            candidates = norm_to_gui.get(self._normalize(sk), [])
            if candidates:
                # Choose the first; or add heuristics if you want
                binding[sk] = candidates[0]

        # Pass 4: fallback to your central/global map
        for sk, gk in (global_map or {}).items():
            if sk in defaults and sk not in binding and gk in gui_keys:
                binding[sk] = gk

        return binding

    def _apply_defaults_using_map(
        self,
        window,
        defaults: Dict[str, object],
        binding: Dict[str, str]
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Apply defaults to GUI using a precomputed binding map {schema_key -> gui_key}.
        Returns (updated_gui_keys, skipped_with_reason).
        """
        updated: List[str] = []
        skipped: List[Tuple[str, str]] = []

        # Update only for keys present in both
        for sk, gk in binding.items():
            if sk not in defaults:
                skipped.append((gk, f"source key '{sk}' missing"))
                continue
            try:
                window[gk].update(str(defaults[sk]))
                updated.append(gk)
            except Exception as e:
                skipped.append((gk, f"update error: {e!r}"))

        # Optionally, report schema keys that had no GUI binding at all
        for sk in defaults.keys():
            if sk not in binding:
                skipped.append((sk, "no GUI binding"))

        if skipped:
            print("[apply_defaults] skipped:", skipped)
        return updated, skipped

    def _load_defaults(self, schema: str) -> Dict[str, object]:
        """Always fetch a fresh dict from SMF per click."""
        try:
            return SMF.MakeParams.create__dict(schema=schema) or {}
        except Exception as e:
            print(f"[load_defaults] schema='{schema}' error: {e!r}")
            return {}
    
    def show_advanced_gui(self) ->  np.ndarray | None:
        _frame_corrections = {}
        _default_frames = 30
        layout = [
            [sg.Text("Correction cube type")],
            [sg.Checkbox("Pick correction cubes to use:", "None", key="dev_logs")],

            # --- Radio + Frame for Dark Cube ---
            [sg.Radio("Dark Correction Cube", "CUBE_TYPE", key="radio_dark", default=True),
            sg.Frame(
                title = "",
                layout = [
                    [sg.Text("Number of frames to average", size=(20, 1)),
                    sg.InputText(str(_default_frames), key="d_key_num", size=(8, 1))],
                    [sg.Button("Set correction", key="d_key")]
                ],
                relief="sunken",
                border_width=2
            )],

            # --- Radio + Frame for Response Cube ---
            [sg.Radio("Response Correction Cube", "CUBE_TYPE", key="radio_resp"),
            sg.Frame(
                title = "",
                layout = [
                    [sg.Text("Number of frames to average", size=(20, 1)),
                    sg.InputText(str(_default_frames), key="r_key_num", size=(8, 1))],
                    [sg.Button("Set correction", key="r_key")]
                ],
                relief= "sunken",
                border_width=2
            )],

            [sg.Button("Apply Correction(s)"), sg.Button("Close")]
        ]

        win = sg.Window("Advanced Settings", layout, modal=True, keep_on_top=True)

        while True:

            e, v = win.read()
            if e in (sg.WIN_CLOSED, "Close"):
                break

            elif e == "d_key":
                print(f"[Dark cube] Averaging {v['d_key_num']} frames")
                dark_cube = HSI_Scanner.get_dark_frame(n = v['d_key_num'])
                _frame_corrections["Dark cube"] = dark_cube
            elif e == "r_key":
                print(f"[Response cube] Averaging {v['r_key_num']} frames")
                resp_cube = HSI_Scanner.get_white_response_frame(n = v['r_key_num'])

            elif e == "Start Scan":
                if v["radio_dark"]:
                    print(f"Starting scan with Dark cube | {v['d_key_num']} frames")
                elif v["radio_resp"]:
                    print(f"Starting scan with Response cube | {v['r_key_num']} frames")
                else:
                    print("No cube type selected.")
                break
            
            
        win.close()
    
    def show_gui(self):
        print(f"[GUI] Showing GUI with default_folder: {self.default_folder}, previous_user_folder: {self.previous_user_folder}")
        
        main_layout = [
            [sg.Text(text = "HSI Scan Configuration", relief = "groove", font=("Source Code Pro", 18, "bold"), justification = 'center', background_color = "#33A4AC")],
            [sg.Text("Run with Debug?", tooltip="DONT USE UNLESS BEN"),
            sg.Radio("Yes", "DEBUG", key="debug_yes", default=False, metadata=bool),
            sg.Radio("No", "DEBUG", key="debug_no", default=True, metadata=bool)],
            
            [sg.Text("Wells/Dish or Plate sample?"),
            sg.Radio("Plate", "MODE", key="plate_sample", default=True, metadata=bool),
            sg.Radio("Wells/Dish", "MODE", key="wells_sample", default=False, metadata=bool)],

            [sg.HorizontalSeparator(color="#04D6D6", key="sep_top1", pad=10)], 
            [sg.HorizontalSeparator(color="#00CCCC", key="sep_top2", pad=10)],

            [
                                        # making all below sg.elements have proper alignment by fixing size of element
            sg.Frame("MS 2000 stage-specific parameters -- adjust for scan type", font = ("Input", 16, "italic"),
                        #for font param; 'name size styles' works
                    layout=[
                        [sg.Text("Scan mode type",
                                tooltip="Both are technically not mutually exclusive; both option can be added later on"),
                        sg.Radio("Brightfield", key="bright", default=True, enable_events=True,
                                group_id="scan_mode_type", metadata=bool),
                        sg.Radio("Fluorescence", key="fluoro", default=False, enable_events=True,
                                group_id="scan_mode_type", metadata=bool)],

                        [sg.Text("Rows:"), sg.InputText("5", key="rows", pad = ((180,0), (0,0)), metadata=int, 
                                                        #pad = ((left,right), (top, bottom))
                                                        tooltip="DONT USE MORE THAN 5 IF DOING WELLS/DISH")],
                        [sg.Text("Columns:"), sg.InputText("5", key="cols", metadata=int,
                                                        tooltip="DONT USE MORE THAN 5 IF DOING WELLS/DISH")],

                        [sg.Text("X distance (mm):"),
                        sg.InputText("1.02", key="x_distance", enable_events=True, metadata=float)],
                        [sg.Text("Y distance (mm):"),
                        sg.InputText("0.938", key="y_distance", enable_events=True, metadata=float)],
                        [sg.Text("Stage speed (mm/s):"),
                        sg.InputText("0.0623", key="stage_speed", enable_events=True, metadata=float)],

                        [sg.Text("Set ASI MS-2000 temporary movement params",
                                tooltip="Scan may break due to too many lines if not custom option"),
                        sg.Radio("Use our Custom Ones", "ASI_MODE", key="custom_asi",
                                default=True, metadata=bool),
                        sg.Radio("Use Factory Default", "ASI_MODE", key="default_asi",
                                default=False, metadata=bool)],

                        [sg.Checkbox("Get ASI_INFO after setting params",
                                    key="check_asi_info", default=False)],
                    ],
                    title_location=sg.TITLE_LOCATION_TOP,
                    relief= "sunken",
                    border_width=3,
                    tooltip="Values should autoupdate upon selection of fluoro/brightfield")
            ],
            [sg.HorizontalSeparator(color="#00CCCC", key="sep_mid1", pad=10)], 
            [sg.HorizontalSeparator(color="#04D6D6", key="sep_mid2", pad=10)], 
                [
                sg.Frame("Camera specific parameters [camera = acA1920-155um] ", font=("Input", 16, "italic"),
                        layout = [
                    # ---- Binning and imaging options ----
                    [sg.Text("Spectral bin:"), sg.InputText("6", key="spectral_bin", metadata=int)],
                    [sg.Text("Spatial bin:"), sg.InputText("2", key="spatial_bin", metadata=int)],
                    [sg.Text("Line bin:"), sg.InputText("1", key="line_bin", metadata=int)],

                    [sg.Text("Exposure Time (ms):"), sg.InputText("16.33", key="exposure_time", metadata=float)],
                    [sg.Text("Gain (dB):"), sg.InputText("20", key="gain", metadata=float)],
                    [sg.Button("Set Correction Cubes")],

                    

                    [sg.Button("Start Scan"), sg.Button("Cancel")]
                    ],
                    title_location= sg.TITLE_LOCATION_TOP,
                    border_width=4,
                    tooltip="Values should autoupdate upon selection of fluoro/brightfield")
                ],
            [sg.HorizontalSeparator(color="#00CCCC", key="sep_mid1", pad=10)], 
            [sg.HorizontalSeparator(color="#04D6D6", key="sep_mid2", pad=10)], 
                [
                sg.Frame("Other settings", font=("Input", 16, "italic"),
                        layout = [
                    [sg.Text("Save folder:"),
                        sg.InputText(str(self.previous_user_folder), key="save_folder", metadata=str,
                        tooltip=f'default_folder_path is {self.default_folder}'),
                    sg.FolderBrowse(initial_folder=str(self.previous_user_folder or self.default_folder))],
                    
                    [sg.Text("Brightfield LED power value")],
                    [sg.Slider((0,99), default_value = 99, orientation = "horizontal", resolution = 1, enable_events = True, 
                               key = "LED_power", metadata = int)],
                        #default value 99 to match brightfield here
                                ]
                        )
                ],
        ]

        window = sg.Window("HSI Scanner", main_layout, use_default_focus=False, keep_on_top=True)
        window.read(timeout=0)  # “prime” the window

        current_defaults = self._load_defaults("bright")
        binding_map = self._build_binding_map(window, current_defaults, self.GLOBAL_FIELD_MAP)
        self._apply_defaults_using_map(window, current_defaults, binding_map)
            
        temp_dictionary = {
        "fluoro": "fluoro",
        "bright": "bright",
        "debug_yes": "debug_yes",
        "plate_sample": "plate_sample",
        "well_sample": "well_sample",
        }

        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, "Cancel"):
                window.close()
                print("Closing window...")
                sys.exit(0)

            if event == "Set Correction Cubes":
                self.show_advanced_gui()
                continue  # nothing else to do for this event

            # Map only schema-selection buttons; ignore others like "Start Scan"
            schema = temp_dictionary.get(event)
            if schema:
                current_defaults = self._load_defaults(schema)
                binding_map = self._build_binding_map(window, current_defaults, self.GLOBAL_FIELD_MAP)
                self._apply_defaults_using_map(window, current_defaults, binding_map)
                print(f"[GUI] Changing parameters to {schema.upper()} using {current_defaults}")
                continue

    # ...handle other events like "Start Scan" here...

            if event == "Start Scan":
                # ... your existing Start Scan logic ...
                print(f"[GUI] Start Scan pressed with values: {values}")
                try:
                    params = {}
                    for key, val in values.items():
                        elem = window[key]
                        caster = getattr(elem, "metadata", None) or str
                        try:
                            params[key] = caster(val)
                        except Exception:
                            sg.popup_error(f"Invalid value for {key}: {val}")
                            raise

                    print(f"[GUI] Params after casting: {params}")
                    
                    '''

                    def _fallback(v, default):
                        return default if (v is None or (isinstance(v, str) and not v.strip())) else v

                    params["x_distance"] = float(_fallback(values["x_distance"], current_defaults["x_dist"]))
                    params["y_distance"] = float(_fallback(values["y_distance"], current_defaults["y_dist"]))
                    params["stage_speed"] = float(_fallback(values["stage_speed"], current_defaults["stage_speed"]))
                    params["exposure_time"] = float(_fallback(values["exposure_time"], current_defaults["exposure_time"]))
                    params["gain"] = float(_fallback(values["gain"], current_defaults["gain"]))
                    '''

                    self._change_meta_data_and_save_structure(params=params, user_txt_meta_file=self.meta_file_path)

                    if params.get("custom_asi") is True:
                        MS2000.get_and_set_backlash()
                        MS2000.get_and_set_acceleration()
                        if params.get("check_asi_info"):
                            MS2000.get_ASI_stagetop_info()

                    self.params = params
                    window.close()
                    return
                except Exception as e:
                    sg.popup_error(f"Exception in GUI: {e}")
                    continue

    '''
    def _pre_scan_check(self, scanner = None):
        assert scanner is not None
        cam = HSI_Scanner.default_camera
        pylon.GrabStrategy_LatestImageOnly
    '''    
        
        
    def run_scan(self):

        if not self.params:
            print("[RUN] No params set. Run show_gui() first.")
            return

        params = self.params

        
        # Scanner debug logic
        global Scanner
        Scanner.debug_frames = params["debug_yes"]

        # \\\\\ Scanner run logic /////
        exposure_time = HSI_Scanner.Utils.safe_num(params["exposure_time"])
        gain = HSI_Scanner.Utils.safe_num(params["gain"])
        Scanner.set_exposure_from_gui(exposure_time)
        Scanner.set_gain_from_gui(gain)
        
        use_wells = bool(params.get("wells_sample", True))
        if use_wells:
            # Pass the function, do NOT call it here
            generated_home_points = Make_GUI.generate_well_snake(
                which_well = "4_by_6_wells",
                to_asi = HSI_Scanner.Utils.metric_to_asi, 
                both_asi_and_mm=True
            )
            custom_home_spot = True
        else:
            # Plate mode: no custom homes, start at (0,0,0)
            generated_home_points = [(0, 0, 0)]
            custom_home_spot = False

        user_bin_factors = (
            params["spectral_bin"],
            params["spatial_bin"],
            params["line_bin"]
        )
        
        Scanner.pre_scan_check(
            #ordering and repetition matters here; e.g. had failrue because <params["save_folder"],> was in signature line twice
            params["rows"],
            params["cols"],
            params["x_distance"],
            params["y_distance"],
            params["stage_speed"],
            params["save_folder"],
            params["LED_power"],
            bin_factors = user_bin_factors,
            custom_home_spot = custom_home_spot,
            home_points = generated_home_points
        )
        sg.popup("Scan complete!")

        
########-------DEBUGGING CLASS(ES)! ####------------######
   # ######################################
   # ######################################\
   ##     #   \ | /    _     _   \ | /  #
     #   #  -- o o --/ o\~~~/o\-- o o --#\
   #     #    ( v ) (    . .    ) ( v ) #    \
     #   #   /_____\ \__\_v_/__/ /_____\# # |
    ################################
   # ######################################
   # ######################################
   

class DebugProgram(ABC):
    """Abstract foundation for debug utilities.

    Add shared helpers (like get_line_number), but do not force every subclass
    to implement I/O or timing they don't support.
    """
    #removed current abstract implementations may readd again.
    @staticmethod
    @abstractmethod
    def place_holder_method():
        pass
    
    @staticmethod
    def get_line_number() -> int:
        frame = inspect.currentframe()
        try:
            if frame and frame.f_back:
                return frame.f_back.f_lineno
            return -1
        finally:
            del frame
# Base filter interface

class CustomWarningsFilter(ABC):
    
    @staticmethod
    def place_holder_method():
        pass

    @classmethod
    @abstractmethod
    def custom_warning(cls, *errors, o: object):
        """Each warning subclass must implement this"""
        pass


# Concrete DebugExceptions container
class DebugExceptions:

    class MyDeletionWarning(Warning, CustomWarningsFilter):
        """Non-fatal cleanup warning; no item to delete."""

        @classmethod
        def custom_warning(cls, message: str = None, o: object = None):
            if message:
                warnings.warn(message, cls)

    class MyAttributeError(Warning, CustomWarningsFilter):
        """Critical error; Warning of missing attribute"""

        def __init__(self, message: str = ""):
            doc = self.__class__.__doc__ or ""
            super().__init__(f"{doc} --> {message}")

        @classmethod
        def custom_warning(cls, *attributes_missing, o: object = None):
            assert attributes_missing is not None
            warnings.filterwarnings("always", category=cls)
            obj_name = type(o).__name__ if o else "<unknown>"
            for attr in attributes_missing:
                warnings.warn(
                    f"Within object {obj_name}; Missing attribute: '{attr}'",
                    cls
                )

    @staticmethod
    def delete_global_names(*names: str) -> None:
        g = globals()

        for name in names:
            if name in g:
                del g[name]
            else:
                warnings.warn(
                    f"delete_global_names: '{name}' not defined; nothing to delete",
                    DebugExceptions.MyDeletionWarning,
                )
        gc.collect()
    
    @staticmethod
    def delete_local_names(*names: str)-> None:
        l = locals()

        for name in names:
            if name in l:
                del l[name]
            else:
                warnings.warn(
                    f"delete_global_names: '{name}' not defined; nothing to delete",
                    DebugExceptions.MyDeletionWarning,
                )
        gc.collect()
        
class DebugTimer(DebugProgram):
    '''Useful notes to self:
    threading.Thread exceptions require threading.excepthook (Python 3.8+)
        sys.excepthook only covers the main thread'''
    default_label = "unnamed_timer"
    _timers = {}
    _lock = threading.Lock()
    _timed_events = {}
    
    @staticmethod
    def place_holder_method():
        pass

    @staticmethod
    def time_event(label: Optional[str] = None):
        """
        Toggle timing for a given label. Starts if not running, stops if running.
        """
        if not label:
            label = DebugTimer.default_label

        line = DebugTimer.get_line_number()

        with DebugTimer._lock:
            # Stop existing timer
            if label in DebugTimer._timers:
                start_time, start_line = DebugTimer._timers.pop(label)
                elapsed = time.perf_counter() - start_time

                if label not in DebugTimer._timed_events:
                    DebugTimer._timed_events[label] = []

                DebugTimer._timed_events[label].append({
                    "elapsed": elapsed,
                    "start_line": start_line,
                    "stop_line": line,
                    "timestamp": time.time()
                })

                print(f"🕒[{label}] elapsed {elapsed:.6f}s (stopped at line {line})")
                return elapsed

            # Otherwise start a new timer
            DebugTimer._timers[label] = (time.perf_counter(), line)
            print(f"🕒[{label}] timer started at code line: {line}")
            return None

    # === Aliased methods that call time_event() ===

    @staticmethod
    def start(label: Optional[str] = None):
        """Alias for .time_event(), starts or toggles a timer."""
        return DebugTimer.time_event(label)

    @staticmethod
    def stop(label: Optional[str] = None):
        """Alias for .time_event(), stops or toggles a timer."""
        return DebugTimer.time_event(label)

    @staticmethod
    def summarize_events():
        """Print a summary of all timed events."""
        print("\n~~===~ Time Events Summary ~~===~~")
        for label, events in DebugTimer._timed_events.items():
            for i, e in enumerate(events, start=1):
                print(f"[{label}] #{i}: {e['elapsed']:.6f}s (lines {e['start_line']}→{e['stop_line']})")
        print("~~=============================~~\n")

    @staticmethod
    def make_local_debug_txt_file_json(override = OVERRIDE_DEBUG_FILE) -> bool:
        """Create a simple JSON file in the same directory as this script."""
        try:
            # Determine the local directory where the script resides
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, "debug.json")

            if os.path.exists(file_path) and not override:
                print(f"File already exists {file_path} (use override = True to overwrite)")
            # Example data to write
            data = {
                "status": "ok",
                "message": "Debug JSON file created successfully",
                "example_values": [1, 2, 3]
            }
            # Write to JSON file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            print(f"✅ Debug JSON file written to: {file_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to create JSON file: {e}")
            return False


# ------------------------- #
# Main
# ------------------------- #
# ------------------------- #
if __name__ == "__main__":
    clock_our_code = True
    if clock_our_code:
        custom_timer = DebugTimer
        custom_timer.default_label = "checking x_movement" 
        pass

    custom_debug_exceptions = DebugExceptions
    Gui = Make_GUI()
    keep_running = True

    while keep_running:
        Gui.show_gui()     # collect params; should probably be all thread based instead of while loop with aborts
        if not Gui.params: # user canceled
            break

        Stage = MS2000(stage_com_port="COM4", stage_baud_rate=115200, report=True)

        Scanner = HSI_Scanner(camera_index="acA1920-155um", alias_MS2k_stage=Stage)

        Gui.run_scan()

        Stage.close()
        del Stage
        del Scanner
        gc.collect()

        # --- ask if user wants to repeat ---
        choice = sg.popup_yes_no("Scan complete! Run another scan?", title="Repeat?")
        if choice != "Yes":
            break


        # Optional: break after one run if you don’t want multiple
        # keep_running = False

    # --- after loop ends ---
    time.sleep(5)   # let hardware (camera lens, etc.) settle
    print("Closing program")

    # --- choose how to exit ---
    close_window = False   # toggle this

    if not close_window:
        print("Shutting down program… (Python stops, terminal stays open)")
        time.sleep(2)
        sys.exit(0)
    else:
        print("Closing terminal window completely…")
        time.sleep(2)
        os._exit(0)


''' -------------===============-----------IMPORTANT TO DO:-------------===============-----------
-------------===============------------------------===============------------------------===============-----------
-------------===============------------------------===============------------------------===============------------------------===============-----------
high prio/debug:
    ==> Address ping-ASI console so it's almost fail_safe; will do 900 (30 x 30) brightfield scans at 0.1 mm distance delta y and delta x; 10x to see if failure happens once
        Results: PASSED - exactly 1800 files (1 data 1 header file) ✅ ✅ ✅ 
    ==> Change GUI lay-out --> Make categories more distinct and style more eye-friendly ✅ ; can still improve further
    
    ==> Power on/off for external arduino mini USB added; probably as new file for clarity on own process? Use case is for turning on/off helmholtz coils
    ==> Research turret rotation automation --> Command: "M T = 1" --> moves to turret 1 for current firmware cached addresses; can change on '@' reset
        --> Firmware version 9.2m and above has both: "SAFE_TURRET" movement and remove shortest distance turret rotation commands 
        --> cc https://asiimaging.com/docs/filter_and_turret_changer
    ==> Fix warning structure in DebugExceptions ✅X
    ==> **ADD PY_INIT FILE
    ==> **Add primary camera to own process; should be seperate from main so no dependency thread conflict (e.g. cv2) or thrown errors crashing program**⬅️⬅️⬅️⬅️
    ==>     ==> **ADD POWER SLIDER FOR BRIGHTFIELD LIGHT** ⬅️⬅️⬅️⬅️ ✅✅
          *****      ==> Need to make it in the header file be apart of typed dictionaries when swapping
    ==> Rename current files for consistency --> all caps for clarity within local repo whats actually uploaded to git✅
    ==> After start separating classes into seperate files as needed

    ==> ** Made secondary camera process close on user esc keystroke + made process shut down before closing main camera to avoid extra frames taken**
    ==>***UPLOAD TO GITHUB***✅
    
    ==> ***MAKE DISPATCH METHOD FOR SYS.EXCEPTHOOK AND THREADING.EXCEPT HOOK --> LINK PSU STATS TO TERMINAL ; for general purpose exceptions to know if theres correlated errors
    ==> **Make test case injections to make sure GUI interface is working** 
    ==> 
    ==>**CLICKING AUTOFRAMERATE ON IN SPECTRONON SETTINGS CAN CHANGE FIRMWARE TO ADJUST TO AUTO EXPOSURE WHICH OVERWRITES OUR VALUES;
        ==> AND/OR "ENABLE ACQUISITION FRAME CHECKED IN PYPYLON VIEWER ACCIDENTALLY UNDER 

    ==?? *******CANNOT HAVE THE JOYSTICK'S BUTTON DEPRESSED (in the slow mode) OTHERWISE 's' appears next to target location instead of 'fB' page 8 of manual
    ==> **Add button for magnifications to scale user input implicitly assuming brightfield**
    ==> **ATTEMPT BASIC AUTOFOCUS ATLEAST INBETWEEN WELLS IF NOT MORE OFTEN**
    ==? **SHUTDOWN OTHER CAMERA IMPLICITLY 
    ==> Make option to be able to swap settings from brightfield, fluoro (potentially phase constrast and pulse), etc.✅
    ==> FIGURE OUT WHY THE SPEED OF STAGE MOVING SLOW BREAKS FLUORO RECORDING 
    ==> Subfiles reference directories in the main 'combined_file_HSI_...'F
    ==> fix secondary camera closing
    ==> EASY: set absolute home position for wells/dishes (can confirm this in I X or I Y commands in asi console); command: TTL includes a subcommand to return report_time
         === to refresh rate of the console
    
    ==> Create a meta file that associates 'new_data_visualization's and 'get_dimensions_of_cubes' path to this files path 
    ==> Bug with meta_folder reference location seeming to lag behind current save dir path for each cube
    ==> Fix file name type and maybe allow user to name, and how many metafiles are produced? Maybe isn't actually an issue -- unsure
    ==> Should use dispatcher to make clones of some current class dynamic funcs as static/class. Broad scale architecture because some instances = real scan run time; but 
        sometimes the camera (pika xc2) is required outside of that context so a super broad generator/dispatcher is required
    ==> read/write exposure, gain, etc. to current metafile
    ==> Make dictionary of all the class variables so they can be referenced by strings regardless if value changes or not

    
    --> **fluoroescent default button**
    ==> Try threading to reduce line count; reading from the ASI MS-2000 ✅
    ==> ****CHANGE SPEED FOR WELLS IT IS GOING EXTREMELY SLOW**** ✅
    ==> fix calls in the main func so basically can have GUI open but rerun code repeatedly ; disconnect from ports after run? ✅
    ==> fix light not turning back on after running code once but not closing HSI_Scanner class ✅
    ==> Write meta_file for integration, numpy indexes, arrays, etc as a .json (UTF-8?) ✅
    ==> Make sure the bands and binning are saved to the meta file and then pulled back to calculate the meta file. ✅
    ==> fix so persistent storage of change in 'backlash' and 'acceleration' on ASI firmware level ✅ ==> indirect fix is just keep default factory but change it everytime
    
   low prio:
    --> Fix GUI lloks
    -->  run meta statistics on stagetop/camera accuracy, code performance, etc.
    -->  add safe_num to some class here
    --> flesh out some of the debug logic... need to link HSI_Scanner class deubg --> Make_GUI class --> Debug_Class
    --> Add to debugging class or subclass to keep track of time.sleeps used throughout program
    --> change imports whether theyre global header or class header (inside class or outside depending on static/dynamic context)
Before Scan:
    ==> Return GUI Settings as a variable to print-out at end and 
           - Print into a text file meta file (UTF-8)
    ==> Make user_file name defined if not default behavior 
        --> may want to change current behavior to parse save folder for any version of base_{x}_{y} then change base_name if true because
            issues where if I do a 2x1 scan then I do a 10x10 scan right after in the same folder... the first two of the 100 scans will be
            base_{x}_{y}_copy but then then the next 98 will be base{x}_{y} which is confusing because the actual 2x1 scan will have base{0}_{0}
            and base{1}_{0} so it looks like name-wise the the 2x1 scan and the last 98 of the 10x10 are part of the same scan run (if not checking date file made)
    ==> **Add dark response / response correction?**
        --> CC supplementary info on how to
During scan
    ==> ***FIX OFFSET NEED TO OFFSET ACTUALLY DATA COLLECTION BC OF RESONON USING ONLY 1600 x 924 instead of 1920 x 1080 (pypylon says 1768 x 1024?)
        --> cc https://resonon.gitlab.io/programming-docs/content/overview.html#pika-l-l-gige-lf-xc2-and-uv
        
    --> fix data loading/dimensions so it saves right; X rows is off by one
    --> clone data feature to manipulate
    --> make sure live integration/gain change keys work (should occur after every data_cube)
    CC supplementary info for code example
    --> change binning to have a median or average mode


After scan:
  --> add turn off camera feature after every scan✅
  --> #make sure camera goes home after every scan✅


During Loading Screen:
    -magnification changes
    -autofocus changes
    -default save folder changes; implement initial folder to be wherever the user saved it last time?
    
    ==> Other loading screen implement the ASI Stage Debug
    ==>


https://resonon.gitlab.io/programming-docs/content/overview.html#pika-l-l-gige-lf-xc2-and-uv
--> Resonon Pika XC2 ON SENSOR ITSELF... ROI width = 1600' ROI Height = 924 //// ROI Width = Within sample spatial  && ROI Height = Spectral  

        --> Initially sampled images in get_dimensions_of_cubes.py and sliced along axis(0, 1, 2) and found axis[1] to have best graph 
        HOWEVER it has 924 width when spectral binning was 9 which means we were sampling correctly but not recreating properly?
            --> https://sharpsight.ai/blog/numpy-axes-explained/ 
            axis 0 = Going down Y-axis
            axis 1 = Going along X-axis
            axis 2= Going along Z axis (assumed)
    ** we are moving in correct axis delta X on ASI corresponds to delta X on spectronon **
            
            https://resonon.com/content/files/Resonon---Camera-Data-Sheets-Pika-XC2.pdf also useful
         
-------------===============------------------------===============------------------------===============-----------
-------------===============------------------------===============------------------------===============------------------------===============-----------

'''