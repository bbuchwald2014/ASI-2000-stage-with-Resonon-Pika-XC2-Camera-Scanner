import tkinter as tk
from pypylon import pylon
import cv2
from PIL import Image, ImageTk
import os
import sys
import re
from multiprocessing.synchronize import Event as MpEvent
import threading
#led x = 7 , integration 400k, gain 18; 
'''
PRIMARY HSI CAMERA IS:  Model=acA1920-155um, Serial=23133331, Friendly=Basler acA1920-155um (23133331) <-- SELECTED (primary)
SECONDARY CAMERA IS  :  Model=a2A2048-114umBAS, Serial=40633494, Friendly=Basler a2A2048-114umBAS (40633494)

#####Light Sources####
    -Fluoroscent LED --> CC picture above https://leddynamics.com/product/indus-star-a007-a008 ; @ 3V, 0.33 A
    -Brightfield LED --> CC picture above; ASI LED
####Filters#### -exciation filter: Brightline single-bandpass, 469 ± 35/2 nm ; T > 90%;  https://www.idex-hs.com/store/product-detail/ff01_469_35_25/fl-004536

                    -mirror: ASI c60 - d-cube-b-1002 dichroic mirror with unknown dichroic brand; 480 - 525 nm; cut-on T>90% https://www.asiimaging.com/drawings/catalog-drawings/pdf/C60%20-%20D-CUBE%20B%20ASSY.PDF
                
                        -emission filter: Brightline single-bandpass 529 ± 39/2 nm; T > 90% https://www.idex-hs.com/store/product-detail/ff01_525_39_25/fl-004547
Assume led x = 99 (100% power) for brightfield unless stated otherwise
Chicago Cancer:
    UNSTAINED CELLS:
        With Filters:
            Secondary camera settings:
                BF = 180 ms , 1 gain
                Fluoro = 1,217 ms, 30 gain
            
            HSI primary camera settings:
                BF = 16.33 ms, 1 gain
                Fluoro = ~426 ms, 22 gain
        No Filters:
            Secondary camera settings:  
                BF = 200 ms , 0 gain; led x = 50 --> can figure out dynamically by looking at timestamps (2718 frames/9 min length duration/60 seconds) = 52.6 fps -> 0.2 Hz -> 200 ms

            HSI primary camera settings:
                BF = 16.33 ms, 0 gain; led x = 50

    
    STAINED CELLS:
        With Filters:
            Secondary camera settings:  
                BF = 50 ms , 3 gain
                Fluoro = 1,300 ms, 25 gain
            
            HSI primary camera settings:
                BF = 16.33 ms, 1 gain
                Fluoro = ~426 ms, 22 gain
        No Filters:
            Secondary camera settings:  
                BF = 300 ms , 0 gain ; led x = 50
                    
            HSI primary camera settings:
                BF = 16.33 ms, 0 gain; led x = 50

    '''

# ---------- Global switches ----------
ENABLE_TIMING = True
COPY_DATA     = True
ROTATE_CCW    = False     # rotate display by 90° CCW
REFLECT_X_AXIS = True     # flip vertically (mirror across X-axis)

if ENABLE_TIMING:
    import time

# ---------- Utilities ----------
def clock(name: str = "CLOCK_IT", start: bool = True):
    if not ENABLE_TIMING:
        return lambda: None
    if start:
        t0 = time.perf_counter()
        def _stop():
            t1 = time.perf_counter()
            print(f"[{name}] took {(t1 - t0) * 1000:.2f} ms")
        return _stop

def error_log(our_error: str):
    try:
        with open("error_log.txt", "a", encoding="utf-8") as f:
            print(our_error, file=f)
    except Exception as e:
        print(f"[error_log] failed: {e}", file=sys.stderr)

# ---------- App launcher ----------
class CameraApp:
    _preset_serial: str | None = None
    root: tk.Tk | tk.Toplevel | None = None
    save_dir: str = ""

    @staticmethod
    def receive_serial(serial: str | None = None) -> str | None:
        CameraApp._preset_serial = serial
        return serial
    
    @classmethod
    def main(
        cls,
        second_camera_save_dir: str = "",
        stop_event: threading.Event | MpEvent | None = None,
        toggle_event: MpEvent | None = None  # optional toggle event
    ):
        cls.save_dir = second_camera_save_dir

        if tk._default_root is None:
            cls.root = tk.Tk()
        else:
            cls.root = tk.Toplevel(tk._default_root)

        serial_to_find = cls._preset_serial or "40633494"
        app = CameraAppChild(cls.root, serial_to_find, stop_event, toggle_event)

        if stop_event is not None:
            def watch_stop():
                stop_event.wait()
                print("[Focus GUI] Received stop signal, quitting...")
                app.quit()
            threading.Thread(target=watch_stop, daemon=True).start()

        if isinstance(cls.root, tk.Tk):
            cls.root.mainloop()


class CameraAppChild(CameraApp):
    def __init__(self, root: tk.Tk, serial_to_find: str,
                 stop_event: threading.Event | MpEvent | None = None,
                 toggle_event: MpEvent | None = None):
        self.root = root
        self.root.title("Basler Camera Live Feed")
        self.stop_event = stop_event
        self.toggle_event = toggle_event

        # --- Setup secondary camera folder ---
        base_dir = (CameraApp.save_dir or "").strip() or os.getcwd()
        self.secondary_save_dir = os.path.join(base_dir, "secondary_camera")
        os.makedirs(self.secondary_save_dir, exist_ok=True)
        print(f"[Secondary Camera] Saving snapshots to: {self.secondary_save_dir}")

        # --- Camera setup ---
        factory = pylon.TlFactory.GetInstance()
        di = pylon.CDeviceInfo()
        di.SetSerialNumber(serial_to_find)
        try:
            self.camera = pylon.InstantCamera(factory.CreateDevice(di))
            self.camera.Open()
        except pylon.GenericException as e:
            raise RuntimeError(f"Camera '{serial_to_find}' not found.") from e

        if hasattr(self.camera, "GainAuto"):
            self.camera.GainAuto.SetValue('Off')
        if hasattr(self.camera, "ExposureAuto"):
            self.camera.ExposureAuto.SetValue('Off')

        self.gain_min = getattr(self.camera.Gain, "Min", 0.0)
        self.gain_max = getattr(self.camera.Gain, "Max", 20.0)
        self.exp_min  = getattr(self.camera.ExposureTime, "Min", 100.0)
        self.exp_max  = getattr(self.camera.ExposureTime, "Max", 1000000.0)

        if hasattr(self.camera.Gain, "Value"):
            self.camera.Gain.Value = 0
        if hasattr(self.camera.ExposureTime, "SetValue"):
            self.camera.ExposureTime.SetValue(16300)#(160000)

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # --- UI elements ---
        self.label = tk.Label(root)
        self.label.pack()
        self.status = tk.Label(root, text=self.status_text(), font=("Arial", 14))
        self.status.pack()
        self.width, self.height = 1230, 905
        if ROTATE_CCW:
            self.width, self.height = self.height, self.width
        self.current_snapshot = None

        # Bind keys
        root.bind('<Up>', self.increase_gain)
        root.bind('<Down>', self.decrease_gain)
        root.bind('<Right>', self.increase_exposure)
        root.bind('<Left>', self.decrease_exposure)
        root.bind('q', self.quit)
        root.bind('<KP_Divide>', self.take_snapshot)

        self.copy_logic = self.setup_copy_func()
        self.continuous_snapshots_enabled = False  # start OFF by default

        # --- Start frame loop ---
        self.update_frame()

    # ---------- Frame loop ----------
    def update_frame(self):
        # --- Toggle continuous snapshots if event is triggered ---
        if self.toggle_event and self.toggle_event.is_set():
            self.continuous_snapshots_enabled = not self.continuous_snapshots_enabled
            state = "ON" if self.continuous_snapshots_enabled else "OFF"
            print(f"[Secondary Camera] Continuous snapshots toggled {state}")
            self.toggle_event.clear()

        # --- Grab frame if camera is active ---
        if self.camera.IsGrabbing():
            try:
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    src = grab_result.Array
                    img = self.convert_image_coloring_scheme(src)
                    self.current_snapshot = img
                    self.produce_image_with_modifiers(img)
                    # Save automatically if enabled
                    if self.continuous_snapshots_enabled:
                        self.save_snapshot(img)
                grab_result.Release()
            except pylon.GenericException as e:
                error_log(f"RetrieveResult failed: {e}")

        self.status.config(text=self.status_text())
        self.root.after(10, self.update_frame)

    # ---------- Save snapshot ----------
    def save_snapshot(self, img):
        counter = 0
        filename = os.path.join(self.secondary_save_dir, f"snapshot_{counter}.png")
        while os.path.exists(filename):
            counter += 1
            filename = os.path.join(self.secondary_save_dir, f"snapshot_{counter}.png")
        Image.fromarray(img).save(filename)
        print(f"[Secondary Camera] Saved {filename}")

    # ---------- Status ----------
    def status_text(self) -> str:
        try:
            g = float(self.camera.Gain.Value)
        except (pylon.GenericException, Exception):
            g = 0.0
        try:
            e = float(self.camera.ExposureTime.Value)
        except (pylon.GenericException, Exception):
            e = 0.0
        return f"Gain: {g:.2f} | Exposure: {e:.0f} µs"

    # ---------- Image transforms ----------
    def convert_image_coloring_scheme(self, img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def produce_image_with_modifiers(self, img):
        im = Image.fromarray(img).resize((self.width, self.height), Image.Resampling.BILINEAR)
        if ROTATE_CCW:
            im = im.transpose(Image.Transpose.ROTATE_90)
        if REFLECT_X_AXIS:
            im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        imgtk = ImageTk.PhotoImage(image=im)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

    # ---------- Controls ----------
    def increase_gain(self, event=None):
        new_val = min(self.camera.Gain.Value + 1, self.gain_max)
        self.camera.Gain.Value = new_val
        print(f"Gain set to: {new_val:.2f}")

    def decrease_gain(self, event=None):
        new_val = max(self.camera.Gain.Value - 1, self.gain_min)
        self.camera.Gain.Value = new_val
        print(f"Gain set to: {new_val:.2f}")

    def increase_exposure(self, event=None):
        cur = float(self.camera.ExposureTime.Value)
        new_val = min(cur + 30000, self.exp_max)
        self.camera.ExposureTime.SetValue(new_val)
        print(f"Exposure set to: {new_val:.0f} µs")

    def decrease_exposure(self, event=None):
        cur = float(self.camera.ExposureTime.Value)
        new_val = max(cur - 3000, self.exp_min)
        self.camera.ExposureTime.SetValue(new_val)
        print(f"Exposure set to: {new_val:.0f} µs")

    # ---------- Manual snapshot ----------
    def take_snapshot(self, event=None):
        if self.current_snapshot is not None:
            self.save_snapshot(self.current_snapshot)

    # ---------- Setup copy ----------
    def setup_copy_func(self):
        return (lambda img: img.copy()) if COPY_DATA else (lambda img: img)

    # ---------- Cleanup ----------
    def quit(self, event=None):
        print("Quitting...")
        if self.stop_event:
            self.stop_event.set()
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        self.camera.Close()
        self.root.destroy()

# ---------- Entry ----------
if __name__ == "__main__":
    CameraApp.main()
