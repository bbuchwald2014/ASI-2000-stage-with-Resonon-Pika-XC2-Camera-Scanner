from typing import TypedDict, Literal, overload, Any, Optional, Union
from typing import Final #make it dynamically
import scan_metadata_headers
# ---- Schemas ----a

class MagnificationParams(TypedDict):
    schema: Literal["mag"]
    mag: Union[str, int, float]

class WellParams(TypedDict):
    schema: Literal["well_sample"]
    rows: Union[str, int]
    cols: Union[str, int]

class PlateParams(TypedDict):
    schema: Literal["plate_sample"]
    rows: Union[str, int]
    cols: Union[str, int]

class BrightfieldParams(TypedDict):
    schema: Literal["bright"]
    x_dist: Union[str, int, float]
    y_dist: Union[str, int, float]
    stage_speed: Union[str, int, float]
    exposure_time: Union[str, int, float]
    gain: Union[str, int, float]

class FluoroParams(TypedDict):
    schema: Literal["fluoro"]
    x_dist: Union[str, int, float]
    y_dist: Union[str, int, float]
    stage_speed: Union[str, int, float]
    exposure_time: Union[str, int, float]
    gain: Union[str, int, float]

class DebugParams(TypedDict):
    schema: Literal["debug_yes"]
    # union of fields you plan to merge in
    mag: Union[str, int, float]
    rows: Union[str, int]
    cols: Union[str, int]
    x_dist: Union[str, int, float]
    y_dist: Union[str, int, float]
    stage_speed: Union[str, int, float]
    exposure_time: Union[str, int, float]
    gain: Union[str, int, float]


ParamsUnion = Union[
    BrightfieldParams,
    FluoroParams,
    MagnificationParams,
    WellParams,
    PlateParams,
    DebugParams,
]
X_DISTANCE = "x_distance"
x_dist = X_DISTANCE

# ---- Factory with defaults ----
class Make_Dict:
    # Overloads
    @overload
    @staticmethod
    def create__dict(*, schema: Literal["bright"],
                     x_dist: Optional[Union[str,int,float]] = ...,
                     y_dist: Optional[Union[str,int,float]] = ...,
                     stage_speed: Optional[Union[str,int,float]] = ...,
                     exposure_time: Optional[Union[str,int,float]] = ...,
                     gain: Optional[Union[str,int,float]] = ...,
                     ) -> BrightfieldParams: ...
    @overload
    @staticmethod
    def create__dict(*, schema: Literal["fluoro"],
                     x_dist: Optional[Union[str,int,float]] = ...,
                     y_dist: Optional[Union[str,int,float]] = ...,
                     stage_speed: Optional[Union[str,int,float]] = ...,
                     exposure_time: Optional[Union[str,int,float]] = ...,
                     gain: Optional[Union[str,int,float]] = ...,
                     ) -> FluoroParams: ...
    @overload
    @staticmethod
    def create__dict(*, schema: Literal["mag"],
                     mag: Optional[Union[str,int,float]] = ...,
                     ) -> MagnificationParams: ...
    @overload
    @staticmethod
    def create__dict(*, schema: Literal["well_sample"],
                     rows: Optional[Union[str,int]] = ...,
                     cols: Optional[Union[str,int]] = ...,
                     ) -> WellParams: ...
    @overload
    @staticmethod
    def create__dict(*, schema: Literal["plate_sample"],
                     rows: Optional[Union[str,int]] = ...,
                     cols: Optional[Union[str,int]] = ...,
                     ) -> PlateParams: ...
    @overload
    @staticmethod
    def create__dict(*, schema: Literal["debug_yes"], **kw: Any) -> DebugParams: ...

    @staticmethod
    def create__dict(*, schema: Literal[
        "bright", "fluoro", "mag", "well_sample", "plate_sample", "debug_yes"
    ], **kw: Any) -> ParamsUnion:
        """
        Returns a TypedDict-compliant dict with sensible defaults per schema.

        bright:       x_dist=1.02,   y_dist=0.938, stage_speed=0.0623, exposure_time=16.33, gain=6
        fluoro:       x_dist=0.02,   y_dist=0.938, stage_speed=0.0023, exposure_time=490.0, gain=20
        mag:          mag=10
        well_sample:  rows=4, cols=4, mag='10x' (if you want that attached elsewhere)
        plate_sample: rows=5, cols=5, mag='10x'
        debug_yes:    merged view: bright + plate_sample (+ optional mag), no key overrides
        """
        if schema == "bright":
            return {
                "schema": "bright",
                "x_dist":        kw.get("x_distance", 1.02),
                "y_dist":        kw.get("y_distance", 0.938),
                "stage_speed":   kw.get("stage_speed", 0.0623),
                "exposure_time": kw.get("exposure_time", 16.33),
                "gain":          kw.get("gain", 6),
            }

        if schema == "fluoro":
            return {
                "schema": "fluoro",
                "x_dist":        kw.get("x_distance", 0.02),
                "y_dist":        kw.get("y_distance", 0.938),
                "stage_speed":   kw.get("stage_speed", 0.0023),
                "exposure_time": kw.get("exposure_time", 490.0),
                "gain":          kw.get("gain", 20),
            }

        if schema == "mag":
            return {
                "schema": "mag",
                "mag": kw.get("mag", 10),
            }

        if schema == "well_sample":
            return {
                "schema": "well_sample",
                "rows": kw.get("rows", 4),
                "cols": kw.get("cols", 4),
            }

        if schema == "plate_sample":
            return {
                "schema": "plate_sample",
                "rows": kw.get("rows", 5),
                "cols": kw.get("cols", 5),
            }

        if schema == "debug_yes":
            # Build sub-dicts fresh
            
            parts: list[dict[str, Any]] = [
                Make_Dict.create__dict(schema="bright"),
                Make_Dict.create__dict(schema="plate_sample"),
            ]
            # Optional: include magnification in the debug union
            if "mag" in kw:
                parts.append(Make_Dict.create__dict(schema="mag", mag=kw["mag"]))

            # Merge without overrides: first source wins
            merged: dict[str, Any] = {
                "rows": kw.get("rows", 1),
                "cols": kw.get("cols", 1),
            }
         # Force the debug schema tag
            merged["schema"] = "debug_yes"
            
            for d in parts:
                for k, v in d.items():
                    if k not in merged:
                        merged[k] = v

            # Ensure required fields exist (populate minimal defaults if missing)
            merged.setdefault("mag", kw.get("mag", 10))
            merged.setdefault("rows", 5)
            merged.setdefault("cols", 5)
            merged.setdefault("x_dist", 1.02)
            merged.setdefault("y_dist", 0.938)
            merged.setdefault("stage_speed", 0.0623)
            merged.setdefault("exposure_time", 16.33)
            merged.setdefault("gain", 6)

            # Type-wise, this now conforms to DebugParams
            return merged  # type: ignore[return-value]

        raise ValueError(f"Unknown schema {schema!r}")
#__main__ = Make_Dict.create__dict(schema= "debug_yes")
#print(__main__)