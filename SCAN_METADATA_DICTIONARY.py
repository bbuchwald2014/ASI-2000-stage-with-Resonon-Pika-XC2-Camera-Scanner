# params_factory.py
from __future__ import annotations

from typing import Any, Literal, Required, TypedDict, Unpack, overload, cast
from scan_metadata_headers import Const, StrNum, SchemaName


# ---------- Mixins ----------
class HasSchema(TypedDict):
    schema: Required[SchemaName]

class HasXY(TypedDict, total=False):
    x_dist: StrNum
    y_dist: StrNum

class HasSpeed(TypedDict, total=False):
    stage_speed: StrNum

class HasExposure(TypedDict, total=False):
    exposure_time: StrNum
    gain: StrNum

class HasGrid(TypedDict, total=False):
    rows: str | int
    cols: str | int

class HasSuperGrid(TypedDict, total=False):
    super_rows: StrNum
    off_set_super_rows: StrNum
    delta_super_rows: StrNum
    super_columns: StrNum
    off_set_super_columns: StrNum
    delta_super_columns: StrNum


# ---------- Concrete param shapes ----------
class MagnificationParams(HasSchema):
    schema: Literal["mag"]
    mag: StrNum

class WellParams(HasSchema, HasGrid):
    schema: Literal["well_sample"]

class PlateParams(HasSchema, HasGrid):
    schema: Literal["plate_sample"]

class BrightfieldParams(HasSchema, HasXY, HasSpeed, HasExposure):
    schema: Literal["bright"]

class FluoroParams(HasSchema, HasXY, HasSpeed, HasExposure, HasGrid, HasSuperGrid):
    schema: Literal["fluoro"]

# Keep total=True; we’ll return a fully-populated dict for debug
class DebugParams(HasSchema, HasXY, HasSpeed, HasExposure, HasGrid, total=True):
    schema: Literal["debug_yes"]
    mag: StrNum


ParamsUnion = (
    BrightfieldParams
    | FluoroParams
    | MagnificationParams
    | WellParams
    | PlateParams
    | DebugParams
)


# ---------- Centralized defaults (flat, using Const where available) ----------
_DEFAULTS: dict[str, dict[str, StrNum]] = {
    "bright": {
        Const.KEY_X_DISTANCE:      Const.VAL_BRIGHT_X_DISTANCE,
        Const.KEY_Y_DISTANCE:      Const.VAL_BRIGHT_Y_DISTANCE,
        Const.KEY_STAGE_SPEED: Const.VAL_BRIGHT_STAGE_SPEED,
        Const.KEY_EXPOSURE:    Const.VAL_BRIGHT_EXPOSURE,
        Const.KEY_GAIN:        Const.VAL_BRIGHT_GAIN,
        Const.KEY_ROWS:        Const.VAL_PLATE_ROWS,
        Const.KEY_COLS:        Const.VAL_PLATE_COLS,
    },   
    "fluoro": {
        Const.KEY_X_DISTANCE:      Const.VAL_FLUORO_X_DISTANCE,
        Const.KEY_Y_DISTANCE:      Const.VAL_FLUORO_Y_DISTANCE,
        Const.KEY_STAGE_SPEED: Const.VAL_FLUORO_STAGE_SPEED,
        Const.KEY_EXPOSURE:    Const.VAL_FLUORO_EXPOSURE,
        Const.KEY_GAIN:        Const.VAL_FLUORO_GAIN,
        Const.KEY_ROWS:        Const.VAL_WELLS_ROWS,
        Const.KEY_COLS:        Const.VAL_WELLS_COLS,
        # Accept grid / super-grid overrides for fluoro:
        #Const.KEY_ROWS:  5,
        #Const.KEY_COLS:  5,
        "super_rows":             1,
        "off_set_super_rows":     0,
        "delta_super_rows":       0,
        "super_columns":          1,
        "off_set_super_columns":  0,
        "delta_super_columns":    0,
    },
    "well_sample": {
        Const.KEY_ROWS: Const.VAL_WELLS_ROWS,
        Const.KEY_COLS: Const.VAL_WELLS_COLS,
    },
    "plate_sample": {
        Const.KEY_ROWS: Const.VAL_PLATE_ROWS,
        Const.KEY_COLS: Const.VAL_PLATE_COLS,
    },
    "mag": {
        Const.KEY_MAG: Const.VAL_MAG_DEFAULT,
    },
}
# ---------- Tiny builder helper (flat, schema-aware) ----------
def _build_from(
    schema: Literal["bright", "fluoro", "mag", "well_sample", "plate_sample"],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Start with per-schema defaults, stamp schema tag, overlay recognized keys.
    Unknown keys are ignored (prevents typos leaking in).
    """
    base = _DEFAULTS[schema]
    out: dict[str, Any] = {Const.KEY_SCHEMA: schema, **base}
    for k, v in overrides.items():
        if k in base:
            out[k] = v
    return out


# ---------- Small builders (flat, 1-liners) ----------
def _build_bright(**kw: Unpack[BrightfieldParams]) -> BrightfieldParams:
    return cast(BrightfieldParams, _build_from("bright", kw))

def _build_fluoro(**kw: Unpack[FluoroParams]) -> FluoroParams:
    return cast(FluoroParams, _build_from("fluoro", kw))

def _build_mag(**kw: Unpack[MagnificationParams]) -> MagnificationParams:
    return cast(MagnificationParams, _build_from("mag", kw))

def _build_well(**kw: Unpack[WellParams]) -> WellParams:
    return cast(WellParams, _build_from("well_sample", kw))

def _build_plate(**kw: Unpack[PlateParams]) -> PlateParams:
    return cast(PlateParams, _build_from("plate_sample", kw))

def _build_debug(**kw: Unpack[DebugParams]) -> DebugParams:
    """
    Debug = bright defaults + plate defaults + user overrides.
    Ensures total=True by filling all fields before returning.
    """
    out: dict[str, Any] = {
        Const.KEY_SCHEMA: "debug_yes",
        **_DEFAULTS["bright"],
        **_DEFAULTS["plate_sample"],
        Const.KEY_MAG: Const.VAL_MAG_DEFAULT,  # seed mag
    }

    # Apply user overrides for recognized keys in debug context
    for k in (
        Const.KEY_MAG, Const.KEY_ROWS, Const.KEY_COLS,
        Const.KEY_X_DISTANCE, Const.KEY_Y_DISTANCE,
        Const.KEY_STAGE_SPEED, Const.KEY_EXPOSURE, Const.KEY_GAIN
    ):
        if k in kw:
            out[k] = kw[k]

    return cast(DebugParams, out)


# ---------- Factory with overloads ----------
class MakeParams:
    @overload
    @staticmethod
    def create__dict(*, schema: Literal["bright"], **kw: Unpack[BrightfieldParams]) -> BrightfieldParams: ...
    @overload
    @staticmethod
    def create__dict(*, schema: Literal["fluoro"], **kw: Unpack[FluoroParams]) -> FluoroParams: ...
    @overload
    @staticmethod
    def create__dict(*, schema: Literal["mag"], **kw: Unpack[MagnificationParams]) -> MagnificationParams: ...
    @overload
    @staticmethod
    def create__dict(*, schema: Literal["well_sample"], **kw: Unpack[WellParams]) -> WellParams: ...
    @overload
    @staticmethod
    def create__dict(*, schema: Literal["plate_sample"], **kw: Unpack[PlateParams]) -> PlateParams: ...
    @overload
    @staticmethod
    def create__dict(*, schema: Literal["debug_yes"], **kw: Unpack[DebugParams]) -> DebugParams: ...

    @staticmethod
    def create__dict(*, schema: SchemaName, **kw: Any) -> ParamsUnion:
        if schema == "bright":
            return _build_bright(**kw)
        if schema == "fluoro":
            return _build_fluoro(**kw)
        if schema == "mag":
            return _build_mag(**kw)
        if schema == "well_sample":
            return _build_well(**kw)
        if schema == "plate_sample":
            return _build_plate(**kw)
        if schema == "debug_yes":
            return _build_debug(**kw)
        raise ValueError(f"Unknown schema: {schema!r}")
