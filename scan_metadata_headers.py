# scan_metadata_headers.py
from __future__ import annotations

from typing import Final, Literal, TypeAlias, TypeVar
import warnings

# ---------- Type aliases (module-level so type checkers “see” them) ----------
StrNum: TypeAlias = str | int | float

SCHEMAS: Final = ("bright", "fluoro", "mag", "well_sample", "plate_sample", "debug_yes")
SchemaName: TypeAlias = Literal[*SCHEMAS]  # Python 3.11+ only
TSchema = TypeVar("TSchema", bound=SchemaName)

MAGS: Final = ("1x", "10x", "20x", "30x", "40x")
MagName: TypeAlias = Literal[*MAGS]


# ---------- Per-schema literal types + value constants (STATIC for checkers) ----------
SchemaBright: TypeAlias      = Literal["bright"]
SchemaFluoro: TypeAlias      = Literal["fluoro"]
SchemaMag: TypeAlias         = Literal["mag"]
SchemaWellSample: TypeAlias  = Literal["well_sample"]
SchemaPlateSample: TypeAlias = Literal["plate_sample"]
SchemaDebugYes: TypeAlias    = Literal["debug_yes"]

BRIGHT: Final[SchemaBright]            = "bright"
FLUORO: Final[SchemaFluoro]            = "fluoro"
MAG: Final[SchemaMag]                  = "mag"
WELL_SAMPLE: Final[SchemaWellSample]   = "well_sample"
PLATE_SAMPLE: Final[SchemaPlateSample] = "plate_sample"
DEBUG_YES: Final[SchemaDebugYes]       = "debug_yes"

# Handy mapping for validation / lookups
SCHEMA_CONSTS: Final[dict[str, str]] = {
    "bright": BRIGHT,
    "fluoro": FLUORO,
    "mag": MAG,
    "well_sample": WELL_SAMPLE,
    "plate_sample": PLATE_SAMPLE,
    "debug_yes": DEBUG_YES,
}

# ---------- Runtime constants (kept in a class for namespacing) ----------
class Const:
    # Canonical dict keys — keep consistent across your codebase
    KEY_SCHEMA: Final[str]      = "schema"
    KEY_X_DISTANCE: Final[str]      = "x_distance"         # canonical key
    KEY_Y_DISTANCE: Final[str]      = "y_distance"
    KEY_STAGE_SPEED: Final[str] = "stage_speed"
    KEY_EXPOSURE: Final[str]    = "exposure_time"
    KEY_GAIN: Final[str]        = "gain"
    KEY_ROWS: Final[str]        = "rows"
    KEY_COLS: Final[str]        = "cols"
    KEY_MAG: Final[str]         = "mag"

    # Defaults: BRIGHT
    VAL_BRIGHT_X_DISTANCE: Final[float]      = 0.938#1.02
    VAL_BRIGHT_Y_DISTANCE: Final[float]      = 0.938
    VAL_BRIGHT_STAGE_SPEED: Final[float] = 0.0623
    VAL_BRIGHT_EXPOSURE: Final[float]    = 16.33
    VAL_BRIGHT_GAIN: Final[int]          = 1

    # Defaults: FLUORO
    #VAL_FLUORO_X_DISTANCE: Final[float]      = 0.4
    VAL_FLUORO_X_DISTANCE: Final[float]      = 0.1

    VAL_FLUORO_Y_DISTANCE: Final[float]      = 0.938
    VAL_FLUORO_STAGE_SPEED: Final[float] = 0.0023 #0.00.0023 #was originally ~0.002 mm/s
    VAL_FLUORO_EXPOSURE: Final[float]    = 750 #427 <-- REAL #563.3 is opposite scaling; #600  #was originally 490 ms
    VAL_FLUORO_GAIN: Final[int]          = 23 #20

    # Defaults: GRID/MAG
    VAL_WELLS_ROWS: Final[int] = 5
    VAL_WELLS_COLS: Final[int] = 5 

    VAL_PLATE_ROWS: Final[int] = 20#25 #5
    VAL_PLATE_COLS: Final[int] = 20#45 #5

    VAL_DEBUG_ROWS: Final[int] = 1
    VAL_DEBUG_COLS: Final[int] = 1

    VAL_MAG_DEFAULT: Final[int] = 10

    @classmethod
    def validate(cls, *, strict: bool = False) -> None:
        """
        Runtime sanity checks:
          * SCHEMAS matches SCHEMA_CONSTS keys
          * Each exported schema constant equals its key string
          * All KEY_* attributes are strings
        """
        problems: list[str] = []

        # 1) SCHEMAS <-> SCHEMA_CONSTS keys
        set_schemas = set(SCHEMAS)
        set_map = set(SCHEMA_CONSTS.keys())
        if set_schemas != set_map:
            problems.append(
                "SCHEMAS != SCHEMA_CONSTS keys: "
                f"missing_in_map={set_schemas - set_map}, extra_in_map={set_map - set_schemas}"
            )

        # 2) Each constant value matches its schema string
        g = globals()
        for schema, const_name in {
            "bright": "BRIGHT",
            "fluoro": "FLUORO",
            "mag": "MAG",
            "well_sample": "WELL_SAMPLE",
            "plate_sample": "PLATE_SAMPLE",
            "debug_yes": "DEBUG_YES",
        }.items():
            val = g.get(const_name, None)
            if val is None:
                problems.append(f"Missing constant {const_name} for schema {schema!r}")
            elif val != schema:
                problems.append(f"Constant {const_name} should be {schema!r}, got {val!r}")

        # 3) Key names should be strings
        for attr in (
            "KEY_SCHEMA", "KEY_X_DISTANCE", "KEY_Y_DISTANCE", "KEY_STAGE_SPEED",
            "KEY_EXPOSURE", "KEY_GAIN", "KEY_ROWS", "KEY_COLS", "KEY_MAG"
        ):
            v = getattr(cls, attr, None)
            if not isinstance(v, str):
                problems.append(f"{attr} should be str, got {type(v).__name__}: {v!r}")

        if problems:
            msg = "scan_metadata_headers validation issues:\n- " + "\n- ".join(problems)
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg, UserWarning)


# Optionally run a soft validation at import time:
# Const.validate(strict=False)
