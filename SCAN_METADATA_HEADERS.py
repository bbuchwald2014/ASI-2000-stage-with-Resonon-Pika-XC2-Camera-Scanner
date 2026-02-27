# scan_metadata_headers.py
from __future__ import annotations

from typing import Final, Literal, TypeAlias, TypeVar, Mapping
import warnings

# ---------- Type aliases (module-level so type checkers “see” them) ----------
StrNum: TypeAlias = str | int | float

SCHEMAS_FOR_DICTIONARY: Final = ("bright", "fluoro", "mag", "well_sample", "plate_sample", "debug_yes", "z_stack")
SchemaName: TypeAlias = Literal[*SCHEMAS_FOR_DICTIONARY]  # Python 3.11+ only
TSchema = TypeVar("TSchema", bound=SchemaName)

MAGS: Final = ("1x", "10x", "20x", "40x", "60x")
MagName: TypeAlias = Literal[*MAGS]
Mag_TSchema = TypeVar("Mag_TSchema", bound = MagName)

# ---------- Per-schema literal types + value constants (STATIC for checkers) ----------
SchemaBright: TypeAlias      = Literal["bright"]
SchemaFluoro: TypeAlias      = Literal["fluoro"]
SchemaMag: TypeAlias         = Literal["mag"]
SchemaWellSample: TypeAlias  = Literal["well_sample"]
SchemaPlateSample: TypeAlias = Literal["plate_sample"]
SchemaDebugYes: TypeAlias    = Literal["debug_yes"]
SchemaZStack: TypeAlias      = Literal["z_stack"]

BRIGHT: Final[SchemaBright]            = "bright"
FLUORO: Final[SchemaFluoro]            = "fluoro"
MAG: Final[SchemaMag]                  = "mag"
WELL_SAMPLE: Final[SchemaWellSample]   = "well_sample"
PLATE_SAMPLE: Final[SchemaPlateSample] = "plate_sample"
DEBUG_YES: Final[SchemaDebugYes]       = "debug_yes"
Z_STACK: Final[SchemaZStack]           = "z_stack"
#Z_STACK: Final[]

# Handy mapping for validation / lookups; can't make this dict dynamic otherwise defeats purpose of interpretor checking pre-execution;
# dictionary is used during run tume though; Should maybe invert the key/value constants here main func calls this section at runtime so unclear
SCHEMA_CONSTS: Final[dict[SchemaName, SchemaName]] = {
    "bright": BRIGHT,
    "fluoro": FLUORO,
    "mag": Mag_TSchema,
    "well_sample": WELL_SAMPLE,
    "plate_sample": PLATE_SAMPLE,
    "debug_yes": DEBUG_YES,
    "z_stack": Z_STACK
}

MAG_CONSTS: Final[dict[MagName, MagName]] = {
    "1x:" : MAGS[0],
    "10x" : MAGS[1],
    "20x" : MAGS[2],
    "40x" : MAGS[3],
    "60x" : MAGS[4],
}
# ---------- Runtime constants (kept in a class for namespacing) ----------
class Const:
    # Canonical dict keys — keep consistent across your codebase
    KEY_SCHEMA: Final[str]      = "schema"
    KEY_X_DISTANCE: Final[str]      = "x_distance"         # canonical key
    KEY_Y_DISTANCE: Final[str]      = "y_distance"
    KEY_Z_DISTANCE: Final[str]      = "z_distance"
    
    KEY_X_STAGE_SPEED: Final[str] = "x_stage_speed" #exclude y_speed as user can only mess up the overall scan duration this way
    KEY_Z_STAGE_SPEED: Final[str] = "z_stage_speed" 
    KEY_NUM_Z_SLICES: Final[str] = "num_z_slices"
    
    KEY_EXPOSURE: Final[str]    = "exposure_time"
    KEY_GAIN: Final[str]        = "gain"
    KEY_ROWS: Final[str]        = "rows"
    KEY_COLS: Final[str]        = "cols"
    KEY_MAG: Final[str]         = "mag"
    KEY_DEF_LED_POW: Final[str] = "default_led_power"
    
    #Defaults: Z_stack -- have for now as brightfield/fluoro agnostic:
    VAL_Z_STAGE_SPEED: Final[float] = 5e-2
    VAL_Z_DISTANCE: Final[float] = 5e-3#1e-3 #equivalent to 5 micron here since distance should be in terms of mm here even if asi firmware takes in 1/10000 of such
    VAL_NUM_Z_SLICES: Final[int] = 5
    
    # Defaults: BRIGHT
    VAL_BRIGHT_X_DISTANCE: Final[float]      = 0.845 #1.02 gives roughly 1001 lines on 16.33 ms exposure + 0.0623 mm/s stage speed
    VAL_BRIGHT_Y_DISTANCE: Final[float]      = 0.845
    VAL_BRIGHT_X_STAGE_SPEED: Final[float] = 0.0623 #0.0312 #0.0312 # is original calibrated speed on 10x
    VAL_BRIGHT_EXPOSURE: Final[float]    = 16.33
    VAL_BRIGHT_GAIN: Final[int]          = 1
    VAL_BRIGHT_LED_POWER: Final[int]     = 99 #by default put on max power for brightfield; user can still modify final result; 0-99 indexed
    
    # Defaults: FLUORO
    #VAL_FLUORO_X_DISTANCE: Final[float]      = 0.4
    VAL_FLUORO_X_DISTANCE: Final[float]      = 0.1
    VAL_FLUORO_Y_DISTANCE: Final[float]      = 0.845 #was wrongfully 0.938 based on sensor calculation
    VAL_FLUORO_X_STAGE_SPEED: Final[float] = 0.0023 #0.00.0023 #was originally ~0.002 mm/s
    VAL_FLUORO_EXPOSURE: Final[float]    = 563 #427 <-- REAL #563.3 is opposite scaling; #600  #was originally 490 ms
    VAL_FLUORO_GAIN: Final[int]          = 0 #20
    VAL_FLUORO_LED_POWER: Final[int]     = 0
    
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
        set_schemas = set(SCHEMAS_FOR_DICTIONARY)
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
            "z_stack": "Z_STACK",
        }.items():
            val = g.get(const_name, None)
            if val is None:
                problems.append(f"Missing constant {const_name} for schema {schema!r}")
            elif val != schema:
                problems.append(f"Constant {const_name} should be {schema!r}, got {val!r}")

        # 3) Key names should be strings
        for attr in (
            "KEY_SCHEMA", "KEY_X_DISTANCE", "KEY_Y_DISTANCE", "KEY_STAGE_SPEED",
            "KEY_EXPOSURE", "KEY_GAIN", "KEY_ROWS", "KEY_COLS", "KEY_MAG", "KEY_Z_DISTANCE",
            "KEY_Z_STAGE_SPEED", "KEY_Z_NUM_SLICES"
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
