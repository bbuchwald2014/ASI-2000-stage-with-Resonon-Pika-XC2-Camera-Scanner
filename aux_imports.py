# aux_imports.py  (fixed)

from __future__ import annotations
import os, sys, subprocess, importlib, importlib.util
from typing import Iterable, Optional
from pathlib import Path

# ----------------------------
# pip helpers
# ----------------------------

def _run_pip(args: list[str]) -> tuple[int, str, str]:
    cmd = [sys.executable, "-m", "pip", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def ensure_packages(packages: Iterable[str], quiet: bool = True) -> None:
    for name in packages:
        try:
            importlib.import_module(name)
            if not quiet:
                print(f"[OK] {name} already installed")
            continue
        except Exception:
            pass
        code, out, err = _run_pip(["install", "--disable-pip-version-check", name])
        if code != 0:
            raise RuntimeError(f"[ERROR] pip install {name!r} failed:\n{err or out}")
        if not quiet:
            print(f"[OK] Installed {name}")

# ----------------------------
# dynamic local import helper
# ----------------------------

def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not create spec for {module_name} at {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[module_name] = mod
    return mod

def _import_local_package(package_name: str, start_dir: str) -> Optional[object]:
    """
    Recursively search for:
      - package folder 'package_name/' containing '__init__.py', or
      - a single-file module 'package_name.py'
    under start_dir. Import and return the module, or None if not found.
    """
    start_dir = os.path.abspath(start_dir)
    for dirpath, dirnames, filenames in os.walk(start_dir):
        # package folder
        if package_name in dirnames:
            pkg_path = os.path.join(dirpath, package_name)
            init_file = os.path.join(pkg_path, "__init__.py")
            if os.path.isfile(init_file):
                sys.path.insert(0, os.path.dirname(pkg_path))
                try:
                    return importlib.import_module(package_name)
                except Exception:
                    pass
        # single-file module
        candidate = os.path.join(dirpath, f"{package_name}.py")
        if os.path.isfile(candidate):
            try:
                return _import_from_path(package_name, candidate)
            except Exception:
                pass
    return None

# ----------------------------
# public APIs you’ll call
# ----------------------------

def import_pysimplegui(root_dir: Optional[str] = None):
    """
    Import PySimpleGUI.
    Strategy:
      1) Try normal import ('PySimpleGUI', then 'pysimplegui').
      2) If not found, search possible roots for either a package folder or a '.py' file.
    Returns the imported module, or raises ModuleNotFoundError.
    """
    # Valid module names for importlib.import_module:
    import_names = ("PySimpleGUI", "pysimplegui")

    # 1) Try standard import
    for name in import_names:
        try:
            mod = importlib.import_module(name)
            print(f"[OK] Imported {name} ({mod})")
            return mod
        except ModuleNotFoundError:
            pass

    # 2) Search locally
    roots: list[Path] = []
    if root_dir is not None:
        roots.append(Path(root_dir))
    else:
        roots.append(Path.cwd())
    # also try the repo root = two levels up from this file
    roots.append(Path(__file__).resolve().parent.parent)

    # Also consider common local names for filesystem search (folders/files)
    fs_names = ("PySimpleGUI", "pysimplegui", "PySimpleGUI-4-foss")

    for root in dict.fromkeys(roots):  # de-dup, order preserved
        for name in fs_names:
            mod = _import_local_package(name, str(root))
            if mod is not None:
                print(f"[OK] Imported local {name} from {root}")
                return mod

    # Nothing worked
    raise ModuleNotFoundError(
        "Could not import 'PySimpleGUI'/'pysimplegui' and no local clone was found under: "
        + ", ".join(str(r) for r in roots)
    )

def safe_import(
    package_name: str,
    module_name: str,
    install_name: Optional[str] = None,
    auto_install: bool = True,
    search_dirs: Optional[list[str]] = None
):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"[WARN] {module_name} not installed, trying local import...")
        dirs = list(search_dirs or [os.getcwd()])
        for dir_ in dirs:
            try:
                mod = _import_local_package(package_name, dir_)
                if mod:
                    print(f"[OK] Found local '{package_name}' under {dir_}")
                    return mod
            except Exception as e:
                print(f"[INFO] Failed local import from {dir_}: {e}")
        if not auto_install:
            raise
        ensure_packages([install_name or module_name], quiet=False)
        return importlib.import_module(module_name)

__all__ = ["ensure_packages", "import_pysimplegui", "safe_import"]

'''
if __name__ == "__main__":
    # No side effects on import; only run when executed directly.
    mod = import_pysimplegui()
    print(mod)
'''
#debug test
