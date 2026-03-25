"""Configuration centralisee pour notebooks, scripts Python et app Android.

- Fournit des chemins uniques (data, artefacts, assets Android).
- Evite de redefinir le telechargement Kaggle dans chaque notebook.
- Peut etre importe depuis n'importe quel sous-dossier en resolvant la racine.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "UTKFace"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
ANDROID_APP_DIR = BASE_DIR / "FacePredictor"
ANDROID_ASSETS_DIR = ANDROID_APP_DIR / "app" / "src" / "main" / "assets"
TESTS_DIR = BASE_DIR / "tests"
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"

# Kaggle dataset identifier
DATASET_ID = "jangedoo/utkface-new"
RAW_ZIP = RAW_DIR / "utkface-new.zip"

# Ensure directories exist (idempotent)
for _dir in (DATA_DIR, RAW_DIR, ARTIFACTS_DIR, NOTEBOOKS_DIR, TESTS_DIR, DIAGNOSTICS_DIR, ANDROID_ASSETS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def ensure_kaggle_credentials(username: Optional[str] = None, key: Optional[str] = None) -> Path:
    """Write ~/.kaggle/kaggle.json using env vars or provided args.

    Parameters
    ----------
    username : Optional[str]
        Kaggle username. Defaults to env var KAGGLE_USERNAME.
    key : Optional[str]
        Kaggle API key. Defaults to env var KAGGLE_KEY.

    Returns
    -------
    Path
        Path to the written kaggle.json file.

    Raises
    ------
    ValueError
        If credentials are missing.
    """

    user = username or os.getenv("KAGGLE_USERNAME")
    api_key = key or os.getenv("KAGGLE_KEY")

    if not user or not api_key:
        raise ValueError(
            "KAGGLE_USERNAME et KAGGLE_KEY doivent être définis (dans l'environnement ou passés en arguments)."
        )

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(mode=0o700, exist_ok=True)
    kaggle_json_path = kaggle_dir / "kaggle.json"

    with kaggle_json_path.open("w") as f:
        json.dump({"username": user, "key": api_key}, f)
    os.chmod(kaggle_json_path, 0o600)

    return kaggle_json_path


def resolve_project_root(current: Optional[Path] = None) -> Path:
    """Resolve project root (folder containing config.py) from any cwd.

    Useful inside notebooks executed from subfolders.
    """

    start = current or Path.cwd()
    for path in (start,) + tuple(start.parents):
        if (path / "config.py").exists():
            return path
    return BASE_DIR


def ensure_kaggle_dataset(download: bool = False) -> Path:
    """Ensure UTKFace dataset location is prepared.

    If ``download`` est True et que le zip n'est pas present, telecharge via Kaggle CLI.
    Evite de dupliquer le code de telechargement dans chaque notebook.
    """

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if IMAGES_DIR.exists():
        return IMAGES_DIR

    if download and not RAW_ZIP.exists():
        # Requiert kaggle CLI et creds deja configures
        os.system(f"kaggle datasets download -d {DATASET_ID} -p {RAW_DIR}")

    # Dezip si necessaire
    if RAW_ZIP.exists():
        import zipfile

        with zipfile.ZipFile(RAW_ZIP, "r") as zf:
            zf.extractall(DATA_DIR)

    return IMAGES_DIR


def describe_config() -> dict:
    """Return a summary of key paths for debugging/logging."""
    return {
        "BASE_DIR": str(BASE_DIR),
        "DATA_DIR": str(DATA_DIR),
        "RAW_DIR": str(RAW_DIR),
        "IMAGES_DIR": str(IMAGES_DIR),
        "ARTIFACTS_DIR": str(ARTIFACTS_DIR),
        "ANDROID_ASSETS_DIR": str(ANDROID_ASSETS_DIR),
        "TESTS_DIR": str(TESTS_DIR),
        "DIAGNOSTICS_DIR": str(DIAGNOSTICS_DIR),
        "DATASET_ID": DATASET_ID,
    }


def artifacts_for(theme: str, notebook: str) -> Path:
    """Return a per-notebook artifacts directory under artifacts/<theme>/<notebook>.

    Both arguments are used as folder names; directories are created if missing.
    """

    theme_dir = ARTIFACTS_DIR / theme
    target = theme_dir / notebook
    target.mkdir(parents=True, exist_ok=True)
    return target


__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "IMAGES_DIR",
    "ARTIFACTS_DIR",
    "NOTEBOOKS_DIR",
    "ANDROID_APP_DIR",
    "ANDROID_ASSETS_DIR",
    "artifacts_for",
    "TESTS_DIR",
    "DIAGNOSTICS_DIR",
    "DATASET_ID",
    "RAW_ZIP",
    "resolve_project_root",
    "ensure_kaggle_dataset",
    "ensure_kaggle_credentials",
    "describe_config",
]
