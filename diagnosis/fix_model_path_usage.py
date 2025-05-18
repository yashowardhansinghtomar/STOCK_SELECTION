# diagnosis/fix_model_path_usage.py
"""
Fixes incorrect usage of `load_data["model_key"]` to `PATHS["model_key"]`.

Run with:
    python -m diagnosis.fix_model_path_usage
"""

import re
from pathlib import Path

ROOT = Path(".")
TARGET_EXTENSIONS = {".py"}

# List of model keys that should use PATHS, not load_data
MODEL_KEYS = {
    "filter_model",
    "regressor_model",
    "exit_model",
    "trade_classifier",
    "return_regressor",
    "meta_models",
    "exit_metadata",
}

pattern = re.compile(r'load_data\s*\[\s*"(' + '|'.join(MODEL_KEYS) + r')"\s*\]')

def fix_file(filepath: Path):
    code = filepath.read_text(encoding="utf-8")
    fixed_code = pattern.sub(lambda m: f'PATHS["{m[1]}"]', code)

    if fixed_code != code:
        filepath.write_text(fixed_code, encoding="utf-8")
        print(f"✅ Fixed: {filepath}")

def main():
    print("🔍 Scanning project for incorrect model path usage...")
    for path in ROOT.rglob("*"):
        if path.suffix in TARGET_EXTENSIONS and path.is_file():
            fix_file(path)

if __name__ == "__main__":
    main()
