"""This module checks if the given python files can be imported without error."""

import sys
import traceback
import importlib.util
from pathlib import Path

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False

    for file in files:
        try:
            file = Path(file).resolve()
            module_name = file.stem

            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        except Exception:
            has_failure = True
            print(file)
            traceback.print_exc()
            print()

    sys.exit(1 if has_failure else 0)