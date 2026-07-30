import sys
from pathlib import Path

# Ensure UTF-8 output on Windows so the app's ✓/⚠ startup glyphs don't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import uvicorn


def main():
    # Watch only the package for reload — NOT the whole cwd, which would include
    # .venv and cause reload storms while packages install / caches update.
    pkg_dir = str(Path(__file__).resolve().parent)
    uvicorn.run(
        "rainwords.main:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        reload_dirs=[pkg_dir],
    )



if __name__ == "__main__":
    main()
