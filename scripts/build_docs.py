# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sphinx<9",
#     "furo",
#     "autoclasstoc",
#     "pymde",
# ]
#
# [tool.uv.sources]
# pymde = { path = "..", editable = true }
# ///
"""Build the pymde documentation using Sphinx."""

import shutil
import subprocess
import sys
from pathlib import Path

DOCS_SRC = Path(__file__).resolve().parent.parent / "docs_src"
SOURCE_DIR = DOCS_SRC / "source"
BUILD_DIR = DOCS_SRC / "build"
OUTPUT_DIR = DOCS_SRC.parent / "docs"
CNAME_SRC = DOCS_SRC / "_CNAME"


def main():
    try:
        import pymde  # noqa: F401
    except ImportError:
        print(
            "Error: pymde must be installed for autodoc to work.\n"
            "Install it in development mode: pip install -e '.[dev]'",
            file=sys.stderr,
        )
        sys.exit(1)

    result = subprocess.run(
        [
            sys.executable, "-m", "sphinx",
            "-M", "html",
            str(SOURCE_DIR),
            str(BUILD_DIR),
        ],
        cwd=DOCS_SRC,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)

    html_dir = BUILD_DIR / "html"

    # Mark for GitHub Pages (no Jekyll processing)
    (html_dir / ".nojekyll").touch()

    # Replace docs/ with freshly built HTML
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    shutil.move(str(html_dir), str(OUTPUT_DIR))

    # Custom domain CNAME
    if CNAME_SRC.exists():
        shutil.copy2(str(CNAME_SRC), str(OUTPUT_DIR / "CNAME"))

    print(f"Documentation built in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
