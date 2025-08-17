#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ultralytics YOLO 🚀 

This is a human-friendly, well-documented packaging script for building and
distributing the Ultralytics YOLO package. It’s deliberately verbose so that
anyone reading it can understand *why* each line exists.

Key things this script does:
  1) Reads the package version from ultralytics/__init__.py (without importing).
  2) Loads README.md as the long description (with a safe fallback).
  3) Parses requirements.txt robustly (works even if pkg_resources is missing).
  4) Declares console entry points so users can run `yolo` from the terminal.

You can adapt this template to your own project by changing the metadata in the
`setup(...)` call below.



from __future__ import annotations

import re
from pathlib import Path
from typing import List

from setuptools import find_packages, setup

# ---------------------------------------------------------------------------
# Paths & basic project constants
# ---------------------------------------------------------------------------

# Absolute path to this file and project root (portable across OSes)
FILE = Path(__file__).resolve()
ROOT = FILE.parent

# Commonly used file paths
README_PATH = ROOT / "README.md"
REQS_PATH = ROOT / "requirements.txt"
VERSION_FILE = ROOT / "ultralytics" / "__init__.py"  # where __version__ lives


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def read_text_safe(path: Path, encoding: str = "utf-8", default: str = "") -> str:
    """
    Safely read text from a file. If the file does not exist (e.g., in
    certain packaging environments), return a sensible default instead of
    crashing the build.
    """
    try:
        return path.read_text(encoding=encoding)
    except FileNotFoundError:
        return default


def parse_requirements_file(path: Path) -> List[str]:
    """
    Parse requirements.txt into a clean list of requirement specifiers.

    We *prefer* pkg_resources.parse_requirements because it understands
    extras, version pins, markers, etc. If it's unavailable or fails for
    any reason, we fall back to a simple line-by-line parser that skips
    comments and blank lines.

    Returns:
        A list like ["numpy>=1.23.0", "torch>=2.1; platform_system!='Windows'", ...]
    """
    text = read_text_safe(path)
    if not text:
        return []

    # First try the robust parser from pkg_resources
    try:
        import pkg_resources as pkg
        return [str(req) for req in pkg.parse_requirements(text)]
    except Exception:
        # Fallback: basic parsing that ignores comments and include directives
        requirements: List[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            # Skip nested requirement files like "-r other.txt" to avoid surprises
            if line.startswith(("-r", "--requirement")):
                continue
            requirements.append(line)
        return requirements


def get_version(version_file: Path = VERSION_FILE) -> str:
    """
    Extract __version__ from ultralytics/__init__.py *without importing* the package.

    Importing inside setup.py is error-prone because dependencies might not be
    installed yet. A simple regex keeps things safe and fast.
    """
    text = read_text_safe(version_file)
    match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', text, re.M)
    if not match:
        raise RuntimeError(
            f"Could not find __version__ in {version_file}. "
            "Make sure it defines a line like: __version__ = 'x.y.z'"
        )
    return match.group(1)


# ---------------------------------------------------------------------------
# Load dynamic content for setup()
# ---------------------------------------------------------------------------

LONG_DESCRIPTION = read_text_safe(
    README_PATH,
    default=(
        "Ultralytics YOLOv8 and HUB — fast, friendly, and state-of-the-art "
        "vision models with simple Python and CLI interfaces."
    ),
)

INSTALL_REQUIRES = parse_requirements_file(REQS_PATH)
PACKAGE_VERSION = get_version()


# ---------------------------------------------------------------------------
# Setup configuration
# ---------------------------------------------------------------------------

# A quick note on find_packages():
# - By default, it finds all packages under ROOT.
# - You can exclude things like tests or docs if you wish:
#     find_packages(exclude=("tests", "docs", "examples"))
packages = find_packages()

setup(
    # ---- Core package identity ------------------------------------------------
    name="ultralytics",                 # Package name on PyPI
    version=PACKAGE_VERSION,            # Pulled from ultralytics/__init__.py
    license="GPL-3.0",                  # SPDX-style identifier
    description="Ultralytics YOLOv8 and HUB",  # Short, one-line description
    long_description=LONG_DESCRIPTION,  # Full description shown on PyPI
    long_description_content_type="text/markdown",

    # ---- Project URLs & metadata ---------------------------------------------
    url="https://github.com/ultralytics/ultralytics",
    project_urls={
        "Bug Reports": "https://github.com/ultralytics/ultralytics/issues",
        "Funding": "https://ultralytics.com",
        "Source": "https://github.com/ultralytics/ultralytics",
    },
    author="Ultralytics",
    author_email="hello@ultralytics.com",

    # ---- Python & platform compatibility -------------------------------------
    python_requires=">=3.7.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
    ],

    # ---- What to include in the wheel/sdist ----------------------------------
    packages=packages,              # Discovered Python packages
    include_package_data=True,      # Respect MANIFEST.in and package_data
    # If you need to include data files explicitly, you can also add:
    # package_data={"ultralytics": ["py.typed", "resources/*"]},

    # ---- Dependencies ---------------------------------------------------------
    install_requires=INSTALL_REQUIRES,
    extras_require={
        # Install with: pip install ultralytics[dev]
        "dev": [
            "check-manifest",
            "pytest",
            "pytest-cov",
            "coverage",
            "mkdocs",
            "mkdocstrings[python]",
            "mkdocs-material",
        ],
    },

    # ---- Keywords help discoverability on PyPI -------------------------------
    keywords=(
        "machine-learning, deep-learning, vision, ML, DL, AI, YOLO, "
        "YOLOv3, YOLOv5, YOLOv8, HUB, Ultralytics"
    ),

    # ---- Console scripts (command-line entry points) -------------------------
    entry_points={
        # After installation, users can run `yolo` or `ultralytics` in the shell.
        "console_scripts": [
            "yolo = ultralytics.yolo.cli:cli",
            "ultralytics = ultralytics.yolo.cli:cli",
        ]
    },
)
