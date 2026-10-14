"""Compatibility shim so legacy pip versions can do an editable install.

All project metadata lives in pyproject.toml; this file just gives older pip
(which requires a setuptools entry point for ``pip install -e .``) something to
find.
"""

from setuptools import setup

setup()
