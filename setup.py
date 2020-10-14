import setuptools
import re

VERSIONFILE="darksirens/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setuptools.setup(
    name="darksirens",
    version=verstr,
    author="Ignacio Magana Hernandez",
    author_email="maganah2@uwm.edu",
    description="A package for joint gravitational wave inference with large scale galaxy surveys.",
    long_description="LONG DESCRIPTION HERE",
    url="https://github.com/ignaciomagana/darksirens",
    packages=[
        "darksirens",
        "darksirens.gw",
        "darksirens.em",
        "darksirens.inference"
    ],
    entry_points={
        "console_scripts": [
            "darksirens_inference=darksirens.inference.inference:main",
        ]
    },
    install_requires=[
        "bilby",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ],
    python_requires='>=3.6',
)
