from setuptools import setup
from setuptools import find_packages

__minimum_jax_version__ = '0.4.34'

setup_requires = ['jax>=' + __minimum_jax_version__]

with open("README.md", "r") as fh:
    long_description = fh.read()    
    
setuptools.setup(
    name="darksirens",
    version='0.0.1',
    author="Ignacio Magana Hernandez",
    author_email="imhernan@andrew.cmu.edu",
    description="A package for joint gravitational wave inference with large scale galaxy surveys.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ignaciomagana/darksirens",
    setup_requires=setup_requires,
    packages=[
        "darksirens",
        "darksirens.gw",
        "darksirens.em",
        "darksirens.inference"
    ],
    entry_points={
        "console_scripts": [
            "darksirens_inference=darksirens.inference.inference:main",
            "darksirens_pixelate_survey=darksirens.em.pixelate_survey:main",
        ]
    },
    classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: Apache Software License",
      "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
