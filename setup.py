import os, sys
sys.path.append('./src')
import setuptools
from indobenchmark.version import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="indobenchmark-toolkit",
    version=__version__,
    author="Samuel Cahyawijaya",
    author_email="samuel.cahyawijaya@gmail.com",
    description="Indobenchmark toolkit for supporting IndoNLU and IndoNLG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamuelCahyawijaya/indobenchmark-toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/SamuelCahyawijaya/indobenchmark-toolkit/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "transformers>=4.3.2",
        "sentencepiece>=0.1.95",
        "datasets>=1.4.1",
        "torch>=1.7.1"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
