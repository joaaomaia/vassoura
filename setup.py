\
from pathlib import Path
from setuptools import setup, find_packages

README = Path(__file__).with_name("README.md").read_text(encoding="utf8")

setup(
    name="vassoura",
    version="0.1.0",
    packages=find_packages(include=["vassoura", "vassoura.*"]),
    py_modules=["heuristics"],  # inclui o stub de reexportação
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.2",
        "numpy>=1.25",
        "seaborn>=0.13",
        "matplotlib>=3.8",
        "scipy>=1.11",
        "statsmodels>=0.14",
        "networkx>=3.2",
    ],
    author="João Maia",
    author_email="joao@example.com",
    description=(
        "Ferramentas para análise e limpeza de correlação "
        "e multicolinearidade em DataFrames."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
