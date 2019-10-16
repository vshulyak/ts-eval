import os

import setuptools


HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with open(os.path.join(HERE, *parts), "r") as fh:
        return fh.read()


LONG_DESCRIPTION = read("README.md") + read("CHANGELOG.md")

INSTALL_REQUIRES = [
    "pandas>=0.23.0",
    "numpy>=1.16.0",
    "xarray>=0.13",
    "scipy>=1.3",
    "statsmodels>=0.10",
    "jupyterlab>=1.1",
    "matplotlib>=3.0",
    "dataclasses",
    "ipykernel",
    "jupyter-contrib-nbextensions",
]
EXTRAS_REQUIRE = {
    "tests": ["hypothesis[numpy]", "pytest", "tox", "pre-commit", "coverage"],
    "pypi": ["twine"],
    "debug": ["pdbpp"],
    "extra_runtime_libs": ["holidays>=0.9"],
}
EXTRAS_REQUIRE["dev"] = (
    EXTRAS_REQUIRE["tests"]
    + EXTRAS_REQUIRE["pypi"]  # NOQA
    + EXTRAS_REQUIRE["debug"]  # NOQA
    + EXTRAS_REQUIRE["extra_runtime_libs"]  # NOQA
)
EXTRAS_REQUIRE["tests_and_extra_runtime_libs"] = (
    EXTRAS_REQUIRE["tests"] + EXTRAS_REQUIRE["extra_runtime_libs"]  # NOQA
)


setuptools.setup(
    name="ts-eval",
    version="0.2.0",
    author="Vladimir Shulyak",
    author_email="vladimir@shulyak.net",
    description="Time Series analysis and evaluation tools",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/vshulyak/ts-eval",
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
