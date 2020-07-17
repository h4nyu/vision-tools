from setuptools import setup

setup(
    name="effdet",
    version="0.0.0",
    packages=["effdet"],
    entry_points={"console_scripts": ["effdet = effdet.cli:main"]},
)
