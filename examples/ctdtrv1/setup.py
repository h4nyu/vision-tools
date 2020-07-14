from setuptools import setup

setup(
    name="ctdtrv1",
    version="0.0.0",
    packages=["ctdtrv1"],
    entry_points={"console_scripts": ["ctdtrv1 = ctdtrv1.cli:main"]},
)
