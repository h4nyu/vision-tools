from setuptools import find_packages, setup

setup(
    name="cots_bench",
    version="0.1.0",
    packages=find_packages(),
    package_data={"cots_bench": ["py.typed"]},
)
