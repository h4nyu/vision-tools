from setuptools import find_packages, setup

requires = open("./requirements.txt").read().splitlines()

setup(
    name="tanacho_bench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requires,
    package_data={"cots_bench": ["py.typed"]},
)
