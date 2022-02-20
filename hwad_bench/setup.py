from setuptools import setup, find_packages

setup(
    name="hwad_bench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "doit",
        "timm",
    ],
    package_data={"cots_bench": ["py.typed"]},
)
