from setuptools import find_packages, setup

setup(
    name="hwad_bench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["doit", "timm", "pytorch-metric-learning", "gitpython"],
    package_data={"cots_bench": ["py.typed"]},
)
