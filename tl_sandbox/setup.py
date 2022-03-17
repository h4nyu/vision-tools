from setuptools import find_packages, setup

setup(
    name="tl_sandbox",
    version="0.1.0",
    install_requires=[
        "timm",
        "pytorch-metric-learning",
        "doit",
        "pytorch-lightning"
    ],
    packages=find_packages(),
    package_data={"tl_sandbox": ["py.typed"]},
)
