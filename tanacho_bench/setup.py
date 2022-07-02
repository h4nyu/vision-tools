from setuptools import find_packages, setup

setup(
    name="tanacho_bench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "doit",
        "timm",
        "pytorch-metric-learning",
        "pytorch-lightning",
        "catalyst[cv]",
        "torchmetrics",
        "signate",
        "iterative-stratification",
    ],
    package_data={"cots_bench": ["py.typed"]},
)
