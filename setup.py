from setuptools import setup

setup(
    name="vision_tools",
    version="0.1.5",
    install_requires=[
        "tqdm",
        "efficientnet_pytorch",
        "torchvision",
        "omegaconf",
        "torch",
    ],
    packages=["vision_tools"],
    package_data={"vision_tools": ["py.typed"]},
    extras_require={
        "dev": [
            "pytest",
            "black",
            "pytest-cov",
            "pytest-benchmark",
            "mypy",
            "kaggle",
            "pandas",
            "cytoolz",
            "torch-optimizer",
            "albumentations",
            "scikit-learn",
        ]
    },
)
