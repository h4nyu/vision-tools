from setuptools import setup

setup(
    name="vision_tools",
    version="0.1.5",
    install_requires=[
        "tqdm",
        "pyyaml",
        "efficientnet_pytorch",
        "hydra-core",
    ],
    packages=["vision_tools"],
    package_data={"vision_tools": ["py.typed"]},
    extras_require={
        "dev": [
            "pytest",
            "black",
            "pytest-cov",
            "pytest-benchmark",
            "pytest-mypy",
            "mypy",
            "kaggle",
            "signate",
            "pandas",
            "ensemble-boxes",
            "cytoolz",
            "albumentations",
            "scikit-learn",
            "tensorboard",
            "codecov",
            "signate",
            "types-PyYAML",
            "notebook",
            "optuna",
            "hydra-core",
        ]
    },
)
