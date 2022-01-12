from setuptools import setup

setup(
    name="vision_tools",
    version="0.1.5",
    install_requires=[
        "tqdm",
        "efficientnet_pytorch",
        "omegaconf",
        "torch@https://download.pytorch.org/whl/cu113/torch-1.10.0%2Bcu113-cp39-cp39-linux_x86_64.whl",
        "torchvision@https://download.pytorch.org/whl/cu113/torchvision-0.11.1%2Bcu113-cp39-cp39-linux_x86_64.whl",
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
            "pandas",
            "cytoolz",
            "torch-optimizer",
            "albumentations",
            "scikit-learn",
            "tensorboard",
            "codecov",
            "signate",
        ]
    },
)
