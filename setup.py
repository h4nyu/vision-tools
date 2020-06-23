from setuptools import setup, find_packages

setup(
    name="app",
    version="0.0.0",
    description="TODO",
    author="Xinyuan Yao",
    author_email="yao.ntno@google.com",
    license="TODO",
    packages=find_packages(),
    package_data={"object_detection": ["py.typed"],},
    install_requires=[
        "pandas",
        "scikit-learn",
        "cytoolz",
        "matplotlib",
        "torch",
        "kaggle",
        "tqdm",
        "scikit-image",
        "torchvision",
        "albumentations",
        "efficientnet_pytorch",
        "typing_extensions",
    ],
    extras_require={"dev": ["mypy", "pytest", "black",]},
    entry_points={"console_scripts": ["app=app.cmd:main"],},
)
