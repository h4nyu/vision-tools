from setuptools import find_packages, setup

setup(
    name="coco_annotator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
    ],
    package_data={"coco_annotator": ["py.typed"]},
    extras_require={"dev": ["nanoid"]},
)
