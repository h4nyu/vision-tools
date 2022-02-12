from setuptools import setup, find_packages

setup(
    name="coco_annotator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
    ],
    package_data={"coco_annotator": ["py.typed"]},
)
