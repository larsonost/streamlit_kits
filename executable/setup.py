from setuptools import find_packages, setup

setup(
    name="executable",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "torch",
        "tqdm",
        "transformers",
        "pyinstaller",
    ],
)
