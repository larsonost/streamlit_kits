from setuptools import find_packages, setup

setup(
    name="football-kits",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "ipython>=8.17.2",
        "ipywidgets>=8.1.1",
        "jupyter>=1.0.0",
        "matplotlib>=3.8.0",
        "nbstripout>=0.7.1",
        "numpy>=1.26.1",
        "openpyxl>=3.1.2",
        "pandas",
        "torch",
        "tqdm",
        "transformers",
        "facebook_scraper",
        "lxml_html_clean",
        "streamlit",
        "requests",
    ],
)
