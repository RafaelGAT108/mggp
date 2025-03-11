import setuptools
from setuptools import setup

# read description
with open("README.md", "r") as fh:
    long_description = fh.read()

req_file = 'requirements.txt'

# read requirements
with open(req_file) as f:
    required = f.read().splitlines()

setup(
    name="mggp",
    version="2.1",
    author="Henrique Castro",
    author_email="henriquec.castro@outlook.com",
    install_requires=required,
    packages=setuptools.find_packages(),
)

