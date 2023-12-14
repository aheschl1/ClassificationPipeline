from setuptools import find_packages
from setuptools import setup

setup(
    name='classeg',
    version='0.1.0',
    install_requires=[
        'numpy',
        'tqdm',
        'click',
        'SimpleITK',
        'scipy'
    ],
    packages=find_packages(),
    url='https://github.com/aheschl1/ClassificationPipeline',
    author='Andrew Heschl',
    author_email='ajheschl@gmail.com',
    description='Classification Pipeline'
)