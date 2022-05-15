from setuptools import setup, find_packages
from os import path
from pkg_resources import DistributionNotFound, get_distribution

with open("README.md", "r") as file:
    long_description = file.read()

_dir = path.dirname(__file__)


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


dependencies = [
    "numpy>=1.17.0",
    "numba>=0.47.0",
    "scikit-image>=0.17.1",
    "scipy>=1.6.3",
    "tqdm>=4.50.0"
]

if get_dist('cupy-cuda102') is None and get_dist('cupy-cuda110') is None and get_dist('cupy-cuda111') is None and \
        get_dist('cupy-cuda112') is None and get_dist('cupy-cuda113') is None and get_dist('cupy-cuda114') is None and \
        get_dist('cupy-cuda115') is None and get_dist('cupy-cuda116') is None:
    dependencies.append("cupy-cuda113>=10.0.0")

setup(
    name='opticalflow3d',
    version="0.2.0",
    description='GPU/CUDA optimized implementation of 3D optical flow algorithms such as Farneback two frame motion estimation and Lucas Kanade dense optical flow algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=dependencies,
    author='Xianbin Yong',
    author_email='xianbin.yong13@sps.nus.edu.sg',
    url='https://gitlab.com/xianbin.yong13/opticalflow3d',

    packages=find_packages(),
    license="GPLv3",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.7',
    project_urls={
        'Research group': 'https://ctlimlab.org/',
        'Source': 'https://gitlab.com/xianbin.yong13/opticalflow3d',
    },
)
