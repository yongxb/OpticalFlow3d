# opticalflow3d

GPU/CUDA optimized implementation of 3D optical flow algorithms such as Farneback two frame 
motion estimation and Lucas Kanade dense optical flow algorithms.

Please see the [related projects](#related-projects) section for the other components of this pipeline

***
## Overview
Being able to efficiently calculate the displacements between two imaging volumes will allow us to query the 
dynamics captured in the images. This would include 3D biological samples and 3D force micrsocopy. However, a CPU 
based implementation would be too time consuming. Thus, this repository was created to address this problem by 
providing GPU accelerated implementation of various optical flow algorithms. Speed is key here, and tricks such as 
separable convolutions are used.

Currently, two optical flow methods are implemented:
- Pyramidal Lucas Kanade dense optical flow algorithm
- Farneback two frame motion estimation algorithm

The following methods are also provided to help in the assessment of the vectors.
- forward mapping of the first image using the calculated vectors

***
## Usage
### Required packages
The following packages are required. Please ensure that they are installed using either pip or conda.
- numpy
- numba
- scikit-image
- scipy
- cupy
- tqdm

### Installation
This package is available via pip and can be installed using:
```
pip install opticalflow3d
```

### Examples
Examples can be found in the examples folder

***
## How to cite


## Related projects
