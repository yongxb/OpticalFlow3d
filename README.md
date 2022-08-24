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
## Validation of the 3D Farneback algorithm
### Accuracy of the algorithm on synthetic 3D fluorescent bead images
The Farneback two frame motion estimation algorithm was validated against the L5 SEM Challenge Sample 14 dataset [[1](#references)].
The RMSE for the x component of the displacement field was 0.0112. The figure below shows the accuracy of the x component. 
The solid blue line represents the mean magnitude of the x component for each x position while the shaded 
region/thin blue shows the standard deviation of the x component magnitudes.

<img src="https://gitlab.com/xianbin.yong13/opticalflow3d/-/raw/master/docs/images/accuracy.png" width="700"/>

### Execution time
The time taken to compute the displacements for a 2048×192×192 voxel image was **2.41s ± 21.4ms**. The computation was 
done on a server running a Quadro RTX 6000 GPU and dual Intel(R) Xeon(R) Gold 5217 CPUs.

For comparison, the FFT-based DVC algorithm took 1980.04s while ALDVC took 9335.2s [[1](#references)]. A few caveats 
have to be mentioned as both of these algorithms were running on the CPU only and ALDVC does calculate other 
parameters as well. 

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

Please also ensure that a compatible cudatoolkit version is installed alongside the cupy package. As the default 
cupy package (cupy-cuda113) relies on cudatoolkit v11.3.1, this needs to be installed as well using conda:

```
conda install -c conda-forge cudatoolkit==11.3.1
```

### Examples
Examples can be found in the examples folder

***
## How to cite
Huang, C. K., Yong, X., She, D. T., & Lim, C. T. (2022). Surface curvature and basal hydraulic stress induce spatial bias in cell extrusion. bioRxiv.

## Related projects
1. [3D Traction Force Microscopy](https://gitlab.com/xianbin.yong13/3dtfm)

***
## References
[1] Yang, J., Hazlett, L., Landauer, A. K., & Franck, C. (2020). Augmented Lagrangian Digital Volume Correlation (ALDVC). Experimental Mechanics, 60(9), 1205-1223.