import typing
from typing import Tuple, List, Union, Any

import numpy.typing as npt
import skimage.io
import numpy as np
import math
import cupy as cp
from cupyx.scipy import ndimage
import cupyx
from numba import njit, prange
import scipy.ndimage


def gaussian_kernel_1d(sigma: float, radius: int = None) -> np.ArrayLike:
    """ Generates a 1d kernel that can be used to perform Gaussian smoothing

    Args:
        sigma (float): Standard deviation of the Gaussian kernel
        radius (int): Size of the Gaussian kernel. Final size is equal to 2*radius+1. Defaults to None

    Returns:
        output_kernel (ndarray): 1d Guassian kernel
    """
    if radius is None:
        radius = math.ceil(2 * sigma)

    output_kernel = np.mgrid[-radius:radius + 1]
    output_kernel = np.exp((-(1 / 2) * (output_kernel ** 2)) / (sigma ** 2))
    output_kernel = output_kernel / np.sum(output_kernel)

    return output_kernel


def gaussian_pyramid_3d(image, sigma: float = 1, scale: float = 0.5) -> typing.Tuple[np.ndarray, npt.ArrayLike]:
    """ Downscales the image for use in a Gaussian pyramid

    Args:
        image (cuda array): Image to generate pyramids from
        sigma (float): Standard deviation of the Gaussian kernel used for downscaling. Defaults to 1
        scale (float): Scale factor used to downscale the image. Defaults to 0.5.

    Returns:
        resized_image (ndarray): Downscaled image
        true_scale (npt.ArrayLike[float, float, float]): Actual scaling factor used. This differs slightly from the input
            scaling factor in cases when the factor used causes the size of the image to not be an integer.
    """
    kernel = cp.asarray(gaussian_kernel_1d(sigma), dtype=cp.float32)
    radius = math.ceil(2 * sigma)

    # gaussian smoothing
    image = cupyx.scipy.ndimage.convolve(image, cp.reshape(kernel, (2 * radius + 1, 1, 1)), mode="reflect")
    image = cupyx.scipy.ndimage.convolve(image, cp.reshape(kernel, (1, 2 * radius + 1, 1)), mode="reflect")
    image = cupyx.scipy.ndimage.convolve(image, cp.reshape(kernel, (1, 1, 2 * radius + 1)), mode="reflect")

    shape = image.shape
    true_scale = [int(round(shape[0] * scale)) / shape[0],
                  int(round(shape[1] * scale)) / shape[1],
                  int(round(shape[2] * scale)) / shape[2]]
    resized_image = cp.empty((int(round(shape[0] * scale)),
                              int(round(shape[1] * scale)),
                              int(round(shape[2] * scale))), dtype=cp.float32)
    ndimage.zoom(image, (scale, scale, scale), output=resized_image, mode="reflect")

    return resized_image, true_scale


def imresize_3d(image, scale: npt.ArrayLike[float, float, float] = (0.5, 0.5, 0.5)) -> np.ndarray:
    """ Upscales the image by the specified factor

    Args:
        image (cuda array): image to generate pyramids from
        scale (npt.ArrayLike[float, float, float]): Scale factor used to downscale the image. Actual factor used is 1/scale.
            Defaults to (0.5, 0.5, 0.5).

    Returns:
        image (ndarray): Upscaled image
    """
    image = ndimage.zoom(image, (1 / scale[0], 1 / scale[1], 1 / scale[2]))

    return image


def get_positions(start_point: typing.Tuple[int, int, int],
                  total_vol: typing.Tuple[int, int, int],
                  vol: typing.Tuple[int, int, int],
                  shape: typing.Tuple[int, int, int],
                  overlap: typing.Tuple[int, int, int],
                  axis: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """ Calculates the starting positions of the subvolumes

    This breaks a large volume that is too large to be analysed in one go into smaller subvolumes.
    If any one of the values in overlap is greater than 0, the subvolumes will include regions outside the region of
    interest. This helps to minimize any edge effects. When the subvolumes are merged back these extra regions will
    be removed.

    Args:
        start_point (typing.Tuple[int, int, int]): starting position of the region of interest in the image volume
        total_vol (typing.Tuple[int, int, int]): total size of the region of interest
        vol (typing.Tuple[int, int, int]): maximum volume size that can be analysed at one go
        shape (typing.Tuple[int, int, int]): size of the image volume
        overlap (typing.Tuple[int, int, int]): amount of overlap between adjacent subvolumes
        axis (int): axis to calculate the positions from

    Returns:
        position (List[Tuple[int, int]]): starting and ending position along the specified axis to generate the subvolume
        valid_position (List[Tuple[int, int]]): starting and ending position of the final displacement field that should be merged
        valid_vol (List[Tuple[int, int]]): starting and ending position of the subvolume displacement field that should be merged
    """
    q, r = divmod(total_vol[axis], vol[axis] - overlap[axis])
    position = []
    valid_vol = []
    valid_position = []

    count = q + (r != 0)
    for i in range(count):
        if i == 0:
            start = start_point[axis] - overlap[axis] // 2
            valid_start = 0
        else:
            start = end - overlap[axis]
            valid_start = valid_end
        end = start + vol[axis]

        _start = max(start, 0)
        start_diff = start - _start
        start_valid = overlap[axis] // 2 + start_diff

        _end = min((end, shape[axis], start_point[axis] + total_vol[axis] + overlap[axis] // 2))
        valid_end = min((end - overlap[axis] // 2 - start_point[axis], total_vol[axis]))

        end_valid = valid_end - valid_start + start_valid

        position.append((_start, _end))
        valid_position.append((valid_start, valid_end))
        valid_vol.append((start_valid, end_valid))

    return position, valid_position, valid_vol


def load_image(path: typing.Union[str, list], axis: int = 0) -> np.ndarray:
    """ Loads the image

    If the input is a list of file paths, each file path is loaded and the images are concatenated into a single image.
    If the input is a file path, the file path is loaded and the image is returned.

    Args:
        path (typing.Union[list, str]): List or single file path that contains the images to load.
        axis (int): Axis used to concatenate the images. Only used if path is a list of file paths.

    Returns:
        image (ndarray): Loaded image
    """
    if type(path) == list:
        images = []
        for file in path:
            images.append(skimage.io.imread(file))
        return np.concatenate(images, axis=axis)
    else:
        return skimage.io.imread(path)


def crop_image(image: np.ndarray,
               z_reverse: bool = True,
               z_start: typing.Union[int, None] = None, z_end: typing.Union[int, None] = None,
               y_start: typing.Union[int, None] = None, y_end: typing.Union[int, None] = None,
               x_start: typing.Union[int, None] = None, x_end: typing.Union[int, None] = None) -> np.ndarray:
    """ Crops the image to facilitate realignment

    Args:
        image (np.ndarray): Image to crop
        z_reverse (bool): Option to determine if the slices in the z axis should be reversed. Defaults to True.
        z_start (typing.Union[int, None]): Amount to crop from the start of the z axis. Defaults to None.
        z_end (typing.Union[int, None]): Amount to crop from the end of the z axis. Defaults to None.
        y_start (typing.Union[int, None]): Amount to crop from the start of the y axis. Defaults to None.
        y_end (typing.Union[int, None]): Amount to crop from the end of the y axis. Defaults to None.
        x_start (typing.Union[int, None]): Amount to crop from the start of the x axis. Defaults to None.
        x_end (typing.Union[int, None]): Amount to crop from the end of the x axis. Defaults to None.

    Returns:
        _image (np.ndarray): cropped image
    """
    _z_start = z_start if z_start else 0
    _z_end = z_end if z_end else image.shape[0]

    _y_start = y_start if y_start else 0
    _y_end = y_end if y_end else image.shape[1]

    _x_start = x_start if x_start else 0
    _x_end = x_end if x_end else image.shape[2]

    _image = image[_z_start:_z_end, _y_start:_y_end, _x_start:_x_end]

    if z_reverse:
        _image = -image[::-1, ...]

    return _image


def save_displacements(path: str, vz: np.ndarray, vy: np.ndarray, vx: np.ndarray) -> None:
    """ Saves the displacement as the binary uncompressed .npz format

    Args:
        path (str): File path to save the displacements. The extension type should be *.npz
        vx (np.ndarray): Array containing the displacements in the x direction
        vy (np.ndarray): Array containing the displacements in the y direction
        vz (np.ndarray): Array containing the displacements in the z direction

    Returns:
        None
    """
    np.savez(path, vx=vx, vy=vy, vz=vz)


def save_confidence(path: str, confidence: np.ndarray) -> None:
    """ Saves the displacement as the binary uncompressed .npz format

    Args:
        path (str): File path to save the displacements. The extension type should be *.npz
        confidence (np.ndarray): Array containing the calculated confidence of the Farneback algorithm

    Returns:
        None
    """
    np.savez(path, confidence=confidence)


def generate_inverse_image(image, vx, vy, vz, use_gpu: bool = True) -> np.ndarray:
    """ Uses the displacements to transform the image

    This transformed image can then be overlaid over the actual image to verify the quality of the displacements.

    Args:
        image (np.ndarray): File path to save the displacements. The extension type should be *.npz
        vx (np.ndarray): Array containing the displacements in the x direction
        vy (np.ndarray): Array containing the displacements in the y direction
        vz (np.ndarray): Array containing the displacements in the z direction
        use_gpu (bool): Option to run some part of the procedure on the gpu

    Returns:
        inverse_image (np.ndarray): transformed image using the displacemennt field
    """
    # image should be the first image that is used for the optical flow calculations
    map_x_inverse, map_y_inverse, map_z_inverse, distance_total = inverse(vx, vy, vz)

    map_x_inverse = map_x_inverse / (distance_total + 1e-12)
    map_y_inverse = map_y_inverse / (distance_total + 1e-12)
    map_z_inverse = map_z_inverse / (distance_total + 1e-12)

    if use_gpu:
        inverse_image_gpu = cupyx.scipy.ndimage.map_coordinates(cp.asarray(image),
                                                                cp.array([map_z_inverse, map_y_inverse, map_x_inverse]),
                                                                mode="mirror")
        inverse_image = inverse_image_gpu.get()
    else:
        inverse_image = scipy.ndimage.map_coordinates(image,
                                                      np.array([map_z_inverse, map_y_inverse, map_x_inverse]),
                                                      mode="mirror")

    return inverse_image


@njit(parallel=True)
def inverse(xmap, ymap, zmap, xmin=0, ymin=0, zmin=0, dist_threshold=1, eps=1e-12):
    shape = xmap.shape
    inverse_x = np.zeros_like(xmap)
    inverse_y = np.zeros_like(xmap)
    inverse_z = np.zeros_like(xmap)
    distance_total = np.zeros_like(xmap)

    for i in prange(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                idz = np.int32(np.round(i + zmap[i, j, k]))
                idy = np.int32(np.round(j + ymap[i, j, k]))
                idx = np.int32(np.round(k + xmap[i, j, k]))

                for zval in range(max(idz - dist_threshold, zmin), min(idz + dist_threshold, zmin + shape[0])):
                    for yval in range(max(idy - dist_threshold, ymin), min(idy + dist_threshold, ymin + shape[1])):
                        for xval in range(max(idx - dist_threshold, xmin),
                                          min(idx + dist_threshold, xmin + shape[2])):
                            distance = (zval - (i + zmap[i, j, k])) ** 2 + (yval - (j + ymap[i, j, k])) ** 2 + (
                                        xval - (k + xmap[i, j, k])) ** 2
                            inverse_distance = 1 / (distance + eps)

                            inverse_z[zval, yval, xval] += inverse_distance * i
                            inverse_y[zval, yval, xval] += inverse_distance * j
                            inverse_x[zval, yval, xval] += inverse_distance * k
                            distance_total[zval, yval, xval] += inverse_distance

    return inverse_x, inverse_y, inverse_z, distance_total