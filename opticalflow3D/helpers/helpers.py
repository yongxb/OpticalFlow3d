import skimage.io
import numpy as np
import math
import cupy as cp
from cupyx.scipy import ndimage
import cupyx
from numba import njit, prange
import scipy.ndimage


def gaussian_kernel_1d(sigma, radius=None):
    if radius is None:
        radius = math.ceil(2 * sigma)

    output_kernel = np.mgrid[-radius:radius + 1]
    output_kernel = np.exp((-(1 / 2) * (output_kernel ** 2)) / (sigma ** 2))
    output_kernel = output_kernel / np.sum(output_kernel)

    return output_kernel


def gaussian_pyramid_3d(image, sigma=1, scale=0.5):
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
    image_resized = cp.empty((int(round(shape[0] * scale)),
                              int(round(shape[1] * scale)),
                              int(round(shape[2] * scale))), dtype=cp.float32)
    ndimage.zoom(image, (scale, scale, scale), output=image_resized, mode="reflect")

    return image_resized, true_scale


def imresize_3d(image, scale=(0.5, 0.5, 0.5)):
    image = ndimage.zoom(image, (1 / scale[0], 1 / scale[1], 1 / scale[2]))

    return image


def get_positions(start_point, total_vol, vol, shape, overlap, n):
    q, r = divmod(total_vol[n], vol[n] - overlap[n])
    position = []
    valid_vol = []
    valid_position = []

    count = q + (r != 0)
    for i in range(count):
        if i == 0:
            start = start_point[n] - overlap[n] // 2
            valid_start = 0
        else:
            start = end - overlap[n]
            valid_start = valid_end
        end = start + vol[n]

        _start = max(start, 0)
        start_diff = start - _start
        start_valid = overlap[n] // 2 + start_diff

        _end = min((end, shape[n], start_point[n] + total_vol[n] + overlap[n] // 2))
        valid_end = min((end - overlap[n] // 2 - start_point[n], total_vol[n]))

        end_valid = valid_end - valid_start + start_valid

        position.append((_start, _end))
        valid_position.append((valid_start, valid_end))
        valid_vol.append((start_valid, end_valid))

    return position, valid_position, valid_vol


def load_image(arg, axis=0):
    if type(arg) == list:
        images = []
        for file in arg:
            images.append(skimage.io.imread(file))
        return np.concatenate(images, axis=axis)
    else:
        return skimage.io.imread(arg)


def realign_image(image, z_reverse=True, z_start=None, z_end=None,
                  y_start=None, y_end=None, x_start=None, x_end=None):
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


def save_flow(path, vz, vy, vx):
    np.savez(path, vx=vx, vy=vy, vz=vz)


def save_error(path, error):
    np.savez(path, error=error)


def generate_inverse_image(image, vx, vy, vz, use_gpu=True):
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