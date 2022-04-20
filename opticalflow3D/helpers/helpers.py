import skimage.io
import numpy as np


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
