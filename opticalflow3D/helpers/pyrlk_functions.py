import math
import typing
from math import sqrt, cos, acos, pi

import cupy as cp
import cupyx
from cupyx.scipy import ndimage
from numba import cuda, float32

from opticalflow3D.helpers.helpers import imresize_3d, gaussian_pyramid_3d


##################################
# Numba cuda kernels
##################################
@cuda.jit(device=True)
def add(A, reg):
    """ Add the regularization matrix to the A matrix

    Args:
        A (cuda array): A matrix
        reg (cuda array): Regularization matrix. Is equivalent to lambda * identity matrix

    Returns:
        A (cuda array): sum of the A and regularization matrix
    """
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            A[i, j] = A[i, j] + reg[i, j]
    return A


@cuda.jit(device=True)
def cofactor(matrix, _cofactor, p: int, q: int):
    """ Calculate the cofactor matrix

    Args:
        matrix (cuda array): Matrix to calculate determinant from
        _cofactor (cuda array): Output cofactor matrix
        p (int): Column index
        q (int): Row Index

    Returns:
        None
    """
    # q is col, p is row
    i, j = 0, 0
    for col in range(0, matrix.shape[1]):
        for row in range(0, matrix.shape[0]):
            if row != p and col != q:
                _cofactor[i, j] = matrix[row, col]
                j = j + 1
                if j == matrix.shape[1] - 1:
                    j = 0
                    i = i + 1


@cuda.jit(device=True)
def determinant_2x2(matrix):
    """ Calculate determinant for a 2×2

    Args:
        matrix (cuda array): Matrix to calculate determinant from

    Returns:
        determinant of the 2×2 matrix
    """
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]


@cuda.jit(device=True)
def determinant_3x3(matrix):
    """ Calculate determinant for a 3×3

    Args:
        matrix (cuda array): Matrix to calculate determinant from

    Returns:
        determinant of the 3×3 matrix
    """
    return matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]) - \
           matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[2, 0] * matrix[1, 2]) + \
           matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[2, 0] * matrix[1, 1])


@cuda.jit(device=True)
def inverse(A, inv):
    """ Calculates inverse of the 3x3 square matrix

    The inverse of the matrix is calculated as the adjoint of the matrix divided by its determinant

    Args:
        A (cuda array): Matrix to calculate the inverse from
        inv (cuda array): Output inverse matrix

    Returns:
        None
    """
    temp_2x2 = cuda.local.array(shape=(2, 2), dtype=float32)

    det = determinant_3x3(A)

    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            cofactor(A, temp_2x2, i, j)
            sign = 1. if ((i + j) % 2 == 0) else -1.
            inv[j, i] = sign * determinant_2x2(temp_2x2) / det  # perform transpose as well


@cuda.jit(device=True)
def matrix_mul(A, B, out):
    """ Calculates multiplication of two matrices

    Args:
        A (cuda array): A matrix
        B (cuda array): B Matrix
        out (cuda array): Output matrix

    Returns:
        None
    """
    for i in range(0, A.shape[0]):
        out[i] = 0
        for k in range(0, B.shape[0]):
            out[i] = out[i] + A[i, k] * B[k]


@cuda.jit(device=True)
def transpose(A, out):
    """ Transpose matrix

    Args:
        A (cuda array): A matrix
        out (cuda array): Output matrix

    Returns:
        None
    """
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            out[j, i] = A[i, j]
    return out


@cuda.jit(device=True)
def calculate_B(A, B, p, q):
    """ Calculate the B matrix

    Args:
        A (cuda array): 3×3 A matrix
        B (cuda array): Output B matrix
        p (int): Column index
        q (int): Row Index

    Returns:
        None
    """
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            if i == j:
                eye = 1
            else:
                eye = 0
            B[i, j] = (1 / p) * (A[i, j] - q * eye)
    return B


@cuda.jit(device=True)
def eig_value(A, B):
    """ Calculates the eigen value

    From https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices

    Args:
        A (cuda array): 3×3 A matrix
        B (cuda array): B matrix

    Returns:
        Smallest eigenvalue
    """
    p1 = A[0, 1] * A[0, 1] + A[0, 2] * A[0, 2] + A[1, 2] * A[1, 2]

    q = (A[0, 0] + A[1, 1] + A[2, 2]) / 3  # trace(A) is the sum of all diagonal values
    p2 = (A[0, 0] - q) * (A[0, 0] - q) + (A[1, 1] - q) * (A[1, 1] - q) + (A[2, 2] - q) * (A[2, 2] - q) + 2 * p1
    p = sqrt(p2 / 6)
    r = determinant_3x3(calculate_B(A, B, p, q)) / 2

    # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
    # but computation error can leave it slightly outside this range.
    if (r <= -1):
        phi = pi / 3
    elif (r >= 1):
        phi = 0
    else:
        phi = acos(r) / 3

    return q + 2 * p * cos(phi + (2 * pi / 3))


##################################
# 3D Lucas Kanade Functions
##################################
def calculate_derivatives(image):
    """ Calculates the derivative of the image using predefined kernels

    The smoothing kernel is [0.036, 0.249, 0.437, 0.249, 0.036]
    The differentiation kernel is [-0.108, -0.283, 0, 0.283, 0.108]

    Args:
        image (cuda array): Image to calculate derivatives from

    Returns:
        Ix (cuda array): Image derivative in x direction
        Iy (cuda array): Image derivative in y direction
        Iz (cuda array): Image derivative in z direction
    """
    p5 = cp.array([0.036, 0.249, 0.437, 0.249, 0.036])
    d5 = cp.array([-0.108, -0.283, 0, 0.283, 0.108])

    # calculate Ix
    Ix = cupyx.scipy.ndimage.convolve(image, cp.reshape(p5, (1, 5, 1)), mode="reflect")
    Ix = cupyx.scipy.ndimage.convolve(Ix, cp.reshape(p5, (5, 1, 1)), mode="reflect")
    Ix = cupyx.scipy.ndimage.convolve(Ix, cp.reshape(d5, (1, 1, 5)), mode="reflect")

    # calculate Iy(1)
    Iy = cupyx.scipy.ndimage.convolve(image, cp.reshape(p5, (1, 1, 5)), mode="reflect")

    # calculate Iz
    Iz = cupyx.scipy.ndimage.convolve(Iy, cp.reshape(p5, (1, 5, 1)), mode="reflect")
    Iz = cupyx.scipy.ndimage.convolve(Iz, cp.reshape(d5, (5, 1, 1)), mode="reflect")

    # calculate Iy(2)
    Iy = cupyx.scipy.ndimage.convolve(Iy, cp.reshape(p5, (5, 1, 1)), mode="reflect")
    Iy = cupyx.scipy.ndimage.convolve(Iy, cp.reshape(d5, (1, 5, 1)), mode="reflect")

    return Ix, Iy, Iz


@cuda.jit
def calculate_difference(image1, image2, vx, vy, vz, It):
    """ Calculates the difference in image intensity across the time frames

    Args:
        image1 (cuda array): First image in the sequence
        image2 (cuda array): Second image in the sequence
        vx (cuda array): Displacement in x direction
        vy (cuda array): Displacement in y direction
        vz (cuda array): Displacement in z direction
        It (cuda array): Output Image derivative in t direction

    Returns:
        None
    """
    z, y, x = cuda.grid(3)
    depth, length, width = vx.shape

    if z < depth and y < length and x < width:
        dx = vx[z, y, x]
        dy = vy[z, y, x]
        dz = vz[z, y, x]

        fx = x + dx
        fy = y + dy
        fz = z + dz

        x1 = int(math.floor(fx))
        y1 = int(math.floor(fy))
        z1 = int(math.floor(fz))

        fx -= x1
        fy -= y1
        fz -= z1

        if 0 <= x1 and 0 <= y1 and 0 <= z1 and x1 < (width - 1) and y1 < (length - 1) and z1 < (depth - 1):
            a000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz)
            a001 = fx * (1.0 - fy) * (1.0 - fz)
            a010 = (1.0 - fx) * fy * (1.0 - fz)
            a100 = (1.0 - fx) * (1.0 - fy) * fz
            a011 = fx * fy * (1.0 - fz)
            a101 = fx * (1.0 - fy) * fz
            a110 = (1.0 - fx) * fy * fz
            a111 = fx * fy * fz

            j = a000 * image2[z1, y1, x1] + \
                a001 * image2[z1, y1, x1 + 1] + \
                a010 * image2[z1, y1 + 1, x1] + \
                a100 * image2[z1 + 1, y1, x1] + \
                a011 * image2[z1, y1 + 1, x1 + 1] + \
                a101 * image2[z1 + 1, y1, x1 + 1] + \
                a110 * image2[z1 + 1, y1 + 1, x1] + \
                a111 * image2[z1 + 1, y1 + 1, x1 + 1]

            It[z, y, x] = image1[z, y, x] - j


def calculate_gradients(Ix, Iy, Iz, filter_fn):
    """ Calculates the component of the A^T W^2 A matrix

    Args:
        Ix (cuda array): Image derivative in x direction
        Iy (cuda array): Image derivative in y direction
        Iz (cuda array): Image derivative in z direction
        filter_fn: Function to determine the window size of the neighbourhood as well as the weights of the window

    Returns:
        Ix2 (cuda array): Image derivative in x*x direction
        IxIy (cuda array): Image derivative in x*y direction
        IxIz (cuda array): Image derivative in x*z direction
        Iy2 (cuda array): Image derivative in y*y direction
        IyIz (cuda array): Image derivative in y*z direction
        Iz2 (cuda array): Image derivative in z*z direction
    """
    Ix2 = filter_fn(cp.multiply(Ix, Ix, dtype=cp.float32))
    IxIy = filter_fn(cp.multiply(Ix, Iy, dtype=cp.float32))
    IxIz = filter_fn(cp.multiply(Ix, Iz, dtype=cp.float32))
    Iy2 = filter_fn(cp.multiply(Iy, Iy, dtype=cp.float32))
    IyIz = filter_fn(cp.multiply(Iy, Iz, dtype=cp.float32))
    Iz2 = filter_fn(cp.multiply(Iz, Iz, dtype=cp.float32))

    return Ix2, IxIy, IxIz, Iy2, IyIz, Iz2


def calculate_mismatch(Ix, Iy, Iz, It, filter_fn):
    """ Calculates the image mismatch b vector

    Args:
        Ix (cuda array): Image derivative in x direction
        Iy (cuda array): Image derivative in y direction
        Iz (cuda array): Image derivative in z direction
        It (cuda array): Image derivative in t direction
        filter_fn: Function to determine the window size of the neighbourhood as well as the weights of the window

    Returns:
        IxIt (cuda array): Image derivative in x*t direction
        IyIt (cuda array): Image derivative in y*t direction
        IzIt (cuda array): Image derivative in z*t direction
    """
    IxIt = filter_fn(cp.multiply(Ix, It, dtype=cp.float32))
    IyIt = filter_fn(cp.multiply(Iy, It, dtype=cp.float32))
    IzIt = filter_fn(cp.multiply(Iz, It, dtype=cp.float32))

    return IxIt, IyIt, IzIt


@cuda.jit
def calculate_vector(vx, vy, vz,
                     Ix2, IxIy, IxIz, Iy2, IyIz, Iz2, IxIt, IyIt, IzIt,
                     reg, threshold):
    """ Update the displacement field using the calculated image gradients

    Update is performed in place.

    Args:
        vx (cuda array): Displacement in x direction
        vy (cuda array): Displacement in y direction
        vz (cuda array): Displacement in z direction
        Ix2 (cuda array): Image derivative in x*x direction
        IxIy (cuda array): Image derivative in x*y direction
        IxIz (cuda array): Image derivative in x*z direction
        Iy2 (cuda array): Image derivative in y*y direction
        IyIz (cuda array): Image derivative in y*z direction
        Iz2 (cuda array): Image derivative in z*z direction
        IxIt (cuda array): Image derivative in x*t direction
        IyIt (cuda array): Image derivative in y*t direction
        IzIt (cuda array): Image derivative in z*t direction
        reg (cuda array): Regularization matrix
        threshold (float): Eigenvalue threshold. The calculated displacement is only accepted if the smallest eigenvalue
            is greater than this value.
    Returns:
        None
    """
    z, y, x = cuda.grid(3)
    depth, length, width = vx.shape

    A = cuda.local.array(shape=(3, 3), dtype=float32)
    B = cuda.local.array(shape=(3,), dtype=float32)
    velocity = cuda.local.array(shape=(3,), dtype=float32)
    inv = cuda.local.array(shape=(3, 3), dtype=float32)
    temp = cuda.local.array(shape=(3, 3), dtype=float32)

    if z < depth and y < length and x < width:
        A[0, 0] = Ix2[z, y, x]
        A[0, 1] = IxIy[z, y, x]
        A[0, 2] = IxIz[z, y, x]
        A[1, 1] = Iy2[z, y, x]
        A[1, 2] = IyIz[z, y, x]
        A[2, 2] = Iz2[z, y, x]

        B[0] = -IxIt[z, y, x]
        B[1] = -IyIt[z, y, x]
        B[2] = -IzIt[z, y, x]

        A[1, 0] = A[0, 1]
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]

        if eig_value(A, temp) > threshold:
            inverse(add(A, reg), inv)

            matrix_mul(inv, B, velocity)

            vx[z, y, x] = vx[z, y, x] + velocity[0]
            vy[z, y, x] = vy[z, y, x] + velocity[1]
            vz[z, y, x] = vz[z, y, x] + velocity[2]


def pyrlk_3d(image1, image2, iters: int, num_levels: int,
             scale: float = 0.5,
             tau: float = 0.1, alpha: float = 0.1,
             filter_type: str = "gaussian", filter_size: int = 15,
             presmoothing: int = None, threadsperblock: typing.Tuple[int, int, int] = (8, 8, 8)):
    """ Implementation of Pyramidal Lucas Kanade for 3D images

    Args:
        image1 (cuda array): First image in the sequence
        image2 (cuda array): Second image in the sequence
        iters (int): number of iterations
        num_levels (int): number of pyramid levels
        scale (float): Scaling factor used to generate the pyramid levels. Defaults to 0.5
        tau (float): Threshold value to accept calculated displacement. Defaults to 0.1
        alpha (float): Regularization parameter. Defaults to 0.1
        filter_type (int): Defines the type of filter used to average the calculated matrices. Defaults to "box"
        filter_size (int): Size of the filter used to average the matrices. Defaults to 15
        presmoothing (int): Standard deviation used to perform Gaussian smoothing of the images. Defaults to None
        threadsperblock (typing.Tuple[int, int, int]): Defines the number of cuda threads. Defaults to (8, 8, 8)
        
    Returns:
        vx (cuda array): Displacement in x direction
        vy (cuda array): Displacement in y direction
        vz (cuda array): Displacement in z direction
    """
    image1 = cp.asarray(image1, dtype=cp.float32)
    image2 = cp.asarray(image2, dtype=cp.float32)

    if presmoothing is not None:
        image1 = cupyx.scipy.ndimage.gaussian_filter(image1, presmoothing)
        image2 = cupyx.scipy.ndimage.gaussian_filter(image2, presmoothing)

    # initialize variables
    reg = alpha ** 2 * cp.eye(3, dtype=cp.float32)

    assert filter_type.lower() in ["gaussian", "box"]
    if filter_type.lower() == "gaussian":
        def filter_fn(x):
            return cupyx.scipy.ndimage.gaussian_filter(x, filter_size / 2 * 0.3)
    elif filter_type.lower() == "box":
        def filter_fn(x):
            return cupyx.scipy.ndimage.uniform_filter(x, size=filter_size)

    # initialize gaussian pyramid
    gauss_pyramid_1 = {1: image1}
    gauss_pyramid_2 = {1: image2}
    true_scale_dict = {}
    for pyr_lvl in range(1, num_levels + 1):
        if pyr_lvl == 1:
            gauss_pyramid_1 = {pyr_lvl: image1}
            gauss_pyramid_2 = {pyr_lvl: image2}
        else:
            gauss_pyramid_1[pyr_lvl], true_scale_dict[pyr_lvl] = gaussian_pyramid_3d(gauss_pyramid_1[pyr_lvl - 1],
                                                                                     sigma=1, scale=scale)
            gauss_pyramid_2[pyr_lvl], _ = gaussian_pyramid_3d(gauss_pyramid_2[pyr_lvl - 1], sigma=1, scale=scale)

    # LK code
    for lvl in range(num_levels, 0, -1):
        #         print("Currently working on pyramid level: {}".format(lvl))
        lvl_image_1 = gauss_pyramid_1[lvl]
        lvl_image_2 = gauss_pyramid_2[lvl]

        if lvl == num_levels:
            # initialize velocities
            vx = cp.zeros(lvl_image_1.shape, dtype=cp.float32)
            vy = cp.zeros(lvl_image_1.shape, dtype=cp.float32)
            vz = cp.zeros(lvl_image_1.shape, dtype=cp.float32)
        else:
            vx = 1 / true_scale_dict[lvl + 1][2] * imresize_3d(vx, scale=true_scale_dict[lvl + 1])
            vy = 1 / true_scale_dict[lvl + 1][1] * imresize_3d(vy, scale=true_scale_dict[lvl + 1])
            vz = 1 / true_scale_dict[lvl + 1][0] * imresize_3d(vz, scale=true_scale_dict[lvl + 1])

        Ix, Iy, Iz = calculate_derivatives(lvl_image_1)
        cp.cuda.Stream.null.synchronize()

        Ix2, IxIy, IxIz, Iy2, IyIz, Iz2 = calculate_gradients(Ix, Iy, Iz, filter_fn)
        cp.cuda.Stream.null.synchronize()

        image_shape = lvl_image_1.shape
        blockspergrid_z = math.ceil(image_shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(image_shape[1] / threadsperblock[1])
        blockspergrid_x = math.ceil(image_shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_z, blockspergrid_y, blockspergrid_x)
        for _ in range(iters):
            It = cp.zeros(lvl_image_1.shape, dtype=cp.float32)
            calculate_difference[blockspergrid, threadsperblock](lvl_image_1, lvl_image_2, vx, vy, vz, It)
            cp.cuda.Stream.null.synchronize()

            IxIt, IyIt, IzIt = calculate_mismatch(Ix, Iy, Iz, It, filter_fn)
            cp.cuda.Stream.null.synchronize()

            calculate_vector[blockspergrid, threadsperblock](vx, vy, vz,
                                                             Ix2, IxIy, IxIz, Iy2, IyIz, Iz2, IxIt, IyIt, IzIt,
                                                             reg, tau)
            cp.cuda.Stream.null.synchronize()

    return vx, vy, vz
