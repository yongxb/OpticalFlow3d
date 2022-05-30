import typing

import cupy as cp
import numpy as np
from tqdm import tqdm

from .helpers.farneback_functions import farneback_3d
from .helpers.helpers import get_positions
from .helpers.pyrlk_functions import pyrlk_3d


class Farneback3D:
    """Farneback3D class used to instantiate the algorithm with its parameters.

    Args:
        iters (int): number of iterations. Defaults to 5
        num_levels (int): number of pyramid levels. Defaults to 5
        scale (float): Scaling factor used to generate the pyramid levels. Defaults to 0.5
        spatial_size (int): size of the support used in the calculation of the standard deviation of the Gaussian
            applicability. Defaults to 9.
        sigma_k (float): scaling factor used to calculate the standard deviation of the Gaussian applicability. The
            formula to calculate sigma is sigma_k*(spatial_size - 1). Defaults to 0.15.
        filter_type (str): Defines the type of filter used to average the calculated matrices. Defaults to "box"
        filter_size (int): Size of the filter used to average the matrices. Defaults to 21
        presmoothing (int): Standard deviation used to perform Gaussian smoothing of the images. Defaults to None
        device_id (int): Device id of the GPU. Defaults to 0
    """

    def __init__(self,
                 iters: int = 5,
                 num_levels: int = 5,
                 scale: float = 0.5,
                 spatial_size: int = 9,
                 sigma_k: float = 0.15,
                 filter_type: str = "box",
                 filter_size: int = 21,
                 presmoothing: int = None,
                 device_id: int = 0,
                 ):
        self.iters = iters
        self.num_levels = num_levels
        self.scale = scale
        self.spatial_size = spatial_size

        self.presmoothing = presmoothing
        self.sigma_k = sigma_k
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.device_id = device_id

    def calculate_flow(self, image1: np.ndarray, image2: np.ndarray,
                       start_point: typing.Tuple[int, int, int] = (0, 0, 0),
                       total_vol: typing.Tuple[int, int, int] = (-1, -1, -1),
                       sub_volume: typing.Tuple[int, int, int] = (256, 256, 256),
                       overlap: typing.Tuple[int, int, int] = (64, 64, 64),
                       threadsperblock: typing.Tuple[int, int, int] = (8, 8, 8),
                       ):
        """ Calculates the displacement across image1 and image2 using the 3D Farneback two frame algorithm

        Args:
            image1 (np.ndarray): first image
            image2 (np.ndarray): second image
            start_point (typing.Tuple[int, int, int]): starting position of the region of interest in the image volume.
                Defaults to (0, 0, 0)
            total_vol (typing.Tuple[int, int, int]): total size of the region of interest. Defaults to (-1, -1, -1)
            sub_volume (typing.Tuple[int, int, int]): maximum volume size that can be analysed at one go.
                Defaults to (256, 256, 256)
            overlap (typing.Tuple[int, int, int]): amount of overlap between adjacent subvolumes.
                Defaults to (64, 64, 64)
            threadsperblock (typing.Tuple[int, int, int]): Defines the number of cuda threads.
                Defaults to (8, 8, 8)

        Returns:
            output_vz (np.ndarray): array containing the displacements in the x direction
            output_vy (np.ndarray): array containing the displacements in the y direction
            output_vx (np.ndarray): array containing the displacements in the z direction
            output_confidence (np.ndarray): array containing the calculated confidence of the Farneback algorithm
        """
        if total_vol is None:
            total_vol = image1.shape - np.array(start_point)

        print("Running 3D Farneback optical flow with the following parameters:")
        print(
            f"Iters: {self.iters} | Levels: {self.num_levels} | Scale: {self.scale} | Kernel: {self.spatial_size} | Filter: {self.filter_type}-{self.filter_size} | Presmoothing: {self.presmoothing}",
            flush=True)

        output_vx = np.zeros(total_vol, dtype=np.float32)
        output_vy = np.zeros(total_vol, dtype=np.float32)
        output_vz = np.zeros(total_vol, dtype=np.float32)
        output_confidence = np.zeros(total_vol, dtype=np.float32)

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        if np.any(total_vol > sub_volume):

            shape = image1.shape
            z_position, z_valid_pos, z_valid = get_positions(start_point, total_vol, sub_volume, shape, overlap, 0)
            y_position, y_valid_pos, y_valid = get_positions(start_point, total_vol, sub_volume, shape, overlap, 1)
            x_position, x_valid_pos, x_valid = get_positions(start_point, total_vol, sub_volume, shape, overlap, 2)

            for z_i in range(len(z_position)):
                for y_i in range(len(y_position)):
                    for x_i in tqdm(range(len(x_position)),
                                    desc=f"Item: {z_i * len(y_position) + y_i + 1}/{len(z_position) * len(y_position)}"):
                        input_image_vol_1 = image1[z_position[z_i][0]:z_position[z_i][1],
                                                   y_position[y_i][0]:y_position[y_i][1],
                                                   x_position[x_i][0]:x_position[x_i][1]].astype(np.float32)
                        input_image_vol_2 = image2[z_position[z_i][0]:z_position[z_i][1],
                                                   y_position[y_i][0]:y_position[y_i][1],
                                                   x_position[x_i][0]:x_position[x_i][1]].astype(np.float32)

                        with cp.cuda.Device(self.device_id):
                            vx, vy, vz, confidence = farneback_3d(input_image_vol_1, input_image_vol_2, self.iters,
                                                                  self.num_levels,
                                                                  scale=self.scale, spatial_size=self.spatial_size,
                                                                  sigma_k=self.sigma_k, filter_type=self.filter_type,
                                                                  filter_size=self.filter_size,
                                                                  presmoothing=self.presmoothing,
                                                                  threadsperblock=threadsperblock)

                        cp.cuda.Stream.null.synchronize()

                        vx_cpu = vx.get()
                        vy_cpu = vy.get()
                        vz_cpu = vz.get()
                        confidence_cpu = confidence.get()

                        output_vx[z_valid_pos[z_i][0]: z_valid_pos[z_i][1],
                                  y_valid_pos[y_i][0]: y_valid_pos[y_i][1],
                                  x_valid_pos[x_i][0]: x_valid_pos[x_i][1]] = vx_cpu[z_valid[z_i][0]: z_valid[z_i][1],
                                                                                     y_valid[y_i][0]: y_valid[y_i][1],
                                                                                     x_valid[x_i][0]: x_valid[x_i][1]]
                        output_vy[z_valid_pos[z_i][0]: z_valid_pos[z_i][1],
                                  y_valid_pos[y_i][0]: y_valid_pos[y_i][1],
                                  x_valid_pos[x_i][0]: x_valid_pos[x_i][1]] = vy_cpu[z_valid[z_i][0]: z_valid[z_i][1],
                                                                                     y_valid[y_i][0]: y_valid[y_i][1],
                                                                                     x_valid[x_i][0]: x_valid[x_i][1]]
                        output_vz[z_valid_pos[z_i][0]: z_valid_pos[z_i][1],
                                  y_valid_pos[y_i][0]: y_valid_pos[y_i][1],
                                  x_valid_pos[x_i][0]: x_valid_pos[x_i][1]] = vz_cpu[z_valid[z_i][0]: z_valid[z_i][1],
                                                                                     y_valid[y_i][0]: y_valid[y_i][1],
                                                                                     x_valid[x_i][0]: x_valid[x_i][1]]

                        output_confidence[z_valid_pos[z_i][0]: z_valid_pos[z_i][1],
                                          y_valid_pos[y_i][0]: y_valid_pos[y_i][1],
                                          x_valid_pos[x_i][0]: x_valid_pos[x_i][1]] = confidence_cpu[z_valid[z_i][0]: z_valid[z_i][1],
                                                                                                     y_valid[y_i][0]: y_valid[y_i][1],
                                                                                                     x_valid[x_i][0]: x_valid[x_i][1]]

                        del vx, vy, vz, confidence
                        del vx_cpu, vy_cpu, vz_cpu, confidence_cpu

                        mempool.free_all_blocks()
                        pinned_mempool.free_all_blocks()
        else:
            with cp.cuda.Device(self.device_id):
                vx, vy, vz, confidence = farneback_3d(image1, image2,
                                                      iters=self.iters,
                                                      num_levels=self.num_levels,
                                                      scale=self.scale, spatial_size=self.spatial_size,
                                                      sigma_k=self.sigma_k,
                                                      filter_type=self.filter_type, filter_size=self.filter_size,
                                                      presmoothing=self.presmoothing,
                                                      threadsperblock=threadsperblock)
            cp.cuda.Stream.null.synchronize()

            output_vx = vx.get()
            output_vy = vy.get()
            output_vz = vz.get()
            output_confidence = confidence.get()

        return output_vz, output_vy, output_vx, output_confidence


class PyrLK3D:
    """Farneback3D class used to instantiate the algorithm with its parameters.

    Args:
        iters (int): number of iterations. Defaults to 15
        num_levels (int): number of pyramid levels. Defaults to 5
        scale (float): Scaling factor used to generate the pyramid levels. Defaults to 0.5
        tau (float): Threshold value to accept calculated displacement. Defaults to 0.1
        alpha (float): Regularization parameter. Defaults to 0.1
        filter_type (str): Defines the type of filter used to average the calculated matrices. Defaults to "gaussian"
        filter_size (int): Size of the filter used to average the matrices. Defaults to 21
        presmoothing (int): Standard deviation used to perform Gaussian smoothing of the images. Defaults to None
        device_id (int): Device id of the GPU. Defaults to 0
    """

    def __init__(self,
                 iters: int = 15,
                 num_levels: int = 5,
                 scale: float = 0.5,
                 tau: float = 0.1,
                 alpha: float = 0.1,
                 filter_type: str = "gaussian",
                 filter_size: int = 21,
                 presmoothing: int = None,
                 device_id: int = 0,
                 ):
        self.iters = iters
        self.num_levels = num_levels
        self.scale = scale

        self.tau = tau
        self.alpha = alpha
        self.presmoothing = presmoothing

        self.filter_type = filter_type
        self.filter_size = filter_size
        self.device_id = device_id

    def calculate_flow(self, image1: np.ndarray, image2: np.ndarray,
                       start_point: typing.Tuple[int, int, int] = (0, 0, 0),
                       total_vol: typing.Tuple[int, int, int] = (-1, -1, -1),
                       sub_volume: typing.Tuple[int, int, int] = (256, 256, 256),
                       overlap: typing.Tuple[int, int, int] = (64, 64, 64),
                       threadsperblock: typing.Tuple[int, int, int] = (8, 8, 8),
                       ):
        """ Calculates the displacement across image1 and image2 using the 3D Pyramidal Lucas Kanade algorithm

        Args:
            image1 (np.ndarray): first image
            image2 (np.ndarray): second image
            start_point (typing.Tuple[int, int, int]): starting position of the region of interest in the image volume.
                Defaults to (0, 0, 0)
            total_vol (typing.Tuple[int, int, int]): total size of the region of interest. Defaults to (-1, -1, -1)
            sub_volume (typing.Tuple[int, int, int]): maximum volume size that can be analysed at one go.
                Defaults to (256, 256, 256)
            overlap (typing.Tuple[int, int, int]): amount of overlap between adjacent subvolumes.
                Defaults to (64, 64, 64)
            threadsperblock (typing.Tuple[int, int, int]): Defines the number of cuda threads.
                Defaults to (8, 8, 8)

        Returns:
            output_vz (np.ndarray): array containing the displacements in the x direction
            output_vy (np.ndarray): array containing the displacements in the y direction
            output_vx (np.ndarray): array containing the displacements in the z direction
        """
        if total_vol is None:
            total_vol = image1.shape - np.array(start_point)

        print("Running 3D pyramidal Lucas Kanade optical flow with the following parameters:")
        print(
            f"Iters: {self.iters} | Levels: {self.num_levels} | Scale: {self.scale} | Tau: {self.tau} | Alpha: {self.alpha} | Filter: {self.filter_type}-{self.filter_size} | Presmoothing: {self.presmoothing}",
            flush=True)

        output_vx = np.zeros(total_vol, dtype=np.float32)
        output_vy = np.zeros(total_vol, dtype=np.float32)
        output_vz = np.zeros(total_vol, dtype=np.float32)

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        if np.any(total_vol > sub_volume):
            shape = image1.shape
            z_position, z_valid_pos, z_valid = get_positions(start_point, total_vol, sub_volume, shape, overlap, 0)
            y_position, y_valid_pos, y_valid = get_positions(start_point, total_vol, sub_volume, shape, overlap, 1)
            x_position, x_valid_pos, x_valid = get_positions(start_point, total_vol, sub_volume, shape, overlap, 2)

            for z_i in range(len(z_position)):
                for y_i in range(len(y_position)):
                    for x_i in tqdm(range(len(x_position)),
                                    desc=f"Item: {z_i * len(y_position) + y_i + 1}/{len(z_position) * len(y_position)}"):
                        input_image_vol_1 = image1[z_position[z_i][0]:z_position[z_i][1],
                                            y_position[y_i][0]:y_position[y_i][1],
                                            x_position[x_i][0]:x_position[x_i][1]].astype(np.float32)
                        input_image_vol_2 = image2[z_position[z_i][0]:z_position[z_i][1],
                                            y_position[y_i][0]:y_position[y_i][1],
                                            x_position[x_i][0]:x_position[x_i][1]].astype(np.float32)

                        with cp.cuda.Device(self.device_id):
                            vx, vy, vz = pyrlk_3d(input_image_vol_1, input_image_vol_2,
                                                  iters=self.iters,
                                                  num_levels=self.num_levels,
                                                  scale=self.scale,
                                                  tau=self.tau, alpha=self.alpha,
                                                  filter_type=self.filter_type,
                                                  filter_size=self.filter_size,
                                                  presmoothing=self.presmoothing,
                                                  threadsperblock=threadsperblock)

                        cp.cuda.Stream.null.synchronize()

                        vx_cpu = vx.get()
                        vy_cpu = vy.get()
                        vz_cpu = vz.get()

                        output_vx[z_valid_pos[z_i][0]: z_valid_pos[z_i][1],
                        y_valid_pos[y_i][0]: y_valid_pos[y_i][1],
                        x_valid_pos[x_i][0]: x_valid_pos[x_i][1]] = vx_cpu[z_valid[z_i][0]: z_valid[z_i][1],
                                                                    y_valid[y_i][0]: y_valid[y_i][1],
                                                                    x_valid[x_i][0]: x_valid[x_i][1]]
                        output_vy[z_valid_pos[z_i][0]: z_valid_pos[z_i][1],
                        y_valid_pos[y_i][0]: y_valid_pos[y_i][1],
                        x_valid_pos[x_i][0]: x_valid_pos[x_i][1]] = vy_cpu[z_valid[z_i][0]: z_valid[z_i][1],
                                                                    y_valid[y_i][0]: y_valid[y_i][1],
                                                                    x_valid[x_i][0]: x_valid[x_i][1]]
                        output_vz[z_valid_pos[z_i][0]: z_valid_pos[z_i][1],
                        y_valid_pos[y_i][0]: y_valid_pos[y_i][1],
                        x_valid_pos[x_i][0]: x_valid_pos[x_i][1]] = vz_cpu[z_valid[z_i][0]: z_valid[z_i][1],
                                                                    y_valid[y_i][0]: y_valid[y_i][1],
                                                                    x_valid[x_i][0]: x_valid[x_i][1]]

                        del vx, vy, vz
                        del vx_cpu, vy_cpu, vz_cpu

                        mempool.free_all_blocks()
                        pinned_mempool.free_all_blocks()
        else:
            with cp.cuda.Device(self.device_id):
                vx, vy, vz = pyrlk_3d(image1, image2,
                                      iters=self.iters,
                                      num_levels=self.num_levels,
                                      scale=self.scale,
                                      tau=self.tau, alpha=self.alpha,
                                      filter_type=self.filter_type,
                                      filter_size=self.filter_size,
                                      presmoothing=self.presmoothing,
                                      threadsperblock=threadsperblock)

            cp.cuda.Stream.null.synchronize()

            output_vx = vx.get()
            output_vy = vy.get()
            output_vz = vz.get()

        return output_vz, output_vy, output_vx
