import cupy as cp
import numpy as np
from tqdm import tqdm

from .helpers.farneback_functions import farneback_3d
from .helpers.helpers import get_positions
from .helpers.pyrlk_functions import pyrlk_3d


class Farneback3D:
    def __init__(self,
                 iters=5,
                 num_levels=5,
                 scale=0.5,
                 kernel_size=7,
                 presmoothing=7,
                 sigma_k=0.15,
                 filter_type="box",
                 filter_size=21,
                 device_id=0,
                 ):
        self.iters = iters
        self.num_levels = num_levels
        self.scale = scale
        self.kernel_size = kernel_size

        self.presmoothing = presmoothing
        self.sigma_k = sigma_k
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.device_id = device_id

    def calculate_flow(self, image1, image2,
                       start_point=(0, 0, 0),
                       total_vol=None,
                       sub_volume=(256, 256, 256),
                       overlap=(64, 64, 64),
                       threadsperblock=(8, 8, 8),
                       ):
        if total_vol is None:
            total_vol = image1.shape - np.array(start_point)

        print("Running 3D Farneback optical flow with the following parameters:")
        print(
            f"Iters: {self.iters} | Levels: {self.num_levels} | Scale: {self.scale} | Kernel: {self.kernel_size} | Filter: {self.filter_type}-{self.filter_size} | Presmoothing: {self.presmoothing}",
            flush=True)

        output_vx = np.zeros(total_vol, dtype=np.float32)
        output_vy = np.zeros(total_vol, dtype=np.float32)
        output_vz = np.zeros(total_vol, dtype=np.float32)
        output_error = np.zeros(total_vol, dtype=np.float32)

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
                            vx, vy, vz, error = farneback_3d(input_image_vol_1, input_image_vol_2, self.iters,
                                                             self.num_levels,
                                                             scale=self.scale, kernelsize=self.kernel_size,
                                                             sigma_k=self.sigma_k, filter_type=self.filter_type,
                                                             filter_size=self.filter_size,
                                                             presmoothing=self.presmoothing,
                                                             threadsperblock=threadsperblock)

                        cp.cuda.Stream.null.synchronize()

                        vx_cpu = vx.get()
                        vy_cpu = vy.get()
                        vz_cpu = vz.get()
                        error_cpu = error.get()

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

                        output_error[z_valid_pos[z_i][0]: z_valid_pos[z_i][1],
                                     y_valid_pos[y_i][0]: y_valid_pos[y_i][1],
                                     x_valid_pos[x_i][0]: x_valid_pos[x_i][1]] = error_cpu[z_valid[z_i][0]: z_valid[z_i][1],
                                                                                           y_valid[y_i][0]: y_valid[y_i][1],
                                                                                           x_valid[x_i][0]: x_valid[x_i][1]]

                        del vx, vy, vz, error
                        del vx_cpu, vy_cpu, vz_cpu, error_cpu

                        mempool.free_all_blocks()
                        pinned_mempool.free_all_blocks()
        else:
            with cp.cuda.Device(self.device_id):
                vx, vy, vz, error = farneback_3d(image1, image2,
                                                 iters=self.iters,
                                                 num_levels=self.num_levels,
                                                 scale=self.scale, kernelsize=self.kernel_size,
                                                 sigma_k=self.sigma_k,
                                                 filter_type=self.filter_type, filter_size=self.filter_size,
                                                 presmoothing=self.presmoothing,
                                                 threadsperblock=threadsperblock)
            cp.cuda.Stream.null.synchronize()

            output_vx = vx.get()
            output_vy = vy.get()
            output_vz = vz.get()
            output_error = error.get()

        return output_vz, output_vy, output_vx, output_error


class PyrLK3D:
    def __init__(self,
                 iters=15,
                 num_levels=5,
                 scale=0.5,
                 tau=0.1, alpha=0.1,
                 filter_type="gaussian",
                 filter_size=21,
                 presmoothing=3,
                 device_id=0,
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

    def calculate_flow(self, image1, image2,
                       start_point=(0, 0, 0),
                       total_vol=None,
                       sub_volume=(256, 256, 256),
                       overlap=(64, 64, 64),
                       threadsperblock=(8, 8, 8),
                       ):
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
