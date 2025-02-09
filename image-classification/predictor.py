'''
Author: yuan
Date: 2021-02-24 16:05:44
LastEditTime: 2021-03-09 10:06:51
FilePath: /yuan-algorithm/image-classification/predictor.py
'''
# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import copy
import multiprocessing as mp
from collections import deque

import cv2
import torch

from engine.defaults import DefaultPredictor


class VisualizationDemo(object):
    def __init__(self, cfg, gid=0, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        # self.metadata = MetadataCatalog.get(
        #     cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        # )
        self.cpu_device = torch.device("cpu")
        # self.instance_mode = instance_mode

        self.parallel = parallel
        # if parallel:
        #     num_gpu = torch.cuda.device_count()
        #     self.predictor = AsyncPredictor(cfg, num_gpus=1)
        # else:
        #     self.predictor = DefaultPredictor(cfg, gid=gid)
        self.predictor = DefaultPredictor(cfg, gid=gid)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        # vis_output = None
        predictions = self.predictor(image)

        # without batch predict

        # predictions = predictions.squeeze(0)  # replace batch
        # _, pred = predictions.topk(k=1, dim=0, largest=True, sorted=True)
        # pre = int(pred.cpu().numpy())

        _, preds = predictions.topk(k=1, dim=1, largest=True, sorted=True)

        pre = preds.cpu().numpy()

        output = torch.sigmoid(predictions)
        prob = output.cpu().numpy()
        return pre, prob


# class AsyncPredictor:
#     """
#     A predictor that runs the model asynchronously, possibly on >1 GPUs.
#     Because rendering the visualization takes considerably amount of time,
#     this helps improve throughput a little bit when rendering videos.
#     """

#     class _StopToken:
#         pass

#     class _PredictWorker(mp.Process):
#         def __init__(self, cfg, task_queue, result_queue, gid):
#             self.cfg = cfg
#             self.task_queue = task_queue
#             self.result_queue = result_queue
#             self.gid = gid
#             super().__init__()

#         def run(self):
#             predictor = DefaultPredictor(self.cfg, self.gid)

#             while True:
#                 task = self.task_queue.get()
#                 if isinstance(task, AsyncPredictor._StopToken):
#                     break
#                 idx, data = task
#                 result = predictor(data)
#                 self.result_queue.put((idx, result))

#     def __init__(self, cfg, num_gpus: int = 1):
#         """
#         Args:
#             cfg (CfgNode):
#             num_gpus (int): if 0, will run on CPU
#         """
#         num_workers = max(num_gpus, 1)
#         self.task_queue = mp.Queue(maxsize=num_workers * 1)
#         self.result_queue = mp.Queue(maxsize=num_workers * 1)
#         self.procs = []
#         for gpuid in range(max(num_gpus, 1)):
#             cfg = cfg.clone()
#             cfg.defrost()
#             # cfg.MODEL.DEVICE = "cuda:{}".format(
#             #     gpuid) if num_gpus > 0 else "cpu"
#             self.procs.append(
#                 AsyncPredictor._PredictWorker(
#                     cfg, self.task_queue, self.result_queue, gpuid)
#             )
#         self.put_idx = 0
#         self.get_idx = 0
#         self.result_rank = []
#         self.result_data = []

#         for p in self.procs:
#             p.start()
#         atexit.register(self.shutdown)

#     def put(self, image):
#         self.put_idx += 1
#         self.task_queue.put((self.put_idx, image))

#     def get(self):
#         self.get_idx += 1  # the index needed for this request
#         if len(self.result_rank) and self.result_rank[0] == self.get_idx:
#             res = self.result_data[0]
#             del self.result_data[0], self.result_rank[0]
#             return res

#         while True:
#             # make sure the results are returned in the correct order
#             idx, res = self.result_queue.get()
#             if idx == self.get_idx:
#                 return res
#             insert = bisect.bisect(self.result_rank, idx)
#             self.result_rank.insert(insert, idx)
#             self.result_data.insert(insert, res)

#     def __len__(self):
#         return self.put_idx - self.get_idx

#     def __call__(self, image):
#         self.put(image)
#         return self.get()

#     def shutdown(self):
#         for _ in self.procs:
#             self.task_queue.put(AsyncPredictor._StopToken())

#     # @property
#     # def default_buffer_size(self):
#     #     return len(self.procs) * 5
