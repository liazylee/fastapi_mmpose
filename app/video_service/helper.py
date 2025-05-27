# write a warpper function to check the function cost time
import time
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmpose.structures.bbox import bbox_xywh2xyxy


def timeit(func):
    """
    A decorator to measure the execution time of a function.

    Args:
        func: The function to be decorated.

    Returns:
        wrapper: The wrapped function that measures execution time.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper


def inference_topdown_batch(model: nn.Module,
                            imgs: List[Union[np.ndarray, str]],
                            bboxes=None,
                            bbox_format: str = 'xyxy'):
    """Inference image with a top-down pose estimator.

    Args:
        model (nn.Module): The top-down pose estimator
        img (np.ndarray | str): The loaded image or image file to inference
        bboxes (np.ndarray, optional): The bboxes in shape (N, 4), each row
            represents a bbox. If not given, the entire image will be regarded
            as a single bbox area. Defaults to ``None``
        bbox_format (str): The bbox format indicator. Options are ``'xywh'``
            and ``'xyxy'``. Defaults to ``'xyxy'``

    Returns:
        List[:obj:`PoseDataSample`]: The inference results. Specifically, the
        predicted keypoints and scores are saved at
        ``data_sample.pred_instances.keypoints`` and
        ``data_sample.pred_instances.keypoint_scores``.
    """
    scope = model.cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    new_bboxes = []
    for i, (img, bboxes_in_img) in enumerate(zip(imgs, bboxes)):
        if bboxes_in_img is None or len(bboxes_in_img) == 0:
            # get bbox from the image size
            if isinstance(img, str):
                w, h = Image.open(img).size
            else:
                h, w = img.shape[:2]

            new_bboxes.append(np.array([[0, 0, w, h]], dtype=np.float32))
        else:
            if isinstance(bboxes_in_img, list):
                new_bboxes.append(np.array(bboxes_in_img))
            else:
                new_bboxes.append(bboxes_in_img)

            assert bbox_format in {'xyxy', 'xywh'}, \
                f'Invalid bbox_format "{bbox_format}".'

            if bbox_format == 'xywh':
                new_bboxes[i] = bbox_xywh2xyxy(new_bboxes[i])

    # construct batch data samples
    data_list = []
    idx_list = []
    for img, bbox in zip(imgs, new_bboxes):
        for i in range(len(bbox)):
            if isinstance(img, str):
                data_info = dict(img_path=img)
            else:
                data_info = dict(img=img)
            data_info['bbox'] = bbox[i][None]
            data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
            data_list.append(pipeline(data_info))
            idx_list.append(i)

    if data_list:
        # collate data list into a batch, which is a dict with following keys:
        # batch['inputs']: a list of input images
        # batch['data_samples']: a list of :obj:`PoseDataSample`
        batch = pseudo_collate(data_list)
        with torch.no_grad():
            results = model.test_step(batch)
    else:
        results = []

    results_new = []
    for result, idx in zip(results, idx_list):
        if idx == 0:
            results_new.append([result])
        else:
            results_new[-1].append(result)

    return results_new
