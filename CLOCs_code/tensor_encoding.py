import pathlib
import numpy as np
import time
import torch
from CLOCs_code.get_detection_results import *
from second.core.box_np_ops import camera_to_lidar

# Input Tensor T
# (IoU, 3D score, 2D score, tensor_index)

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# @numba.jit(nopython=True, parallel=True)


def get_tensor_T(boxes, query_boxes, criterion, scores_3d, scores_2d, dis_to_lidar_3d, overlaps, tensor_index):
    # boxes : projected 3D boxes
    # query_boxes : 2D bounding boxes
    # criterion == -1
    N = boxes.shape[0]  # 2000  # projected 3D boxes의 corners로 이루어진 2D box
    K = query_boxes.shape[0]  # 200  # 2D bounding boxes
    max_num = 100000
    ind = 0
    ind_max = ind
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:  # CLOCs에서는 criterion=-1
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[ind, 0] = iw * ih / ua
                    overlaps[ind, 1] = scores_3d[n, 0]
                    overlaps[ind, 2] = scores_2d[k, 0]
                    overlaps[ind, 3] = dis_to_lidar_3d[n, 0]
                    tensor_index[ind, 0] = k
                    tensor_index[ind, 1] = n
                    ind = ind+1

                elif k == K-1:
                    overlaps[ind, 0] = -10
                    overlaps[ind, 1] = scores_3d[n, 0]
                    overlaps[ind, 2] = -10
                    overlaps[ind, 3] = dis_to_lidar_3d[n, 0]
                    tensor_index[ind, 0] = k
                    tensor_index[ind, 1] = n
                    ind = ind+1
            elif k == K-1:
                overlaps[ind, 0] = -10
                overlaps[ind, 1] = scores_3d[n, 0]
                overlaps[ind, 2] = -10
                overlaps[ind, 3] = dis_to_lidar_3d[n, 0]
                tensor_index[ind, 0] = k
                tensor_index[ind, 1] = n
                ind = ind+1
    if ind > ind_max:
        ind_max = ind
    return overlaps, tensor_index, ind


def get_final_tensor_T(example, detection_2d_path, detection_3d_path):
    rect = example['rect'][0].cpu()
    Trv2c = example['Trv2c'][0].cpu()
    rect = rect.float()
    Trv2c = Trv2c.float()
    img_idx = example['image_idx'][0]

    # 3D detection results
    preds_dict = get_3D_results(img_idx, detection_3d_path)
    box_2d_preds = preds_dict['bbox'][0, :, :]
    box_3d_scores = preds_dict['score'].reshape(-1, 1)
    xyz_camera = preds_dict['location'][0, :, :].cpu()
    xyz_lidar = camera_to_lidar(xyz_camera, rect, Trv2c)
    xyz_lidar = torch.tensor(xyz_lidar)
    dis_to_lidar = torch.norm(xyz_lidar[:, :2], p=2, dim=1, keepdim=True) / 82.0

    # 2D detection results
    box_2d_detector = np.zeros((200, 4))
    detection_2d_results = get_2D_results(img_idx, detection_2d_path)
    box_2d_detector[0:detection_2d_results.shape[0],
                    :] = detection_2d_results[:, :4]
    box_2d_detector = detection_2d_results[:, :4]
    box_2d_scores = detection_2d_results[:, 4].reshape(-1, 1)

    time_iou_build_start = time.time()
    overlaps = np.zeros((100000, 4), dtype=box_2d_preds.detach().cpu().numpy().dtype)
    tensor_index = np.zeros((100000, 2), dtype=box_2d_preds.detach().cpu().numpy().dtype)
    overlaps[:, :] = -1
    tensor_index[:, :] = -1

    iou_test, tensor_index, max_num = get_tensor_T(box_2d_preds.detach().cpu().numpy(),
                                                   box_2d_detector,
                                                   -1,
                                                   box_3d_scores.detach().cpu().numpy(),
                                                   box_2d_scores,
                                                   dis_to_lidar.detach().cpu().numpy(),
                                                   overlaps,
                                                   tensor_index)

    time_iou_build_end = time.time()
    # iou_test_tensor shape: [160000,4]
    iou_test_tensor = torch.FloatTensor(iou_test)
    tensor_index_tensor = torch.LongTensor(tensor_index)
    iou_test_tensor = iou_test_tensor.permute(1, 0)
    iou_test_tensor = iou_test_tensor.reshape(1, 4, 1, 100000)
    tensor_index_tensor = tensor_index_tensor.reshape(-1, 2)
    if max_num == 0:
        non_empty_iou_test_tensor = torch.zeros(1, 4, 1, 2)
        non_empty_iou_test_tensor[:, :, :, :] = -1
        non_empty_tensor_index_tensor = torch.zeros(2, 2)
        non_empty_tensor_index_tensor[:, :] = -1
    else:
        non_empty_iou_test_tensor = iou_test_tensor[:, :, :, :max_num]
        non_empty_tensor_index_tensor = tensor_index_tensor[:max_num, :]

    return non_empty_iou_test_tensor, non_empty_tensor_index_tensor


#def camera_to_lidar(points, r_rect, velo2cam):
#    points_shape = list(points.shape[0:-1])
#    if points.shape[-1] == 3:
#        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
#    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
#    return lidar_points[..., :3]

