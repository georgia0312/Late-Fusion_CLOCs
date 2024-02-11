import pickle
import pathlib
import numpy as np
import torch
import time


# 2D detection results
def get_2D_results(img_idx, detection_2d_path):
    img_idx = "{:06d}".format(int(img_idx))
    detection_2d_result_path = pathlib.Path(detection_2d_path)
    detection_2d_file_name = f"{detection_2d_result_path}/{img_idx}.txt"
    with open(detection_2d_file_name, 'r') as f:
        lines = f.readlines()
    # Car -1 -1 -10 1133.50 278.19 1225.04 329.51 -1 -1 -1 -1000 -1000 -1000 -10 0.0150
    content = [line.strip().split(' ') for line in lines]
    predicted_class = np.array([x[0] for x in content], dtype='object')
    predicted_class_index = np.where(predicted_class == 'Car')

    # (xmin, ymin, xmax, ymax) 좌표
    detection_result = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    score = np.array([float(x[15]) for x in content])
    f_detection_result = np.append(detection_result, score.reshape(-1, 1), 1)
    middle_predictions = f_detection_result[predicted_class_index, :].reshape(
        -1, 5)
    # score >= -100 일 경우 top_predictions에 포함
    # 2D detection을 fusion에 포함시키는 기준이 score >= -100 인게 이상함
    top_predictions = middle_predictions[np.where(middle_predictions[:, 4] >= -100)]

    return top_predictions


# 3D detection results

def get_3D_results_v2(img_idx, detection_3d_path):
    img_idx = "{:06d}".format(int(img_idx))
    detection_3d_result_path = pathlib.Path(detection_3d_path)
    detection_3d_file_name = f"{detection_3d_result_path}/{img_idx}.txt"
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': []
    })

    with open(detection_3d_file_name, 'r') as f:
        lines = f.readlines()
    # Car 0.0000 0 1.7741 388.0230 181.2514 418.0886 200.8547 3.9893 1.4921 1.5774 -16.8237 2.1997 58.7204 1.4959 0.7411
    content = [line.strip().split(' ') for line in lines]
    predicted_class = np.array(
        [x[0] for x in content], dtype='object').reshape(-1, 1)
    predicted_class_index = np.where(predicted_class == 'Car')
    # truncated
    # truncated = np.array([[float(info) for info in x[1]] for x in content])
    truncated = np.array([float(x[1]) for x in content]).reshape(-1, 1)
    # occluded
    occluded = np.array([int(x[2]) for x in content]).reshape(-1, 1)
    # alpha
    alpha = np.array([float(x[3]) for x in content]).reshape(-1, 1)
    # (xmin, ymin, xmax, ymax) 좌표
    box_2d_preds = np.array([[float(info) for info in x[4:8]]
                            for x in content]).reshape(-1, 4)
    # dimensions: h, w, l -> l, h, w - dimensions will convert hwl format to standard lhw(camera) format.
    dimensions = np.array([[float(info) for info in x[8:11]]
                          for x in content]).reshape(-1, 3)[:, [2, 0, 1]]
    # location: x, y, z (객체 중심의 3D 좌표)
    location = np.array([[float(info) for info in x[11:14]]
                        for x in content]).reshape(-1, 3)
    # 객체의 회전 각도 (카메라에서 본 시계 방향의 각도)
    rotation_y = np.array([float(x[14]) for x in content]).reshape(-1, 1)
    score = np.array([float(x[15]) for x in content]).reshape(-1, 1)
    detection_3d_result = np.concatenate([predicted_class, truncated, occluded, alpha,
                                          box_2d_preds, dimensions, location, rotation_y, score], axis=1)
    preds_dict = detection_3d_result[predicted_class_index, :].reshape(-1, 16)

    return preds_dict


def get_3D_results(img_idx, detection_3d_path):
    img_idx = "{:06d}".format(int(img_idx))
    detection_3d_result_path = pathlib.Path(detection_3d_path)
    detection_3d_file_name = f"{detection_3d_result_path}/{img_idx}.txt"
    anno = {}
    anno.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': []
    })

    with open(detection_3d_file_name, 'r') as f:
        lines = f.readlines()
    # Car 0.0000 0 1.7741 388.0230 181.2514 418.0886 200.8547 3.9893 1.4921 1.5774 -16.8237 2.1997 58.7204 1.4959 0.7411
    content = [line.strip().split(' ') for line in lines]
    predicted_class = np.array([x[0] for x in content], dtype='object')
    anno['name'] = predicted_class
    predicted_class_index = np.where(predicted_class == 'Car')
    # truncated
    anno['truncated'] = np.array([float(x[1]) for x in content])
    # occluded
    anno['occluded'] = np.array([int(x[2]) for x in content])
    # alpha
    anno['alpha'] = np.array([float(x[3]) for x in content])
    # (xmin, ymin, xmax, ymax) 좌표
    anno['bbox'] = np.array([[float(info) for info in x[4:8]]
                            for x in content]).reshape(-1, 4)
    # dimensions: h, w, l -> l, h, w
    anno['dimensions'] = np.array([[float(info) for info in x[8:11]] for x in content]).reshape(-1, 3)[:, [2, 0, 1]]

    # location: x, y, z (객체 중심의 3D 좌표)
    anno['location'] = np.array([[float(info) for info in x[11:14]]
                                 for x in content]).reshape(-1, 3)
    # 객체의 회전 각도 (카메라에서 본 시계 방향의 각도)
    anno['rotation_y'] = np.array([float(x[14]) for x in content])
    anno['score'] = np.array([float(x[15]) for x in content])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k, v in anno.items():
        if k == 'name':
            v = v[predicted_class_index]
            anno[k] = v
        elif len(v.shape) == 2:
            v = v[predicted_class_index, :]
            anno[k] = torch.tensor(v, dtype=torch.float32, device=device)
        else:
            v = v[predicted_class_index]
            anno[k] = torch.tensor(v, dtype=torch.float32, device=device)

    return anno

