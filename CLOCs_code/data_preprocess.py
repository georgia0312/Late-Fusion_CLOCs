import numpy as np
from second.pytorch.core import box_torch_ops
import torch


def remove_dontcare(image_anno):
    img_filtered_annotations = {}
    relevant_annotation_indices = [i for i, x in enumerate(image_anno['name']) if x != "DontCare"]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])

    return img_filtered_annotations


def get_class_to_label_map():
    class_to_label = {
        'Car': 0,
        'Pedestrian': 1,
        'Cyclist': 2,
        'Van': 3,
        'Person_sitting': 4,
        'Truck': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': -1,
    }
    return class_to_label


def get_classes():
    return get_class_to_label_map().keys()


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def cat2label(classes, name):
    cat2label = {k: i for i, k in enumerate(classes)}
    labels = []
    for i in name:
        label = cat2label[i]
        labels.append(label)
    return np.array(labels)


def _read_info(info, used_classes):
    image_idx = info['image_idx']
    rect = info['calib/R0_rect'].astype(np.float32)
    Trv2c = info['calib/Tr_velo_to_cam'].astype(np.float32)
    P2 = info['calib/P2'].astype(np.float32)

    input_dict = {
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2,
        'image_shape': np.array(info["img_shape"], dtype=np.int32),
        'image_idx': int(image_idx),
        'image_path': info['img_path'],
    }

    if 'annos' in info:
        annos = info['annos']
        annos = remove_dontcare(annos)
        loc = annos["location"]
        dims = annos["dimensions"]   # dimensions: l, h, w
        rots = annos["rotation_y"]
        gt_names = annos["name"]
        selected = keep_arrays_by_name(gt_names, used_classes)
        gt_names = gt_names[selected]
        class_to_label = get_class_to_label_map()
        gt_names2label = np.array([], dtype=np.int32)
        for name in gt_names:
            if name in get_classes():
                label = class_to_label[name]
                gt_names2label = np.append(gt_names2label, label)
        gt_2d_box = annos["bbox"][selected,:].astype(np.float32)
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        gt_boxes_camera = gt_boxes_camera[selected,:]
        difficulty = annos["difficulty"][selected]
        input_dict.update({
            'gt_boxes_camera': gt_boxes_camera,
            'gt_names': gt_names2label,
            'difficulty': difficulty,
            'gt_2d_bboxes': gt_2d_box,
        })
        if 'group_ids' in annos:
            input_dict['group_ids'] = annos["group_ids"][selected]

    return input_dict


def get_3d_camera(gt_info):
    location = gt_info['location'].reshape(-1, 3)
    dimensions = gt_info['dimensions'].reshape(-1, 3)  # lhw
    rotation_y = gt_info['rotation_y'].reshape(-1, 1)
    box_camera = torch.concat([location, dimensions, rotation_y], dim=1)
    return box_camera


def get_preds_result(example, preds_3d):
    rect = example['rect'][0].cpu()
    Trv2c = example['Trv2c'][0].cpu()
    bbox = preds_3d['bbox'].cpu()
    xyz = preds_3d['location'][0, :, :].cpu()
    lhw = preds_3d['dimensions'][0, :, :].cpu()
    r = preds_3d['rotation_y'].reshape(-1, 1).cpu()

    # 2d bbox
    pred_bbox = torch.zeros(1, 2000, 4)
    bbox = preds_3d['bbox'][0, :, :].cpu()
    pred_bbox[:, :bbox.shape[0], :] = bbox

    # box3d_camera
    box_preds_camera = torch.zeros(1, 2000, 7)
    box_camera = torch.cat([xyz, lhw, r], dim=1)
    box_preds_camera[:, :box_camera.shape[0], :] = box_camera

    # box3d_lidar
    box_preds_lidar = torch.zeros(1, 2000, 7)
    box_lidar = box_torch_ops.box_camera_to_lidar(box_camera, rect, Trv2c).type(torch.float32)
    box_preds_lidar[:, :box_lidar.shape[0], :] = box_lidar

    final_scores = preds_3d['score'].cpu().type(torch.float32)

    preds_dict = {
        "box_preds": box_preds_lidar,
        "cls_preds": final_scores,
        "bbox": pred_bbox,
        "box3d_camera": box_preds_camera
    }

    return preds_dict
