from torchplus.train.checkpoint import *
from torchplus.train.common import *
import os
import torch
import torch.nn as nn
from mmengine import Config
import numpy as np
import pathlib
import pickle
import json
import time
from tqdm.notebook import tqdm
from tensorboardX import SummaryWriter
from second.pytorch.builder import lr_scheduler_builder, optimizer_builder
from second.utils.progress_bar import ProgressBar
from second.utils.eval import get_coco_eval_result, get_official_eval_result, bev_box_overlap, d3_box_overlap
from second.core.box_np_ops import second_box_decode as decode_torch
from CLOCs_code.get_detection_results import get_3D_results
from CLOCs_code.tensor_encoding import get_final_tensor_T
import second.data.kitti_common as kitti
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import SigmoidFocalClassificationLoss
from CLOCs_code.fusion_model_v2 import Fusion
from CLOCs_code.dataset import *
from CLOCs_code.data_preprocess import get_preds_result, get_classes


def evaluate(config_path,
            model_dir,
            result_path=None,
            predict_test=True,
            ckpt_path=None,
            ref_detfile=None,
            pickle_result=True,
            measure_time=False,
            batch_size=None,
            device=None):

    model_dir = pathlib.Path(model_dir)
    print("Predict_evaluation: ", predict_test)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)

    config = Config.fromfile(config_path)
    input_cfg = config.eval_input_reader
    train_cfg = config.train_config
    detection_2d_path = config.train_config.detection_2d_path
    detection_3d_path = config.train_config.detection_3d_path

    # this one is used for training car detector
    fusion = Fusion()
    fusion.to(device)

    # restore parameters for fusion layer
    if ckpt_path is None:
        print("load existing model for fusion layer")
        try_restore_latest_checkpoints(model_dir, [fusion])
    else:
        restore(ckpt_path, fusion)

    batch_size = input_cfg.batch_size

    root_path = input_cfg.kitti_root_path
    eval_info_path = input_cfg.kitti_info_path
    used_classes = input_cfg.used_classes

    eval_dataset = KITTIDataset(root_path, eval_info_path, used_classes)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # input_cfg.num_workers,
        pin_memory=False)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    fusion.eval()
    result_path_step = result_path / f"step_{fusion.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    dt_annos = []
    
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    prep_example_times = []
    prep_times = []
    t2 = time.time()
    eval_loss_final = 0
    for example in iter(eval_dataloader):
        if measure_time:
            prep_times.append(time.time() - t2)
            t1 = time.time()
            torch.cuda.synchronize()
        example = example_convert_to_torch(example, float_dtype, device)
        if measure_time:
            torch.cuda.synchronize()
            prep_example_times.append(time.time() - t1)

        if pickle_result:
            dt_annos_i, eval_losses = predict_kitti_to_anno(fusion, detection_2d_path, detection_3d_path, example, used_classes)
            dt_annos += dt_annos_i
            eval_loss_final = eval_loss_final + eval_losses
        else:
            _predict_kitti_to_file(fusion, detection_2d_path, detection_3d_path, example, used_classes, result_path_step)
        bar.print_bar()
        if measure_time:
            t2 = time.time()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')
    print("test_loss:", eval_loss_final / len(eval_dataloader))
    if measure_time:
        print(
            f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms")
        print(f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
    for name, val in fusion.get_avg_time_dict().items():
        print(f"avg {name} time = {val * 1000:.3f} ms")
    if not predict_test:
        gt_annos = [info["annos"] for info in eval_dataset.kitti_infos]
        if not pickle_result:
            dt_annos = kitti.get_label_annos(result_path_step)
        result = get_official_eval_result(gt_annos, dt_annos, used_classes)
        print("################## Evaluation Result ##################")
        print(result)
        result = get_coco_eval_result(gt_annos, dt_annos, used_classes)
        print()
        print("################## COCO Evaluation Result ##################")
        print(result)
        if pickle_result:
            with open(result_path_step / "result.pkl", 'wb') as f:
                pickle.dump(dt_annos, f)
    else:
        if pickle_result:
            with open(result_path_step / "result.pkl", 'wb') as f:
                pickle.dump(dt_annos, f)