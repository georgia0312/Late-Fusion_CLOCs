import torch
from torch import nn
from torch.nn import functional as F
import torchplus
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
import time


class Fusion(nn.Module):
    def __init__(self):
        self._num_class = 1
        self._encode_background_as_zeros = True
        self._use_sigmoid_score = True
        self._use_rotate_nms = True
        self._multiclass_nms = False
        self._nms_score_threshold = 0.5
        self._nms_pre_max_size = 1000
        self._nms_post_max_size = 20
        self._nms_iou_threshold = 0.1
        self._time_dict = {}
        self._time_total_dict = {}
        self._time_count_dict = {}
        self.measure_time = False

        super(Fusion, self).__init__()
        self.name = 'fusion'
        self.corner_points_feature = Sequential(
            nn.Conv2d(24, 48, 1),
            nn.ReLU(),
            nn.Conv2d(48, 96, 1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 1),
            nn.ReLU(),
            nn.Conv2d(96, 4, 1),
        )
        self.fuse_2d_3d = Sequential(
            nn.Conv2d(4, 18, 1),
            nn.ReLU(),
            nn.Conv2d(18, 36, 1),
            nn.ReLU(),
            nn.Conv2d(36, 36, 1),
            nn.ReLU(),
            nn.Conv2d(36, 1, 1),
        )
        self.maxpool = Sequential(
            nn.MaxPool2d([200, 1], 1),
        )

    def forward(self, input_1, tensor_index):
        flag = -1
        if tensor_index[0, 0] == -1:
            out_1 = torch.zeros(1, 200, 70400, dtype=input_1.dtype, device=input_1.device)
            out_1[:, :, :] = -9999999
            flag = 0
        else:
            x = self.fuse_2d_3d(input_1)
            out_1 = torch.zeros(1, 200, 70400, dtype=input_1.dtype, device=input_1.device)
            out_1[:, :, :] = -9999999
            out_1[:, tensor_index[:, 0], tensor_index[:, 1]] = x[0, :, 0, :]
            flag = 1
        x = self.maxpool(out_1)
        # x, _ = torch.max(out_1,1)
        x = x.squeeze().reshape(1, -1, 1)

        return x, flag

    def start_timer(self, *names):
        if not self.measure_time:
            return
        for name in names:
            self._time_dict[name] = time.time()
        torch.cuda.synchronize()

    def end_timer(self, name):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        time_elapsed = time.time() - self._time_dict[name]
        if name not in self._time_count_dict:
            self._time_count_dict[name] = 1
            self._time_total_dict[name] = time_elapsed
        else:
            self._time_count_dict[name] += 1
            self._time_total_dict[name] += time_elapsed
        self._time_dict[name] = 0

    def clear_timer(self):
        self._time_count_dict.clear()
        self._time_dict.clear()
        self._time_total_dict.clear()

    def get_avg_time_dict(self):
        ret = {}
        for name, val in self._time_total_dict.items():
            count = self._time_count_dict[name]
            ret[name] = val / max(1, count)
        return ret

    def set_timer(self):
        self._time_dict = {}
        self._time_total_dict = {}
        self._time_count_dict = {}

    def set_global_step(self, step_value):
        self.global_step = step_value

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])
