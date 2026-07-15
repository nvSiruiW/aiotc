#!/usr/bin/env python3
"""Extra open-source PDR/inertial-odometry model builders for the edge benchmark.

Each builder returns (net.eval(), input_shape) like profile_device.MODEL_REGISTRY.
Builders import repo-specific code lazily; RUN ONE MODEL PER PROCESS to avoid
module-name clashes between repos (several ship a `model_resnet1d.py`).

Covered: MobileNetV2 / MnasNet / EfficientNetB0 (IMUNet repo, mobile CNN family),
TLIO (ResNet1D + uncertainty head), TinyOdom (NAS dilated-TCN, PyTorch reimpl),
EqNIO (SO(2)/O(2)-equivariant frame net, Vector-Neuron layers, no escnn).
"""
import os, sys, copy
import torch, torch.nn as nn, torch.nn.functional as F

AIOTC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _add_path(rel):
    p = os.path.join(AIOTC, rel)
    if p not in sys.path: sys.path.insert(0, p)
    return p

# ---------------- mobile CNN family (already adapted to IMU in IMUNet repo) ----
def _mobilenetv2():
    _add_path("IMUNet/RONIN_torch"); from MobileNetV2 import MobileNetV2
    return MobileNetV2(n_class=2).eval(), (1, 6, 200)

def _mnasnet():
    _add_path("IMUNet/RONIN_torch"); from MnasNet import MnasNet
    return MnasNet(n_class=2).eval(), (1, 6, 200)

def _efficientnet_b0():
    _add_path("IMUNet/RONIN_torch"); from EfficientnetB0 import EfficientNetB0
    return EfficientNetB0(n_class=2).eval(), (1, 6, 200)

# ---------------- TLIO (RA-L 2020): ResNet1D regressor + covariance head -------
def _tlio_resnet():
    # import model_resnet directly (bypasses network/__init__.py -> liegroups);
    # the module itself only needs torch.nn.
    _add_path("TLIO/src/network")
    import model_resnet as mr
    # factory: ResNet1D(BasicBlock1D, in_dim=6, out_dim=3, [2,2,2,2], inter_dim);
    # inter_dim = window//32+1 = 7 for a 200-sample window.
    net = mr.ResNet1D(mr.BasicBlock1D, 6, 3, [2, 2, 2, 2], 7).eval()
    return net, (1, 6, 200)

# ---------------- TinyOdom (IPSN 2022): NAS dilated-TCN, faithful PyTorch reimpl
# keras-tcn config from tinyodom/RoNIN/TinyOdom_RoNIN.ipynb:
#   TCN(nb_filters=32, kernel_size=7, dilations=[1,2,4,8,16,32,64,128],
#       use_skip_connections=False) -> last step -> reshape -> MaxPool1D(2)
#       -> Flatten -> Dense(32) -> Dense(1)velx + Dense(1)vely ; window_size=400
class _TCNResBlock(nn.Module):
    def __init__(self, in_ch, filters, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation                       # causal padding
        self.conv1 = nn.Conv1d(in_ch, filters, k, dilation=dilation, padding=self.pad)
        self.conv2 = nn.Conv1d(filters, filters, k, dilation=dilation, padding=self.pad)
        self.down = nn.Conv1d(in_ch, filters, 1) if in_ch != filters else None
    def forward(self, x):
        y = self.conv1(x)[..., :-self.pad]; y = F.relu(y)
        y = self.conv2(y)[..., :-self.pad]; y = F.relu(y)
        res = x if self.down is None else self.down(x)
        return F.relu(y + res)

class TinyOdomTCN(nn.Module):
    def __init__(self, in_ch=6, nb_filters=32, k=7, dilations=(1,2,4,8,16,32,64,128)):
        super().__init__()
        blocks, c = [], in_ch
        for d in dilations:
            blocks.append(_TCNResBlock(c, nb_filters, k, d)); c = nb_filters
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.MaxPool1d(2)
        self.pre = nn.Linear(nb_filters // 2, 32)
        self.velx = nn.Linear(32, 1); self.vely = nn.Linear(32, 1)
    def forward(self, x):                     # x: (B, 6, L)
        h = self.tcn(x)[..., -1]              # last timestep -> (B, nb_filters)
        h = self.pool(h.unsqueeze(1)).flatten(1)   # (B, nb_filters/2)
        h = self.pre(h)                       # (B, 32)  (linear activation)
        return torch.cat([self.velx(h), self.vely(h)], dim=-1)

def _tinyodom():
    return TinyOdomTCN().eval(), (1, 6, 400)

# ---------------- EqNIO (2024): O(2)-equivariant frame net (Vector-Neuron) ------
def _preprocess_eq_o2_frame(feat):
    """Copied from EqNIO ronin_resnet.py: raw IMU (B,6,N) -> (vector, scalar, orig)."""
    feat = feat.permute(0, 2, 1)                     # (B,N,6)
    gyro = feat[..., :3]; accel = feat[..., -3:]
    v1 = torch.zeros((*gyro.shape[:-1], 3), device=gyro.device, dtype=gyro.dtype)
    v2 = torch.zeros((*gyro.shape[:-1], 3), device=gyro.device, dtype=gyro.dtype)
    mask = (torch.linalg.norm(gyro[..., :-1], axis=-1) == 0).to(torch.int32)
    R = torch.tensor([[0,-1,0],[1,0,0],[0,0,1]], dtype=gyro.dtype, device=gyro.device)
    gyro_flip = copy.deepcopy(gyro @ R.T); gyro_flip[..., -1] = 0
    v1[mask == 0] = torch.linalg.cross(gyro[mask == 0], gyro_flip[mask == 0])
    v2[mask == 0] = torch.linalg.cross(gyro[mask == 0], v1[mask == 0])
    x = torch.zeros_like(gyro); x[..., -2] = 1
    v1[mask == 1] = torch.linalg.cross(x[mask == 1], gyro[mask == 1])
    v2[mask == 1] = torch.linalg.cross(gyro[mask == 1], v1[mask == 1])
    gn = torch.linalg.norm(gyro, axis=-1, keepdims=True)
    v1 = v1 * gn / torch.linalg.norm(v1, axis=-1, keepdims=True).clamp(min=1e-7)
    v2 = v2 * gn / torch.linalg.norm(v2, axis=-1, keepdims=True).clamp(min=1e-7)
    a = accel[..., :2].unsqueeze(-2)
    v1xy = v1[..., :2].unsqueeze(-2); v2xy = v2[..., :2].unsqueeze(-2)
    scalar = torch.cat((accel[..., -1].unsqueeze(-1), v1[..., -1].unsqueeze(-1), v2[..., -1].unsqueeze(-1),
                        torch.norm(a, dim=-1), torch.norm(v1xy, dim=-1), torch.norm(v2xy, dim=-1),
                        (a @ v1xy.permute(0,1,3,2)).squeeze(-1), (v1xy @ v2xy.permute(0,1,3,2)).squeeze(-1),
                        (a @ v2xy.permute(0,1,3,2)).squeeze(-1)), dim=-1)
    vector = torch.cat((a, v1xy, v2xy), dim=-2).permute(0,1,3,2)
    orig = torch.cat((accel[..., -1].unsqueeze(-1), v1[..., -1].unsqueeze(-1), v2[..., -1].unsqueeze(-1)), dim=-1)
    return vector, scalar, orig

class EqNIOWrapper(nn.Module):
    """Feeds a standard (1,6,200) IMU window through EqNIO's o2 preprocessing + net."""
    def __init__(self):
        super().__init__()
        _add_path("EqNIO/RONIN/source")
        from model_resnet1d_eq_frame_o2 import Eq_Motion_Model_o2
        self.net = Eq_Motion_Model_o2(dim_in=3, dim_out=2, scalar_dim_in=9, pooling_dim=1,
                                      ronin_in_dim=6, ronin_out_dim=2, hidden_dim=64,
                                      scalar_hidden_dim=64, depth=2, stride=1,
                                      padding='same', kernel=(32,1), bias=False)
    def forward(self, x):
        v, s, o = _preprocess_eq_o2_frame(x)
        _, vel = self.net(v, s, o)
        return vel

def _eqnio():
    return EqNIOWrapper().eval(), (1, 6, 200)

EXT_REGISTRY = {
    "mobilenetv2": _mobilenetv2,
    "mnasnet": _mnasnet,
    "efficientnet_b0": _efficientnet_b0,
    "tlio_resnet": _tlio_resnet,
    "tinyodom": _tinyodom,
    "eqnio": _eqnio,
}
