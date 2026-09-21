from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_eresnet
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.weave_face import WeaveFace
import numpy as np
from thop import profile
from thop import clever_format

parser = argparse.ArgumentParser(description='WeaveFace Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()

# ─────────────────────────────────────────────────────────────
# Mamba (SSM) FLOPs handler — VMamba 공식 공식
# ─────────────────────────────────────────────────────────────
def flops_selective_scan_fn(B, L, D, N, with_D=True, with_Z=False):
    flops = 0
    flops += B * D * L * N      # deltaA
    flops += B * D * L * N * 2  # deltaB_u (mul 2개)
    flops += B * D * L * N      # recurrence (mul+add = 1 MAC)  ← 여기 2 → 1
    flops += B * D * L * N      # y = x @ C
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops
# 총 5 · B·D·L·N


def selective_scan_flop_jit(inputs, outputs, backend="prefixsum", verbose=False):
    if verbose:
        for i in range(min(5, len(inputs))):
            try:
                print(f"  inputs[{i}]: {inputs[i].type().sizes()}")
            except Exception:
                pass
    B_, D_, L_ = inputs[0].type().sizes()           # u: [B, D, L]
    N_ = inputs[2].type().sizes()[1]                # A: [D, N]
    return flops_selective_scan_fn(B=B_, L=L_, D=D_, N=N_, with_D=True, with_Z=False)


# ─────────────────────────────────────────────────────────────
# DCNv4 FLOPs handler
# DCNv1 (torchvision::deform_conv2d)을 DCNv4로 대체했으므로
# 더 이상 deform_conv2d_flop_jit는 필요 없음.
# ─────────────────────────────────────────────────────────────
def dcnv3_flop_jit(inputs, outputs, kernel_size=3):
    """
    DCNv4Function (deformable aggregation) FLOPs.

    DCNv4 내부 nn.Linear들(value_proj, offset_mask, output_proj)은
    fvcore가 aten::linear로 자동 카운트하므로, 여기선 deformable
    aggregation 코어만 계산.

    공식: B · L · C · K²
      각 출력 픽셀의 각 채널마다 K² 위치에서 bilinear sampling 후
      mask 가중합 → K² 회의 MAC 연산.

    Args:
        kernel_size: 기본 3 (3x3 = K²=9). 다른 kernel 쓰면 인자로 전달.
    """
    out_sizes = outputs[0].type().sizes()
    if len(out_sizes) == 3:        # (B, L, C)
        B, L, C = out_sizes
    elif len(out_sizes) == 4:      # (B, H, W, C)
        B, H, W, C = out_sizes
        L = H * W
    else:
        return 0

    K_sq = kernel_size * kernel_size
    return B * L * C * K_sq


import copy
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count


# ─────────────────────────────────────────────────────────────
# 메인 측정 함수
# ─────────────────────────────────────────────────────────────
def calculate_flops(net, img_dim=(480, 640), verbose=False, show_module_breakdown=False):
    """
    Mamba + DCNv4 + CNN 모델용 FLOPs/Params 측정.
    Convention: inference-time MACs (BN folded into conv, EResFD/MobileNet 컨벤션).
    """
    if isinstance(img_dim, int):
        H, W = img_dim, img_dim
    else:
        H, W = img_dim

    model = copy.deepcopy(net).cuda().eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, H, W, device=device)

    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy_input)

    # ── 커스텀 op handler ──
    if verbose:
        flops.set_op_handle(
            "prim::PythonOp.SelectiveScanCuda",
            lambda i, o: selective_scan_flop_jit(i, o, verbose=True),
        )
    else:
        flops.set_op_handle("prim::PythonOp.SelectiveScanCuda", selective_scan_flop_jit)

    # DCNv4 handler 등록
    flops.set_op_handle("prim::PythonOp.DCNv3Function", dcnv3_flop_jit)

    # ── 제외 op (inference-time convention) ──
    for op in [
        # === BN (모든 변형) ===
        "aten::batch_norm",
        "aten::native_batch_norm",
        "aten::_native_batch_norm_legit",
        "aten::_native_batch_norm_legit_no_training",
        "aten::_native_batch_norm_legit_functional",
        "aten::cudnn_batch_norm",
        "aten::_batch_norm_impl_index",
        "aten::miopen_batch_norm",

        # === LayerNorm (모든 변형) ===
        "aten::layer_norm",
        "aten::native_layer_norm",
        "aten::_native_layer_norm",

        # === GroupNorm / InstanceNorm ===
        "aten::group_norm",
        "aten::native_group_norm",
        "aten::instance_norm",

        # === Upsample/Interpolate (전부) ===
        "aten::upsample_nearest1d",
        "aten::upsample_nearest2d",
        "aten::upsample_nearest3d",
        "aten::upsample_linear1d",
        "aten::upsample_bilinear2d",
        "aten::upsample_bilinear2d.vec",
        "aten::upsample_trilinear3d",
        "aten::upsample_bicubic2d",
        "aten::_upsample_bilinear2d_aa",
        "aten::_upsample_nearest_exact2d",
        "aten::interpolate",

        # === Activations ===
        "aten::silu", "aten::silu_",
        "aten::gelu", "aten::gelu_",
        "aten::sigmoid", "aten::sigmoid_",
        "aten::relu", "aten::relu_",
        "aten::hardswish", "aten::hardswish_",
        "aten::hardsigmoid", "aten::hardsigmoid_",
        "aten::tanh", "aten::tanh_",
        "aten::softmax", "aten::log_softmax",

        # === Element-wise math ===
        "aten::neg", "aten::exp", "aten::log", "aten::flip",
        "aten::add", "aten::add_", "aten::sub", "aten::sub_", "aten::rsub",
        "aten::mul", "aten::mul_", "aten::div", "aten::div_",
        "aten::sum", "aten::mean",
        "aten::clamp", "aten::clamp_", "aten::clamp_min", "aten::clamp_max",
        "aten::abs", "aten::sqrt", "aten::rsqrt",
        "aten::pow",

        # === Shape/memory ops (FLOPs 0) ===
        "aten::pad", "aten::constant_pad_nd",
        "aten::expand", "aten::expand_as",
        "aten::view", "aten::reshape", "aten::permute", "aten::transpose",
        "aten::cat", "aten::stack", "aten::split", "aten::chunk",
        "aten::contiguous", "aten::clone",
        "aten::unsqueeze", "aten::squeeze",

        # === Pooling ===
        "aten::avg_pool2d", "aten::adaptive_avg_pool2d",
        "aten::max_pool2d", "aten::adaptive_max_pool2d",
        "aten::max_pool2d_with_indices",

        # === Mamba cross-scan (메모리 op) ===
        "prim::PythonOp.CrossScanTritonFk2",
        "prim::PythonOp.CrossMergeTritonFk2",
        "prim::PythonOp.CrossScan",
        "prim::PythonOp.CrossMerge",
    ]:
        flops.set_op_handle(op, None)

    # ── 측정 ──
    params = parameter_count(model)[""]
    total_flops = flops.total()
    by_op = dict(flops.by_operator())
    unsupported = flops.unsupported_ops()

    # ── op-type breakdown 출력 ──
    print(f'\nInput resolution: {H}×{W}')
    print(f'Convention: inference-time MACs (BN/LN folded, activations excluded)')
    print('Per-op-type breakdown:')
    total_g = total_flops / 1e9
    for op, val in sorted(by_op.items(), key=lambda x: -x[1]):
        val_g = val / 1e9
        pct = val_g / total_g * 100 if total_g > 0 else 0
        print(f'  {op:30s} {val_g:8.3f} G ({pct:5.1f}%)')
    print(f'  {"TOTAL":30s} {total_g:8.3f} G')

    if show_module_breakdown:
        print('\nPer-module breakdown (top 15):')
        by_mod = dict(flops.by_module())
        top = sorted(by_mod.items(), key=lambda x: -x[1])[:15]
        for name, val in top:
            if val > 0:
                print(f'  {name:50s} {val/1e9:8.3f} G')

    def fmt(n):
        for unit, div in [("G", 1e9), ("M", 1e6), ("K", 1e3)]:
            if n >= div:
                return f"{n/div:.3f}{unit}"
        return f"{n:.3f}"

    print('========================')
    print(f'Computational complexity: {fmt(total_flops)}')
    print(f'Number of parameters    : {fmt(params)}')
    if unsupported:
        print(f'[WARN] Still unsupported: {dict(unsupported)}')
    print('========================\n')

    del model, dummy_input
    return fmt(total_flops), fmt(params)

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "eresnet":
    cfg = cfg_eresnet

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = cfg['lr']
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = WeaveFace(cfg=cfg)
print("Printing net...")
calculate_flops(net)

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

# cudnn.benchmark = True

if cfg['optim'] == 'adamw':
    optimizer = torch.optim.AdamW(net.parameters(), cfg['lr'])
else:
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)

scheduler_data = WiderFaceDetection(training_dataset,preproc(img_dim, rgb_mean))
scheduler_dataloader = data.DataLoader(scheduler_data, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate, worker_init_fn=seed_worker, generator=g) 

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=initial_lr, steps_per_epoch=len(scheduler_dataloader), epochs = max_epoch)

criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train():
    
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset,preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    epoch_loss = 0.0
    best_loss = float('inf')
    
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    print('start model')
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator

            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate, worker_init_fn=seed_worker, generator=g))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time() 
        if iteration in stepvalues:
            step_index += 1

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        out = net(images)
               
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 에폭 평균을 구하기 위해 loss 누적
        epoch_loss = epoch_loss + loss.item()

        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        lr = optimizer.param_groups[0]['lr']
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
        
        # --- 추가된 최소 Loss 저장 로직 ---
        # 1 에폭이 끝나는 시점에 도달했을 때
        if (iteration + 1) % epoch_size == 0:
            avg_epoch_loss = epoch_loss / epoch_size
            # print(f'*** Epoch {epoch} Average Loss: {avg_epoch_loss:.4f} ***')
            
            # 현재 에폭의 평균 Loss가 역대 최소 Loss보다 낮을 경우
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                # print(f'>>> 새로운 최소 Loss 달성! 모델을 저장합니다: {best_loss:.4f}')
                torch.save(net.state_dict(), save_folder + cfg['name'] + '_best.pth')
            
            # 다음 에폭을 위해 누적 변수 초기화
            epoch_loss = 0.0

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')

if __name__ == '__main__':
    train()
