import rembg
import random
import torch
import numpy as np
from PIL import Image, ImageOps
import PIL
from typing import Any
import matplotlib.pyplot as plt
import io

def resize_foreground(
    image: Image,
    ratio: float,
) -> Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = Image.fromarray(new_image)
    return new_image

def remove_background(image: Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image

def background_preprocess(input_image, do_remove_background):
    if input_image is None:
        return None
    rembg_session = rembg.new_session() if do_remove_background else None

    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    return input_image

def axis_angle_rotation_batch(axis: torch.Tensor, theta: torch.Tensor, homogeneous: bool = False) -> torch.Tensor:
    """
    支持batch输入的版本：
    Args:
        axis: (3,) or (N,3)
        theta: scalar or (N,)
        homogeneous: 是否输出 4x4 齐次矩阵

    Returns:
        (N,3,3) or (N,4,4)
    """
    axis = torch.as_tensor(axis).float()
    theta = torch.as_tensor(theta).float()

    if axis.ndim == 1:
        axis = axis.unsqueeze(0)  # (1,3)
    if theta.ndim == 0:
        theta = theta.unsqueeze(0)  # (1,)

    N = axis.shape[0]
    
    # normalize axis
    axis = axis / torch.norm(axis, dim=1, keepdim=True)

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    one_minus_cos = 1 - cos_t

    # 公式展开
    rot = torch.zeros((N, 3, 3), dtype=axis.dtype, device=axis.device)
    rot[:, 0, 0] = cos_t + x*x*one_minus_cos
    rot[:, 0, 1] = x*y*one_minus_cos - z*sin_t
    rot[:, 0, 2] = x*z*one_minus_cos + y*sin_t
    rot[:, 1, 0] = y*x*one_minus_cos + z*sin_t
    rot[:, 1, 1] = cos_t + y*y*one_minus_cos
    rot[:, 1, 2] = y*z*one_minus_cos - x*sin_t
    rot[:, 2, 0] = z*x*one_minus_cos - y*sin_t
    rot[:, 2, 1] = z*y*one_minus_cos + x*sin_t
    rot[:, 2, 2] = cos_t + z*z*one_minus_cos

    if homogeneous:
        rot_homo = torch.eye(4, dtype=axis.dtype, device=axis.device).unsqueeze(0).repeat(N, 1, 1)
        rot_homo[:, :3, :3] = rot
        return rot_homo

    return rot

def azi_ele_rot_to_Obj_Rmatrix_batch(azi: torch.Tensor, ele: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """支持batch输入的: (azi, ele, rot) -> R matrix (N,3,3)"""
    # 转成tensor
    azi = torch.as_tensor(azi).float() * torch.pi / 180.
    ele = torch.as_tensor(ele).float() * torch.pi / 180.
    rot = torch.as_tensor(rot).float() * torch.pi / 180.

    # 保证有batch维度
    if azi.ndim == 0:
        azi = azi.unsqueeze(0)
    if ele.ndim == 0:
        ele = ele.unsqueeze(0)
    if rot.ndim == 0:
        rot = rot.unsqueeze(0)

    N = azi.shape[0]
    
    device = azi.device
    dtype = azi.dtype
    
    z0_axis = torch.tensor([0.,0.,1.], device=device, dtype=dtype).expand(N, -1)
    y0_axis = torch.tensor([0.,1.,0.], device=device, dtype=dtype).expand(N, -1)
    x0_axis = torch.tensor([1.,0.,0.], device=device, dtype=dtype).expand(N, -1)
    # print(z0_axis.shape, azi.shape)
    R_azi = axis_angle_rotation_batch(z0_axis, -1 * azi)
    R_ele = axis_angle_rotation_batch(y0_axis, ele)
    R_rot = axis_angle_rotation_batch(x0_axis, rot)

    R_res = R_rot @ R_ele @ R_azi
    return R_res

def Cam_Rmatrix_to_azi_ele_rot_batch(R: torch.Tensor):
    """支持batch输入的: R matrix -> (azi, ele, rot)，角度制 (度)"""
    R = torch.as_tensor(R).float()

    # 如果是(3,3)，补batch维度
    if R.ndim == 2:
        R = R.unsqueeze(0)

    r0 = R[:, :, 0]  # shape (N,3)
    r1 = R[:, :, 1]
    r2 = R[:, :, 2]

    ele = torch.asin(r0[:, 2])  # r0.z
    cos_ele = torch.cos(ele)

    # 创建默认azi、rot
    azi = torch.zeros_like(ele)
    rot = torch.zeros_like(ele)

    # 正常情况
    normal_mask = (cos_ele.abs() >= 1e-6)
    if normal_mask.any():
        azi[normal_mask] = torch.atan2(r0[normal_mask, 1], r0[normal_mask, 0])
        rot[normal_mask] = torch.atan2(-r1[normal_mask, 2], r2[normal_mask, 2])

    # Gimbal lock特殊情况
    gimbal_mask = ~normal_mask
    if gimbal_mask.any():
        # 这里设azi为0
        azi[gimbal_mask] = 0.0
        rot[gimbal_mask] = torch.atan2(-r1[gimbal_mask, 0], r1[gimbal_mask, 1])

    # 弧度转角度
    azi = azi * 180. / torch.pi
    ele = ele * 180. / torch.pi
    rot = rot * 180. / torch.pi

    return azi, ele, rot

def Get_target_azi_ele_rot(azi: torch.Tensor, ele: torch.Tensor, rot: torch.Tensor, rel_azi: torch.Tensor, rel_ele: torch.Tensor, rel_rot: torch.Tensor):
    Rmat0    = azi_ele_rot_to_Obj_Rmatrix_batch(azi = azi    , ele = ele    , rot = rot)
    Rmat_rel = azi_ele_rot_to_Obj_Rmatrix_batch(azi = rel_azi, ele = rel_ele, rot = rel_rot)
    # Rmat_rel = Rmat1 @ Rmat0.permute(0, 2, 1)
    # azi_out, ele_out, rot_out = Cam_Rmatrix_to_azi_ele_rot_batch(Rmat_rel.permute(0, 2, 1))
    
    Rmat1 = Rmat_rel @ Rmat0
    azi_out, ele_out, rot_out = Cam_Rmatrix_to_azi_ele_rot_batch(Rmat1.permute(0, 2, 1))
    
    return azi_out, ele_out, rot_out
