val_res_dict = {
    'coco_az'    : [],
    'coco_alpha' : [],
    'op_az'    : [],
    'op_el'    : [],
    'op_ro'    : [],
    'op_alpha'    : [],
    'oppp_az'    : [],
    'oppp_el'    : [],
    'oppp_ro'    : [],
    'oppp_alpha'    : [],
    'op_rnd_az'      : [],
    'op_rnd_el'      : [],
    'op_rnd_ro'      : [],
    'op_rnd_alpha'   : [],
    'oppp_rnd_az'    : [],
    'oppp_rnd_el'    : [],
    'oppp_rnd_ro'    : [],
    'oppp_rnd_alpha' : [],
    'lm_rnd_az'    : [],
    'lm_rnd_el'    : [],
    'lm_rnd_ro'    : [],
    'lm_rnd_alpha' : [],
    'ycbv_rnd_az'    : [],
    'ycbv_rnd_el'    : [],
    'ycbv_rnd_ro'    : [],
    'ycbv_rnd_alpha' : [],
    'lm_az'    : [],
    'lm_el'    : [],
    'lm_ro'    : [],
    'lm_alpha' : [],
    'ycbv_az'    : [],
    'ycbv_el'    : [],
    'ycbv_ro'    : [],
    'ycbv_alpha' : [],
    'om6dp_az'    : [],
    'om6dp_el'    : [],
    'om6dp_ro'    : [],
    'om6dp_alpha' : [],
    'objtron_az'    : [],
    'objtron_el'    : [],
    'objtron_ro'    : [],
    'objtron_alpha'    : [],
    'ark_az'    : [],
    'ark_el'    : [],
    'ark_ro'    : [],
    'ark_alpha'    : [],
    'sunrgbd_az'    : [],
    'sunrgbd_el'    : [],
    'sunrgbd_ro'    : [],
    'sunrgbd_alpha'    : [],
}

val_len_dict = {
    'coco_az'    :1589,
    'coco_alpha' :1589,
    'op_az'      : 3166,
    'op_el'      : 3166,
    'op_ro'      : 3166,
    'op_alpha'   : 3166,
    'oppp_az'    : 2751,
    'oppp_el'    : 2751,
    'oppp_ro'    : 2751,
    'oppp_alpha' : 2751,
    'lm_az'      : 5796,
    'lm_el'      : 5796,
    'lm_ro'      : 5796,
    'lm_alpha'   : 5796,
    'ycbv_az'    : 1436,
    'ycbv_el'    : 1436,
    'ycbv_ro'    : 1436,
    'ycbv_alpha' : 1436,
    'op_rnd_az'      : 2000,
    'op_rnd_el'      : 2000,
    'op_rnd_ro'      : 2000,
    'op_rnd_alpha'   : 2000,
    'oppp_rnd_az'    : 1800,
    'oppp_rnd_el'    : 1800,
    'oppp_rnd_ro'    : 1800,
    'oppp_rnd_alpha' : 1800,
    'lm_rnd_az'      : 2600,
    'lm_rnd_el'      : 2600,
    'lm_rnd_ro'      : 2600,
    'lm_rnd_alpha'   : 2600,
    'ycbv_rnd_az'    : 1800,
    'ycbv_rnd_el'    : 1800,
    'ycbv_rnd_ro'    : 1800,
    'ycbv_rnd_alpha' : 1800,
    'om6dp_az'    : 419,
    'om6dp_el'    : 419,
    'om6dp_ro'    : 419,
    'om6dp_alpha' : 419,
    'objtron_az'    : 10768,
    'objtron_el'    : 10768,
    'objtron_ro'    : 10768,
    'objtron_alpha' : 10768,
    'ark_az'    : 36760,
    'ark_el'    : 36760,
    'ark_ro'    : 36760,
    'ark_alpha' : 36760,
    'sunrgbd_az'    : 31098,
    'sunrgbd_el'    : 31098,
    'sunrgbd_ro'    : 31098,
    'sunrgbd_alpha' : 31098,
}

import torch
import torch.nn.functional as F
import numpy as np
    
def angles_to_matrix(angles):
    """Compute the rotation matrix from euler angles for a mini-batch.
    This is a PyTorch implementation computed by myself for calculating
    R = Rz(inp) Rx(ele - pi/2) Rz(-azi)
    
    For the original numpy implementation in StarMap, you can refer to:
    https://github.com/xingyizhou/StarMap/blob/26223a6c766eab3c22cddae87c375150f84f804d/tools/EvalCls.py#L20
    """
    azi = angles[:, 0]
    ele = angles[:, 1]
    rol = angles[:, 2]
    element1 = (torch.cos(rol) * torch.cos(azi) - torch.sin(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element2 = (torch.sin(rol) * torch.cos(azi) + torch.cos(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element3 = (torch.sin(ele) * torch.sin(azi)).unsqueeze(1)
    element4 = (-torch.cos(rol) * torch.sin(azi) - torch.sin(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element5 = (-torch.sin(rol) * torch.sin(azi) + torch.cos(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element6 = (torch.sin(ele) * torch.cos(azi)).unsqueeze(1)
    element7 = (torch.sin(rol) * torch.sin(ele)).unsqueeze(1)
    element8 = (-torch.cos(rol) * torch.sin(ele)).unsqueeze(1)
    element9 = (torch.cos(ele)).unsqueeze(1)
    return torch.cat((element1, element2, element3, element4, element5, element6, element7, element8, element9), dim=1)

def rotation_err(preds, targets):
    """compute rotation error for viewpoint estimation"""
    preds = preds.float().clone()
    targets = targets.float().clone()
    preds[:, 2] = preds[:, 2] * 0.
    # print(preds[:10])
    # print((targets[:10] * 180./ np.pi) % 360.)
    # get elevation and inplane-rotation in the right format
    # change degrees to radians
    # R = Rz(inp) Rx(ele - pi/2) Rz(-azi)
    # preds[:, 1] = preds[:, 1] - 180.
    # preds[:, 2] = preds[:, 2] - 180.
    preds   = preds * torch.pi / 180.
    targets = targets * torch.pi / 180.
    
    # targets = targets * np.pi / 180.
    # targets[:, 1] = targets[:, 1] - np.pi
    # targets[:, 2] = targets[:, 2] - np.pi
    
    # get rotation matrix from euler angles
    R_pred = angles_to_matrix(preds)
    R_gt = angles_to_matrix(targets)
    
    # compute the angle distance between rotation matrix in degrees
    R_err = torch.acos(((torch.sum(R_pred * R_gt, 1)).clamp(-1., 3.) - 1.) / 2)
    R_err = R_err * 180. / torch.pi
    return R_err

def rotation_acc(preds, targets, th=30.):
    R_err = rotation_err(preds, targets)
    return 100. * torch.mean((R_err <= th).float())

def angle_err(preds, targets):
    """compute rotation error for viewpoint estimation"""
    errs = torch.abs(preds - targets)
    errs = torch.min(errs, 360. - errs)
    return errs

def normalize(v):
    return v / torch.norm(v, dim=-1, keepdim=True)

def cross_product(a, b):
    return torch.cross(a, b, dim=-1)

def look_at_rot(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])) -> torch.Tensor:
    """
    Args:
        eye: Tensor of shape (3,) or (B, 3) - camera position in world coordinates
        target: Tensor of shape (3,) or (B, 3) - look-at target point
        up: Tensor of shape (3,) - up vector
    Returns:
        view_matrix: Tensor of shape (4, 4) or (B, 4, 4)
    """
    f = normalize(eye - target)
    l = normalize(cross_product(up.expand_as(f), f))
    u = cross_product(f, l)

    # Construct rotation matrix
    rot = torch.stack([l, u, f], dim=-1)  # (3, 3) or (B, 3, 3)

    return rot

def angles_to_matrix_onepose(angles):
    azi = angles[:, 0]
    ele = angles[:, 1]
    rol = angles[:, 2]
    
    camera_location_batch = torch.stack([torch.sin(azi) * torch.cos(ele), torch.sin(ele), torch.cos(azi) * torch.cos(ele)]).T

    R_batch = look_at_rot(camera_location_batch * torch.tensor([-1.,-1.,-1.], device=angles.device), 
                          target = torch.zeros_like(camera_location_batch, device=angles.device), 
                          up = torch.tensor([0.0, -1.0, 0.0], device=angles.device)).permute(0, 2, 1)
    # print(R)
    
    return R_batch
    

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

def Get_relate_azi_ele_rot(azi: torch.Tensor, ele: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    if azi.ndim == 1:
        Rmat0 = azi_ele_rot_to_Obj_Rmatrix_batch(azi = azi[0] , ele = ele[0] , rot = rot[0])
        Rmat1 = azi_ele_rot_to_Obj_Rmatrix_batch(azi = azi[1] , ele = ele[1] , rot = rot[1])
    else:
        Rmat0 = azi_ele_rot_to_Obj_Rmatrix_batch(azi = azi[:,0] , ele = ele[:,0] , rot = rot[:,0])
        Rmat1 = azi_ele_rot_to_Obj_Rmatrix_batch(azi = azi[:,1] , ele = ele[:,1] , rot = rot[:,1])
    Rmat_rel = Rmat1 @ Rmat0.permute(0, 2, 1)
    azi_out, ele_out, rot_out = Cam_Rmatrix_to_azi_ele_rot_batch(Rmat_rel.permute(0, 2, 1))
    
    return azi_out, ele_out, rot_out

def relate_azi_ele_rot_to_OneposeRMatrix(azi: torch.Tensor, ele: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    device = azi.device
    R_trans_axis_left  = torch.tensor([[0.,1.,0.],[0.,0.,-1.],[-1.,0.,0.]]).to(device)
    R_trans_axis_right = torch.tensor([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]]).to(device)
    Rmat_rel_val = azi_ele_rot_to_Obj_Rmatrix_batch(azi, ele, rot)
    aligned_Rmat_rel = R_trans_axis_left @ Rmat_rel_val @ R_trans_axis_left.T
    return aligned_Rmat_rel

# from vggt utils rotation
def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)

def compute_quat_batch_seq(azi: torch.Tensor, ele: torch.Tensor, rot: torch.Tensor, device) -> torch.Tensor:
    Rmat = azi_ele_rot_to_Obj_Rmatrix_batch(azi, ele, rot).to(device)
    quat = mat_to_quat(Rmat)
    return quat

def quat_to_angles_batch(quat: torch.Tensor) -> torch.Tensor:
    Rmat = quat_to_mat(quat)
    azi_out, ele_out, rot_out = Cam_Rmatrix_to_azi_ele_rot_batch(Rmat.permute(0, 2, 1))
    return azi_out, ele_out, rot_out

def Omni6D_azi_ele_rot_to_Obj_Rmatrix_batch(azi: torch.Tensor, ele: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
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

    R_azi = axis_angle_rotation_batch(y0_axis, -1 * azi)
    R_ele = axis_angle_rotation_batch(z0_axis, -1 * ele)
    R_rot = axis_angle_rotation_batch(x0_axis, -1 * rot)

    R_res = R_rot @ R_ele @ R_azi
    
    R_trans_axis_left  = torch.tensor([[0.,0.,-1.],[0.,-1.,0.],[-1.,0.,0.]])
    R_res = R_trans_axis_left @ R_res
    
    return R_res

def Omni6D_Cam_Rmatrix_to_azi_ele_rot_batch(R: torch.Tensor):
    """支持batch输入的: R matrix -> (azi, ele, rot)，角度制 (度)"""
    R = torch.as_tensor(R).float()

    # 如果是(3,3)，补batch维度
    if R.ndim == 2:
        R = R.unsqueeze(0)

    R_trans_axis_left  = torch.tensor([[0.,0.,-1.],[0.,-1.,0.],[-1.,0.,0.]])
    R = R @ R_trans_axis_left
    
    r0 = R[:, :, 0]  # shape (N,3)
    r1 = R[:, :, 1]
    r2 = R[:, :, 2]

    ele = torch.asin(r0[:, 1])  # r0.y
    cos_ele = torch.cos(ele)

    # 创建默认azi、rot
    azi = torch.zeros_like(ele)
    rot = torch.zeros_like(ele)

    # 正常情况
    normal_mask = (cos_ele.abs() >= 1e-6)
    if normal_mask.any():
        azi[normal_mask] = torch.atan2(-r0[normal_mask, 2], r0[normal_mask, 0])  # -r0.z / r0.x
        rot[normal_mask] = torch.atan2(-r2[normal_mask, 1], r1[normal_mask, 1])  # -r2.y / r1.y

    # Gimbal lock特殊情况
    gimbal_mask = ~normal_mask
    if gimbal_mask.any():
        # 这里设azi为0
        azi[gimbal_mask] = 0.0
        rot[gimbal_mask] = torch.atan2(r2[gimbal_mask, 0], r2[gimbal_mask, 2])

    # 弧度转角度
    azi = azi * 180. / torch.pi
    ele = ele * 180. / torch.pi
    rot = rot * 180. / torch.pi

    return azi, ele, rot

from scipy.special import i0
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from functools import partial

def von_mises_pdf_alpha_numpy(alpha, x, mu, kappa):
    normalization = 2 * np.pi
    pdf = np.exp(kappa * np.cos(alpha * (x - mu))) / normalization
    return pdf

def val_fit_alpha(distribute):
    fit_alphas = []
    for y_noise in distribute:
        x = np.linspace(0, 2 * np.pi, 360)
        y_noise /= trapezoid(y_noise, x) + 1e-8
        
        initial_guess = [x[np.argmax(y_noise)], 1]

        alphas = [1.0, 2.0, 4.0]
        saved_params = []
        saved_r_squared = []

        for alpha in alphas:
            try:
                von_mises_pdf_alpha_partial = partial(von_mises_pdf_alpha_numpy, alpha)
                params, covariance = curve_fit(von_mises_pdf_alpha_partial, x, y_noise, p0=initial_guess)

                residuals = y_noise - von_mises_pdf_alpha_partial(x, *params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_noise - np.mean(y_noise))**2)
                r_squared = 1 - (ss_res / (ss_tot+1e-8))

                saved_params.append(params)
                saved_r_squared.append(r_squared)
                if r_squared > 0.8:
                    break
            except:
                saved_params.append((0.,0.))
                saved_r_squared.append(0.)

        max_index = np.argmax(saved_r_squared)
        alpha = alphas[max_index]
        mu_fit, kappa_fit = saved_params[max_index]
        r_squared = saved_r_squared[max_index]
        
        if alpha == 1. and kappa_fit>=0.6 and r_squared>=0.5:
            pass
        elif alpha == 2. and kappa_fit>=0.45 and r_squared>=0.45:
            pass
        elif alpha == 4. and kappa_fit>=0.25 and r_squared>=0.45:
            pass
        else:
            alpha=0.
        fit_alphas.append(alpha)
    return torch.tensor(fit_alphas)
