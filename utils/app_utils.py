import rembg
import random
import torch
import numpy as np
from PIL import Image, ImageOps
from typing import Any

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

from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as TF

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
        
        # support 1,2,4
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
        print(saved_params, saved_r_squared)
        if alpha == 1. and kappa_fit>=0.6 and r_squared>=0.45:
            pass
        elif alpha == 2. and kappa_fit>=0.5 and r_squared>=0.45:
            pass
        elif alpha == 4. and kappa_fit>=0.25 and r_squared>=0.45:
            pass
        else:
            alpha=0.
        fit_alphas.append(alpha)
    return torch.tensor(fit_alphas)

def preprocess_images(image_list, mode="crop"):

    # Check for empty list
    if len(image_list) == 0:
        raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    # for image_path in image_path_list:
    for img in image_list:
        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")
        width, height = img.size
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        try:
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = to_tensor(img)  # Convert to tensor (0, 1)
        except Exception as e:
            print(e)
            print(width, height)
            print(new_width, new_height)
            assert False

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]
        
        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

@torch.no_grad()
def inf_single_batch(model, batch):
    device = model.get_device()
    batch_img_inputs = batch # (B, S, 3, H, W)
    # print(batch_img_inputs.shape)
    B, S, C, H, W = batch_img_inputs.shape
    pose_enc = model(batch_img_inputs) # (B, S, D) S = 1
    
    pose_enc = pose_enc.view(B*S, -1)
    angle_az_pred = torch.argmax(pose_enc[:, 0:360]       , dim=-1)
    angle_el_pred = torch.argmax(pose_enc[:, 360:360+180] , dim=-1) - 90
    angle_ro_pred = torch.argmax(pose_enc[:, 360+180:360+180+360] , dim=-1) - 180
    
    # ori_val
    # trained with BCE loss
    distribute = F.sigmoid(pose_enc[:, 0:360]).cpu().float().numpy()
    # trained with CE loss
    # distribute = pose_enc[:, 0:360].cpu().float().numpy()
    alpha_pred = val_fit_alpha(distribute = distribute)

    # ref_val
    if S > 1:
        ref_az_pred = angle_az_pred.reshape(B,S)[:,0]
        ref_el_pred = angle_el_pred.reshape(B,S)[:,0]
        ref_ro_pred = angle_ro_pred.reshape(B,S)[:,0]
        ref_alpha_pred = alpha_pred.reshape(B,S)[:,0]
        rel_az_pred = angle_az_pred.reshape(B,S)[:,1]
        rel_el_pred = angle_el_pred.reshape(B,S)[:,1]
        rel_ro_pred = angle_ro_pred.reshape(B,S)[:,1]
    else:
        ref_az_pred = angle_az_pred[0]
        ref_el_pred = angle_el_pred[0]
        ref_ro_pred = angle_ro_pred[0]
        ref_alpha_pred = alpha_pred[0]
        rel_az_pred = 0.
        rel_el_pred = 0.
        rel_ro_pred = 0.

    ans_dict = {
        'ref_az_pred': ref_az_pred,
        'ref_el_pred': ref_el_pred,
        'ref_ro_pred': ref_ro_pred,
        'ref_alpha_pred' : ref_alpha_pred,
        'rel_az_pred'  : rel_az_pred,
        'rel_el_pred'  : rel_el_pred,
        'rel_ro_pred'  : rel_ro_pred,
    }
    
    return ans_dict 

# input PIL Image
@torch.no_grad()
def inf_single_case(model, image_ref, image_tgt):
    if image_tgt is None:
        image_list = [image_ref]
    else:
        image_list = [image_ref, image_tgt]
    image_tensors = preprocess_images(image_list, mode="pad").to(model.get_device())
    ans_dict = inf_single_batch(model=model, batch=image_tensors.unsqueeze(0))
    print(ans_dict)
    return ans_dict


