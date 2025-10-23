import torch
from PIL import Image
from utils.app_utils import *
import torch.nn.functional as F
import numpy as np
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
