from PIL import Image, ImageDraw

import torch
import torch.utils.data as data
import os
import numpy as np
import random
import pandas as pd

from utils.paths import *
from utils.utils import *
import random

from vggt.utils.load_fn import load_and_preprocess_images
from torchvision import transforms as TF

coco_class_map = [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

def resize_image_to_short_side_512(image):
    # image = Image.open(image_path)
    width, height = image.size
    if width >= 512 and height >= 512:
        return image

    # 计算缩放比例，将短边缩放至512像素
    if width < height:
        new_width = 512
        new_height = int((512 / width) * height)
    else:
        new_height = 512
        new_width = int((512 / height) * width)

    resized_image = image.resize((new_width, new_height), Image.BILINEAR)

    return resized_image

def crop_to_object_pil(image, background_color="dark"):
    # image = Image.open(image_path)
    image_np = np.array(image)

    # 创建掩膜，识别出物体（非背景）
    if background_color == "dark":
        mask = np.all(image_np > np.array([5, 5, 5]), axis=-1)  # 检测非黑色
    elif background_color == "white":
        mask0 = np.all(image_np < np.array([202, 202, 202]), axis=-1)  # 检测非灰色
        mask1 = np.all(image_np > np.array([210, 210, 210]), axis=-1)  # 检测非灰色
        mask  = mask0 | mask1  # 检测非灰色
    # 找到物体的边界
    coords = np.argwhere(mask)

    if coords.any():
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1  # 加1是因为切片时上限不包括
        # print(y_min, x_min, y_max, x_max)
        # 裁切图像
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        return cropped_image
    else:
        return image  # 如果未找到物体，返回原始图像

def rotate_and_fill(image, angle, background_color='white'):
    # 将图像转换为 RGBA 模式，这样可以指定背景颜色
    image = image.convert('RGBA')
    # 旋转图片，逆时针旋转，扩展图像尺寸以避免裁切
    if background_color=='white':
        rotated_image = image.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=(206, 206, 206, 255))
        # 创建一个新的背景图像，背景色为 [206, 206, 206]，并确保它是 RGB 模式
        new_background = Image.new("RGBA", rotated_image.size, (206, 206, 206, 255))
    else:
        rotated_image = image.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=(1, 1, 1, 255))
        new_background = Image.new("RGBA", rotated_image.size, (1, 1, 1, 255))
    # 将旋转后的图像粘贴到新图像上
    new_background.paste(rotated_image, (0, 0), rotated_image)

    # 最终，如果需要，可以转换回 RGB（没有透明度通道）
    final_image = new_background.convert('RGB')

    return final_image

def load_and_preprocess_images_rotation(image_path_list, rotations, bkgs, mode="crop", mask_aug=False, min_mask_ratio=0.1, max_mask_ratio=0.25, min_mask_num=0, max_mask_num=2):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
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
    for image_path, angle, bkg in zip(image_path_list, rotations, bkgs):
        
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        if angle != 0.:
            img = rotate_and_fill(img, angle, background_color=bkg)
        
        # if mask_aug and mode == "crop":
        if mask_aug:
            draw = ImageDraw.Draw(img)
            width, height = img.size
            num_masks = random.randint(min_mask_num, max_mask_num)
            for _ in range(num_masks):
                mask_w = random.randint(int(min_mask_ratio * width) , int(max_mask_ratio * width))
                mask_h = random.randint(int(min_mask_ratio * height), int(max_mask_ratio * height))
                x0 = random.randint(0, width - mask_w)
                y0 = random.randint(0, height - mask_h)
                draw.rectangle([x0, y0, x0 + mask_w, y0 + mask_h], fill=(1, 1, 1))  # 黑色遮挡
        
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
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

class COCO_Bench_VGGT(data.Dataset):
    def __init__(self):
        self.image_root = COCO_ROOT
        self.meta       = pd.read_csv(COCO_META)
        # self.preprocess = preprocess
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        # print(index)
        image_name   = os.path.join(self.image_root, str(self.meta['class'][index]), '{0}.jpg'.format(self.meta['uid'][index]))
        image        = load_and_preprocess_images_rotation([image_name], [torch.tensor(0.)], bkgs=[None], mode="pad")
        # print(image_inputs.keys())
        # print(image.shape)
        return image
        # return torch.ones([1, 3, 518, 518])

class Ref_Pose_Val(data.Dataset):
    def __init__(self, mode='linemod'):
        self.mode       = mode
        if mode == 'linemod':
            self.meta       = torch.load(LINEMOD_META, weights_only=False)
            self.meta['ref_related_path'] = [p.replace('color', 'merge') for p in self.meta['ref_related_path']]
            self.meta['tgt_related_path'] = [p.replace('color', 'merge') for p in self.meta['tgt_related_path']]
        elif mode == 'onepose':
            self.meta       = torch.load(ONEPOSE_META, weights_only=False)
        elif mode == 'onepose++':
            self.meta       = torch.load(ONEPOSEPP_META, weights_only=False)
        elif mode == 'ycbv':
            self.meta       = torch.load(YCBV_META, weights_only=False)
            self.meta['ref_related_path'] = [p.replace('color', 'merge') for p in self.meta['ref_related_path']]
            self.meta['tgt_related_path'] = [p.replace('color', 'merge') for p in self.meta['tgt_related_path']]
        elif mode == 'linemod_random':
            self.meta       = torch.load(LINEMOD_RANDOM_META, weights_only=False)
        elif mode == 'onepose_random':
            self.meta       = torch.load(ONEPOSE_RANDOM_META, weights_only=False)
        elif mode == 'onepose++_random':
            self.meta       = torch.load(ONEPOSEPP_RANDOM_META, weights_only=False)
        elif mode == 'ycbv_random':
            self.meta       = torch.load(YCBV_RANDOM_META, weights_only=False)

        self.image_root = POPE_ROOT
        self.len        = len(self.meta['cate_name'])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        try:
            image_names   = [os.path.join(self.image_root, self.meta['ref_related_path'][index]), os.path.join(self.image_root, self.meta['tgt_related_path'][index])]
            bkgs          = [None, None]
            image         = load_and_preprocess_images_rotation(image_names, torch.zeros(2), bkgs, mode="pad")
            # R_label       = self.meta['Mrel'][index][:3,:3].reshape(-1)
        except Exception as e:
            print(self.meta['ref_related_path'][index], e)
            return self.__getitem__(0)
        # return image_inputs, angle_ax, angle_pl, has_direction
        return image

class Omni6DPose_Bench_VGGT(data.Dataset):
    def __init__(self, ref=False):
        self.ref = ref
        if ref:
            self.meta       = pd.read_csv(OMNI6DPOSE_META_REF)
            self.image_root = OMNI6DPOSE_ROOT
        else:
            self.meta       = pd.read_csv(OMNI6DPOSE_META_BAL)
            self.image_root = OMNI6DPOSE_ROOT

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        # print(index)
        # image_name   = os.path.join(self.image_root, self.meta['relate_path'][index], 'random_dark{0}.jpg'.format(self.meta['img_idx'][index]))
        if self.ref:
            # image_name   = os.path.join(self.image_root, self.meta['relate_path'][index], 'random_dark{0}.jpg'.format(self.meta['img_idx'][index]))
            image_name   = os.path.join(self.image_root, self.meta['relate_path'][index], 'fix_white0.jpg')
            # ref_name     = os.path.join(self.image_root, self.meta['relate_path'][index], 'random_dark{0}.jpg'.format(self.meta['ref_img_idx'][index]))
            # ref_name     = os.path.join(self.image_root, self.meta['relate_path'][index], 'fix_dark0.jpg')
            # image        = load_and_preprocess_images_rotation([image_name, ref_name], torch.tensor([0., 0.]), bkgs=[None, None], mode="pad")
            image        = load_and_preprocess_images_rotation([image_name], [torch.tensor(0.)], bkgs=[None], mode="pad")
        else:
            image_name   = os.path.join(self.image_root, self.meta['relate_path'][index], 'fix_white0.jpg')
            image        = load_and_preprocess_images_rotation([image_name], [torch.tensor(0.)], bkgs=[None], mode="pad")

        return image
 
class Abs_Ori_Bench_VGGT(data.Dataset):
    def __init__(self, mode='objectron'):
        self.mode       = mode
        if mode == 'objectron':
            self.meta       = pd.read_csv(OBJECTRON_META)
            self.image_root = OBJECTRON_ROOT
        elif mode == 'sunrgbd':
            self.meta       = pd.read_csv(SUNRGBD_META)
            self.image_root = SUNRGBD_ROOT
        elif mode == 'arkitscenes':
            self.meta       = pd.read_csv(ARK_META)
            self.image_root = ARK_ROOT   

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        # print(index)
        try:
            image_name   = os.path.join(self.image_root, '{0}.jpg'.format(self.meta['img_idx'][index]))
            image        = load_and_preprocess_images_rotation([image_name], [torch.tensor(0.)], bkgs=[None], mode="pad")
            return image
        except Exception as e:
            print(image_name, e)
            return self.__getitem__(0)

