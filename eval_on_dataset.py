import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
import lightning.pytorch as pl
import pandas as pd

from utils.utils import *
from utils.paths import *

from vision_tower import VGGT_OriAny_Ref
from utils.data import COCO_Bench_VGGT, coco_class_map, Abs_Ori_Bench_VGGT, Ref_Pose_Val, Omni6DPose_Bench_VGGT

def validate_coco_8dir(labels, az_preds, alpha_preds, cls_s):
    device = labels.device
    
    mask = labels != 8
    labels = labels[mask]
    az_preds = az_preds[mask]
    alpha_preds = alpha_preds[mask]
    cls_s = cls_s[mask]
    
    if len(labels) == 0:
        return torch.tensor(0.0), {'acc': torch.tensor(0.0)}

    preds_0 = (((az_preds + 22.5) % 360) // 45).to(torch.int)

    preds_180 = (preds_0 + 4) % 8
    
    preds_90 = (preds_0 + 2) % 8
    
    preds_270 = (preds_0 + 6) % 8

    is_correct = (labels == preds_0)
    
    mask_a2 = (alpha_preds == 2)
    if torch.any(mask_a2):
        is_correct[mask_a2] = (labels[mask_a2] == preds_0[mask_a2]) | (labels[mask_a2] == preds_180[mask_a2])

    mask_a4 = (alpha_preds == 4)
    if torch.any(mask_a4):
        is_correct[mask_a4] = (
            (labels[mask_a4] == preds_0[mask_a4]) |
            (labels[mask_a4] == preds_90[mask_a4]) |
            (labels[mask_a4] == preds_180[mask_a4]) |
            (labels[mask_a4] == preds_270[mask_a4])
        )
        
    correct_count = torch.sum(is_correct)
    total_count = len(labels)
    acc = correct_count / total_count if total_count != 0 else torch.tensor(0.0)
    acc = acc.cpu()

    detail = {}
    unique_classes = torch.unique(cls_s).cpu().tolist()

    for c in unique_classes:
        class_mask = (cls_s == c)
        correct_in_class = is_correct[class_mask]
        correct_cls_count = torch.sum(correct_in_class)
        cls_count = len(correct_in_class)
        
        cls_accuracy = correct_cls_count / cls_count if cls_count != 0 else torch.tensor(0.0)
        detail[coco_class_map[c]] = cls_accuracy.cpu()

    detail['acc'] = acc
    return acc, detail

def validate_absolute_R_err(az_preds, el_preds, ro_preds, alpha_preds, mode='arkitscenes'):
    if mode == 'arkitscenes':
        annos = pd.read_csv(ARK_META)
        az_label   = torch.tensor(list(annos['azi']))
        el_label   = torch.tensor(list(annos['ele']))
        ro_label   = torch.tensor(list(annos['rot']))
    elif mode == 'sunrgbd':
        annos = pd.read_csv(SUNRGBD_META)
        az_label   = torch.tensor(list(annos['azi']))
        el_label   = torch.tensor(list(annos['ele']))
        ro_label   = torch.tensor(list(annos['rot']))
    elif mode == 'objectron':
        annos = pd.read_csv(OBJECTRON_META)
        az_label   = torch.tensor(list(annos['azi']))
        el_label   = torch.tensor(list(annos['ele']))
        ro_label   = torch.tensor(list(annos['rot']))
    else:
        assert False, f"no such mode: {mode}"
    
    labels = torch.cat([az_label.unsqueeze(1), el_label.unsqueeze(1), ro_label.unsqueeze(1)], dim=1)
    labels = labels.float().to(az_preds.device)

    # === 新增逻辑：根据 alpha_preds 处理对称性 ===

    # 1. 计算原始预测的误差 (azi + 0)
    predictions_0 = torch.cat([az_preds.unsqueeze(1), el_preds.unsqueeze(1), ro_preds.unsqueeze(1)], dim=1)
    errs_0 = rotation_err(predictions_0, labels)
    
    # 初始化最终误差为原始误差
    final_errs = errs_0.clone()
    
    # 2. 处理 alpha=2 和 alpha=4 的情况
    # 为了向量化计算，我们统一计算所有对称情况的误差，然后根据alpha值选择
    
    # 计算 azi + 180 度的误差
    az_preds_180 = (az_preds + 180) % 360
    predictions_180 = torch.cat([az_preds_180.unsqueeze(1), el_preds.unsqueeze(1), ro_preds.unsqueeze(1)], dim=1)
    errs_180 = rotation_err(predictions_180, labels)
    
    # 对于 alpha=2 的情况，取 0 度和 180 度误差中的较小值
    mask_a2 = (alpha_preds == 2)
    if torch.any(mask_a2):
        final_errs[mask_a2] = torch.minimum(errs_0[mask_a2], errs_180[mask_a2])
        
    # 3. 处理 alpha=4 的情况
    mask_a4 = (alpha_preds == 4)
    if torch.any(mask_a4):
        # 计算 azi + 90 和 azi + 270 度的误差
        az_preds_90 = (az_preds + 90) % 360
        predictions_90 = torch.cat([az_preds_90.unsqueeze(1), el_preds.unsqueeze(1), ro_preds.unsqueeze(1)], dim=1)
        errs_90 = rotation_err(predictions_90, labels)
        
        az_preds_270 = (az_preds + 270) % 360
        predictions_270 = torch.cat([az_preds_270.unsqueeze(1), el_preds.unsqueeze(1), ro_preds.unsqueeze(1)], dim=1)
        errs_270 = rotation_err(predictions_270, labels)
        
        # 对于 alpha=4 的情况，取四种旋转角度误差中的最小值
        # 我们已经有了 errs_0 和 errs_180
        min_err_for_a4 = torch.stack([
            errs_0[mask_a4],
            errs_90[mask_a4],
            errs_180[mask_a4],
            errs_270[mask_a4]
        ]).min(dim=0).values
        
        final_errs[mask_a4] = min_err_for_a4

    # 将最终计算出的最小误差转换为 numpy 数组
    test_errs = final_errs.cpu().numpy()
    
    # === 后续统计逻辑保持不变 ===
    
    # === 全体准确率 ===
    Acc_all = np.mean(test_errs <= 30)
    Med_all = np.median(test_errs)

    # === 按 cate_name 分类准确率 ===
    # cate_names = annos['cate_name'].unique()
    # Acc_per_cate = {}
    # for cate in cate_names:
    #     mask = (annos['cate_name'] == cate).values
    #     if np.any(mask):
    #         Acc_per_cate[cate] = np.mean(test_errs[mask] <= 30)

    Acc_per_meta_cate = {}
    # === 按 meta_cate 分类准确率 ===
    if mode == 'imagenet3d':
        meta_cates = annos['meta_cate'].unique()
        for meta_cate in meta_cates:
            mask = (annos['meta_cate'] == meta_cate).values
            if np.any(mask):
                Acc_per_meta_cate[meta_cate] = np.mean(test_errs[mask] <= 30)

    Acc_per_meta_cate[f'{mode}_All'] = Acc_all
    Acc_per_meta_cate[f'{mode}_Med'] = Med_all
    
    return Acc_per_meta_cate

def validate_relate_R_err(az_preds, el_preds, ro_preds, alpha_preds, mode='onepose++'):
    if mode == 'onepose':
        annos = torch.load(ONEPOSE_META, weights_only=False)
    elif mode == 'onepose++':
        annos = torch.load(ONEPOSEPP_META, weights_only=False)
    elif mode == 'ycbv':
        annos = torch.load(YCBV_META, weights_only=False)
    elif mode == 'linemod':
        annos = torch.load(LINEMOD_META, weights_only=False)
    elif mode == 'onepose_random':
        annos = torch.load(ONEPOSE_RANDOM_META, weights_only=False)
    elif mode == 'onepose++_random':
        annos = torch.load(ONEPOSEPP_RANDOM_META, weights_only=False)
    elif mode == 'ycbv_random':
        annos = torch.load(YCBV_RANDOM_META, weights_only=False)
    elif mode == 'linemod_random':
        annos = torch.load(LINEMOD_RANDOM_META, weights_only=False)
    else:
        assert False, f"no such mode: {mode}"

    R_gt = torch.stack(annos['Mrel'], 0)[:, :3, :3].reshape(-1, 9).float().to(az_preds.device)

    # --- 根据 alpha_preds 处理对称性 ---
    
    # 辅助函数，用于计算给定旋转矩阵和真值之间的误差
    def calculate_R_error_tensor(R_pred, R_gt_flat):
        R_pred_flat = R_pred.reshape(-1, 9)
        # 确保 R_pred_flat 和 R_gt_flat 的样本数匹配
        if R_pred_flat.shape[0] != R_gt_flat.shape[0]:
            raise ValueError("Prediction and Ground Truth have different number of samples.")
            
        # Geodesic distance calculation
        dot_product = torch.sum(R_pred_flat * R_gt_flat, 1)
        # Clamp to avoid numerical issues with acos
        clamped_val = torch.clamp((dot_product - 1.) / 2., -1.0, 1.0)
        R_err_rad = torch.acos(clamped_val)
        return R_err_rad * 180. / torch.pi

    # 1. 计算原始预测的误差 (azi + 0)
    R_pred_0 = relate_azi_ele_rot_to_OneposeRMatrix(az_preds, el_preds, ro_preds)
    R_err_0 = calculate_R_error_tensor(R_pred_0, R_gt)
    
    # 初始化最终误差为原始误差
    final_R_err = R_err_0.clone()

    # 2. 处理 alpha=2 的情况 (180度对称)
    mask_a2 = (alpha_preds == 2)
    if torch.any(mask_a2):
        az_preds_180 = (az_preds + 180) % 360
        R_pred_180 = relate_azi_ele_rot_to_OneposeRMatrix(az_preds_180, el_preds, ro_preds)
        R_err_180 = calculate_R_error_tensor(R_pred_180, R_gt)
        
        # 对于alpha=2的样本，取0度和180度误差中的较小值
        final_R_err[mask_a2] = torch.minimum(R_err_0[mask_a2], R_err_180[mask_a2])

    # 3. 处理 alpha=4 的情况 (90度对称)
    mask_a4 = (alpha_preds == 4)
    if torch.any(mask_a4):
        # 计算 azi + 90, 180, 270 度的误差
        # 我们需要为alpha=4重新计算180度的误差，以确保维度匹配
        az_preds_90 = (az_preds + 90) % 360
        R_pred_90 = relate_azi_ele_rot_to_OneposeRMatrix(az_preds_90, el_preds, ro_preds)
        R_err_90 = calculate_R_error_tensor(R_pred_90, R_gt)
        
        az_preds_180_a4 = (az_preds + 180) % 360
        R_pred_180_a4 = relate_azi_ele_rot_to_OneposeRMatrix(az_preds_180_a4, el_preds, ro_preds)
        R_err_180_a4 = calculate_R_error_tensor(R_pred_180_a4, R_gt)

        az_preds_270 = (az_preds + 270) % 360
        R_pred_270 = relate_azi_ele_rot_to_OneposeRMatrix(az_preds_270, el_preds, ro_preds)
        R_err_270 = calculate_R_error_tensor(R_pred_270, R_gt)
        
        # 对于alpha=4的样本，取四个旋转角度误差中的最小值
        min_err_for_a4 = torch.stack([
            R_err_0[mask_a4],
            R_err_90[mask_a4],
            R_err_180_a4[mask_a4],
            R_err_270[mask_a4]
        ]).min(dim=0).values
        
        final_R_err[mask_a4] = min_err_for_a4
        
    # --- 后续统计逻辑基于最终的最小误差 ---
    R_err = final_R_err.cpu().numpy()
    
    # === 全体准确率 ===
    Acc30 = np.mean(R_err <= 30)
    Acc15 = np.mean(R_err <= 15)
    Med_Err = np.median(R_err)
    
    # === 按 cate 分类准确率 ===
    meta_cates = set(annos['cate_name'])
    Acc_per_meta_cate = {}
    cates = np.array(annos['cate_name'])
    for meta_cate in meta_cates:
        mask = (cates == meta_cate)
        if np.any(mask):
            # Acc_per_meta_cate[f'{mode}_{meta_cate}'] = np.mean(R_err[mask] <= 30)
            Acc_per_meta_cate[f'{mode}_{meta_cate}_MedErr'] = np.median(R_err[mask])
            
    Acc_per_meta_cate[f'{mode}_Acc30'] = Acc30
    Acc_per_meta_cate[f'{mode}_Acc15'] = Acc15
    Acc_per_meta_cate[f'{mode}_MedErr'] = Med_Err
    
    return Acc_per_meta_cate

def validate_symm(alpha, mode='omni6dpose'):
    if mode == 'omni6dpose':
        annos = pd.read_csv(OMNI6DPOSE_META_BAL)
    else:
        assert False, f"no such mode: {mode}"
    
    labels = torch.tensor(list(annos['dir_num'])).to(alpha.device)
    acc    = (alpha == labels).sum()/len(labels)
    
    return acc


class Projector(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        self.mark_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        # self.mark_dtype = torch.float16
        print(self.mark_dtype)
        self.model = VGGT_OriAny_Ref(
                        out_dim     = self.cfg.label_dim,
                        dtype       = self.mark_dtype,
                        nopretrain  = True
                    )

        self.image_dir_dict = val_res_dict
        self.val_len = val_len_dict
    
    def on_validation_epoch_start(self):
        self.image_dir_dict = val_res_dict
        
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        ori_val_loaders = [8,9,10,11,12]
        ref_val_loaders = [0,1,2,3,4,5,6,7]

        device = self.model.get_device()
        batch_img_inputs = batch # (B, S, 3, H, W)

        # print(batch_img_inputs.shape)
        B, S, C, H, W = batch_img_inputs.shape
        # print(batch_img_inputs.shape)
        pose_enc = self.model(batch_img_inputs) # (B, S, D) S = 1
        
        pose_enc = pose_enc.view(B*S, -1)
        
        angle_az_pred = torch.argmax(pose_enc[:, 0:360]       , dim=-1)
        angle_el_pred = torch.argmax(pose_enc[:, 360:360+180] , dim=-1) - 90
        angle_ro_pred = torch.argmax(pose_enc[:, 360+180:360+180+360] , dim=-1) - 180
        
        distribute = F.sigmoid(pose_enc[:, 0:360]).cpu().float().numpy()

        alpha_pred = val_fit_alpha(distribute = distribute)

        if dataloader_idx in ori_val_loaders:
            if dataloader_idx==8:
                self.image_dir_dict['ark_az'].append(angle_az_pred)
                self.image_dir_dict['ark_el'].append(angle_el_pred)
                self.image_dir_dict['ark_ro'].append(angle_ro_pred)
                self.image_dir_dict['ark_alpha'].append(alpha_pred)
            if dataloader_idx==9:
                self.image_dir_dict['sunrgbd_az'].append(angle_az_pred)
                self.image_dir_dict['sunrgbd_el'].append(angle_el_pred)
                self.image_dir_dict['sunrgbd_ro'].append(angle_ro_pred)
                self.image_dir_dict['sunrgbd_alpha'].append(alpha_pred)
            if dataloader_idx==10:
                self.image_dir_dict['objtron_az'].append(angle_az_pred)
                self.image_dir_dict['objtron_el'].append(angle_el_pred)
                self.image_dir_dict['objtron_ro'].append(angle_ro_pred)
                self.image_dir_dict['objtron_alpha'].append(alpha_pred)
            if dataloader_idx==11:
                self.image_dir_dict['coco_az'].append(angle_az_pred)
                self.image_dir_dict['coco_alpha'].append(alpha_pred)
            if dataloader_idx==12:
                self.image_dir_dict['om6dp_az'].append(angle_az_pred)
                self.image_dir_dict['om6dp_el'].append(angle_el_pred)
                self.image_dir_dict['om6dp_ro'].append(angle_ro_pred)
                self.image_dir_dict['om6dp_alpha'].append(alpha_pred)

        elif dataloader_idx in ref_val_loaders:
            rel_az_pred = angle_az_pred.reshape(B,S)[:,1]
            rel_el_pred = angle_el_pred.reshape(B,S)[:,1]
            rel_ro_pred = angle_ro_pred.reshape(B,S)[:,1]
            rel_alpha_pred = alpha_pred.reshape(B,S)[:,0]
            
            if dataloader_idx==0:
                self.image_dir_dict['op_az'].append(rel_az_pred)
                self.image_dir_dict['op_el'].append(rel_el_pred)
                self.image_dir_dict['op_ro'].append(rel_ro_pred)
                self.image_dir_dict['op_alpha'].append(rel_alpha_pred)
            if dataloader_idx==1:
                self.image_dir_dict['oppp_az'].append(rel_az_pred)
                self.image_dir_dict['oppp_el'].append(rel_el_pred)
                self.image_dir_dict['oppp_ro'].append(rel_ro_pred)
                self.image_dir_dict['oppp_alpha'].append(rel_alpha_pred)
            if dataloader_idx==2:
                self.image_dir_dict['lm_az'].append(rel_az_pred)
                self.image_dir_dict['lm_el'].append(rel_el_pred)
                self.image_dir_dict['lm_ro'].append(rel_ro_pred)
                self.image_dir_dict['lm_alpha'].append(rel_alpha_pred)
            if dataloader_idx==3:
                self.image_dir_dict['ycbv_az'].append(rel_az_pred)
                self.image_dir_dict['ycbv_el'].append(rel_el_pred)
                self.image_dir_dict['ycbv_ro'].append(rel_ro_pred)
                self.image_dir_dict['ycbv_alpha'].append(rel_alpha_pred)

            
            if dataloader_idx==4:
                self.image_dir_dict['op_rnd_az'].append(rel_az_pred)
                self.image_dir_dict['op_rnd_el'].append(rel_el_pred)
                self.image_dir_dict['op_rnd_ro'].append(rel_ro_pred)
                self.image_dir_dict['op_rnd_alpha'].append(rel_alpha_pred)
            if dataloader_idx==5:
                self.image_dir_dict['oppp_rnd_az'].append(rel_az_pred)
                self.image_dir_dict['oppp_rnd_el'].append(rel_el_pred)
                self.image_dir_dict['oppp_rnd_ro'].append(rel_ro_pred)
                self.image_dir_dict['oppp_rnd_alpha'].append(rel_alpha_pred)
            if dataloader_idx==6:
                self.image_dir_dict['lm_rnd_az'].append(rel_az_pred)
                self.image_dir_dict['lm_rnd_el'].append(rel_el_pred)
                self.image_dir_dict['lm_rnd_ro'].append(rel_ro_pred)
                self.image_dir_dict['lm_rnd_alpha'].append(rel_alpha_pred)
            if dataloader_idx==7:
                self.image_dir_dict['ycbv_rnd_az'].append(rel_az_pred)
                self.image_dir_dict['ycbv_rnd_el'].append(rel_el_pred)
                self.image_dir_dict['ycbv_rnd_ro'].append(rel_ro_pred)
                self.image_dir_dict['ycbv_rnd_alpha'].append(rel_alpha_pred)
           
    def on_validation_epoch_end(self):
        device = self.model.get_device()
        
        for k,v in self.image_dir_dict.items():
            self.image_dir_dict[k] = torch.cat(v)
            
        all_img_out = self.all_gather(self.image_dir_dict)
        # maybe cat
        for k,v in all_img_out.items():
            all_img_out[k] = v.permute(1, 0).reshape(-1)
            all_img_out[k] = all_img_out[k][:self.val_len[k]]
            # print(k, all_img_out[k].shape)
        
        if self.trainer.is_global_zero:
            coco_meta = pd.read_csv(COCO_META)
            coco_dir_label = torch.tensor(list(coco_meta['direction']))
            coco_dir_cls   = torch.tensor(list(coco_meta['class']))
            coco_rec, coco_detail = validate_coco_8dir(coco_dir_label.to(device), 
                                                       all_img_out['coco_az'].to(device), 
                                                       all_img_out['coco_alpha'].to(device), 
                                                       coco_dir_cls.to(device))

            ark_acc_per_meta_cate       = validate_absolute_R_err(all_img_out['ark_az'].to(device), 
                                                              all_img_out['ark_el'].to(device), 
                                                              all_img_out['ark_ro'].to(device),
                                                              all_img_out['ark_alpha'].to(device),
                                                              mode='arkitscenes')
            sunrgbd_acc_per_meta_cate   = validate_absolute_R_err(all_img_out['sunrgbd_az'].to(device), 
                                                              all_img_out['sunrgbd_el'].to(device), 
                                                              all_img_out['sunrgbd_ro'].to(device),
                                                              all_img_out['sunrgbd_alpha'].to(device),
                                                              mode='sunrgbd')
            objtron_acc_per_meta_cate   = validate_absolute_R_err(all_img_out['objtron_az'].to(device), 
                                                              all_img_out['objtron_el'].to(device), 
                                                              all_img_out['objtron_ro'].to(device),
                                                              all_img_out['objtron_alpha'].to(device),
                                                              mode='objectron')
            
            op_acc_per_meta_cate= validate_relate_R_err(all_img_out['op_az'].to(device),
                                                        all_img_out['op_el'].to(device),
                                                        all_img_out['op_ro'].to(device),
                                                        all_img_out['op_alpha'].to(device),
                                                        mode='onepose')
            oppp_acc_per_meta_cate= validate_relate_R_err(all_img_out['oppp_az'].to(device),
                                                          all_img_out['oppp_el'].to(device),
                                                          all_img_out['oppp_ro'].to(device),
                                                          all_img_out['oppp_alpha'].to(device),
                                                          mode='onepose++')
            lm_acc_per_meta_cate= validate_relate_R_err(all_img_out['lm_az'].to(device),
                                                        all_img_out['lm_el'].to(device),
                                                        all_img_out['lm_ro'].to(device),
                                                        all_img_out['lm_alpha'].to(device),
                                                        mode='linemod')
            ycbv_acc_per_meta_cate= validate_relate_R_err(all_img_out['ycbv_az'].to(device),
                                                          all_img_out['ycbv_el'].to(device),
                                                          all_img_out['ycbv_ro'].to(device),
                                                          all_img_out['ycbv_alpha'].to(device),
                                                          mode='ycbv')
            op_rnd_acc_per_meta_cate= validate_relate_R_err(all_img_out['op_rnd_az'].to(device),
                                                        all_img_out['op_rnd_el'].to(device),
                                                        all_img_out['op_rnd_ro'].to(device),
                                                        all_img_out['op_rnd_alpha'].to(device),
                                                        mode='onepose_random')
            oppp_rnd_acc_per_meta_cate= validate_relate_R_err(all_img_out['oppp_rnd_az'].to(device),
                                                          all_img_out['oppp_rnd_el'].to(device),
                                                          all_img_out['oppp_rnd_ro'].to(device),
                                                          all_img_out['oppp_rnd_alpha'].to(device),
                                                          mode='onepose++_random')
            lm_rnd_acc_per_meta_cate= validate_relate_R_err(all_img_out['lm_rnd_az'].to(device),
                                                        all_img_out['lm_rnd_el'].to(device),
                                                        all_img_out['lm_rnd_ro'].to(device),
                                                        all_img_out['lm_rnd_alpha'].to(device),
                                                        mode='linemod_random')
            ycbv_rnd_acc_per_meta_cate= validate_relate_R_err(all_img_out['ycbv_rnd_az'].to(device),
                                                          all_img_out['ycbv_rnd_el'].to(device),
                                                          all_img_out['ycbv_rnd_ro'].to(device),
                                                          all_img_out['ycbv_rnd_alpha'].to(device),
                                                          mode='ycbv_random')
            
            omni6dpose_symm_acc = validate_symm(all_img_out['om6dp_alpha'].to(device), mode='omni6dpose')
            
            val_dict = {}
            print('\n')
            val_dict['coco_rec']     = coco_rec
            print('coco_rec', coco_rec)

            for k,v in ark_acc_per_meta_cate.items():
                val_dict[k] = v
                print(k, v)
            for k,v in sunrgbd_acc_per_meta_cate.items():
                val_dict[k] = v
                print(k, v)
            for k,v in objtron_acc_per_meta_cate.items():
                val_dict[k] = v
                print(k, v)

            for k,v in op_acc_per_meta_cate.items():
                val_dict[k] = v
                print(k, v)
            for k,v in oppp_acc_per_meta_cate.items():
                val_dict[k] = v
                print(k, v)
            for k,v in lm_acc_per_meta_cate.items():
                val_dict[k] = v
                print(k, v)
            for k,v in ycbv_acc_per_meta_cate.items():
                val_dict[k] = v
                print(k, v)

            for k,v in op_rnd_acc_per_meta_cate.items():
                val_dict[k] = v
                print(k, v)
            for k,v in oppp_rnd_acc_per_meta_cate.items():
                val_dict[k] = v
                print(k, v)
            for k,v in lm_rnd_acc_per_meta_cate.items():
                val_dict[k] = v
                print(k, v)
            for k,v in ycbv_rnd_acc_per_meta_cate.items():
                val_dict[k] = v
                print(k, v)
            
            
            val_dict['omni6dpose_symm']     = omni6dpose_symm_acc
            print('omni6dpose_symm', omni6dpose_symm_acc)

        else:
            pass
        
        torch.cuda.empty_cache()
        self.trainer.strategy.barrier()

def main(cfg):
    torch.set_float32_matmul_precision('medium')
    # Create the dataset and data loader
    pl.seed_everything(cfg.seed)
    
    print(cfg)
    pl_model = Projector(cfg)

    ark_val = Abs_Ori_Bench_VGGT(mode='arkitscenes')
    ark_val_loader = DataLoader(ark_val, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    sunrgbd_val = Abs_Ori_Bench_VGGT(mode='sunrgbd')
    sunrgbd_val_loader = DataLoader(sunrgbd_val, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    objtron_val = Abs_Ori_Bench_VGGT(mode='objectron')
    objtron_val_loader = DataLoader(objtron_val, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    
    op_val = Ref_Pose_Val(mode='onepose')
    op_val_loader = DataLoader(op_val, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    
    oppp_val = Ref_Pose_Val(mode='onepose++')
    oppp_val_loader = DataLoader(oppp_val, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    
    op_rnd_val = Ref_Pose_Val(mode='onepose_random')
    op_rnd_val_loader = DataLoader(op_rnd_val, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    
    oppp_rnd_val = Ref_Pose_Val(mode='onepose++_random')
    oppp_rnd_val_loader = DataLoader(oppp_rnd_val, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    
    lm_rnd_val = Ref_Pose_Val(mode='linemod_random')
    lm_rnd_val_loader = DataLoader(lm_rnd_val, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    
    ycbv_rnd_val = Ref_Pose_Val(mode='ycbv_random')
    ycbv_rnd_val_loader = DataLoader(ycbv_rnd_val, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    lm_val = Ref_Pose_Val(mode='linemod')
    lm_val_loader = DataLoader(lm_val, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    
    ycbv_val = Ref_Pose_Val(mode='ycbv')
    ycbv_val_loader = DataLoader(ycbv_val, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    
    coco_image = COCO_Bench_VGGT()
    coco_img_loader = DataLoader(coco_image, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    
    om6dp_val = Omni6DPose_Bench_VGGT(ref=False)
    om6dp_val_loader = DataLoader(om6dp_val, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=False, sampler=None, drop_last=False, num_workers=6
    )
    
    val_loaders = []
    val_loaders += [
                    op_val_loader,
                    oppp_val_loader,
                    lm_val_loader,
                    ycbv_val_loader,
                    op_rnd_val_loader,
                    oppp_rnd_val_loader,
                    lm_rnd_val_loader,
                    ycbv_rnd_val_loader,
                    ark_val_loader,
                    sunrgbd_val_loader,
                    objtron_val_loader,
                    coco_img_loader,
                    om6dp_val_loader,
                    ]

    print('evaluate')
    trainer = pl.Trainer(accelerator="gpu", devices=cfg.devices)

    if os.path.exists(LOCAL_CKPT_PATH):
        ckpt_path = LOCAL_CKPT_PATH
    else:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything-V2", filename=HF_CKPT_PATH, repo_type="model", cache_dir='./', resume_download=True)
    print(ckpt_path)
    pl_model.model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    trainer.validate(pl_model, val_loaders)
    
        
def load_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=44, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--devices", default=8, type=int)
    parser.add_argument("--label_dim", default=360+180+360, type=int)
    
    
    
    return parser

if __name__ == '__main__':
    parser = load_parser()
    args = parser.parse_args()
    main(args)


