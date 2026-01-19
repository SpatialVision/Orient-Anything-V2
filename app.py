import gradio as gr
import numpy as np
from PIL import Image
import torch
import tempfile
import os

from utils.paths import *
from vision_tower import VGGT_OriAny_Ref
from utils.app_utils import *
from utils.axis_renderer import BlendRenderer

if os.path.exists(LOCAL_CKPT_PATH):
    ckpt_path = LOCAL_CKPT_PATH
else:
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id="Viglong/OriAnyV2_ckpt", filename=HF_CKPT_PATH, repo_type="model", cache_dir='./', resume_download=True)

mark_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
# device = 'cuda:0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VGGT_OriAny_Ref(out_dim=900, dtype=mark_dtype, nopretrain=True)
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model.eval()
model = model.to(device)
print('Model loaded.')

axis_renderer = BlendRenderer(RENDER_FILE)


# ====== 工具函数：安全图像处理 ======
def safe_image_input(image):
    """确保返回合法的 numpy 数组或 None"""
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return image
    try:
        return np.array(image)
    except Exception:
        return None


# ====== 推理函数 ======
@torch.no_grad()
def run_inference(image_ref, image_tgt, do_rm_bkg):
    image_ref = safe_image_input(image_ref)
    image_tgt = safe_image_input(image_tgt)

    if image_ref is None:
        raise gr.Error("Please upload a reference image before running inference.")

    # 转为 PIL（用于背景去除和后续叠加）
    pil_ref = Image.fromarray(image_ref.astype(np.uint8)).convert("RGB")
    pil_tgt = None

    if image_tgt is not None:
        pil_tgt = Image.fromarray(image_tgt.astype(np.uint8)).convert("RGB")
        if do_rm_bkg:
            pil_ref = background_preprocess(pil_ref, True)
            pil_tgt = background_preprocess(pil_tgt, True)
    else:
        if do_rm_bkg:
            pil_ref = background_preprocess(pil_ref, True)

    try:
        ans_dict = inf_single_case(model, pil_ref, pil_tgt)
    except Exception as e:
        print("Inference error:", e)
        raise gr.Error(f"Inference failed: {str(e)}")

    def safe_float(val, default=0.0):
        try:
            return float(val)
        except:
            return float(default)

    az = safe_float(ans_dict.get('ref_az_pred', 0))
    el = safe_float(ans_dict.get('ref_el_pred', 0))
    ro = safe_float(ans_dict.get('ref_ro_pred', 0))
    alpha = int(ans_dict.get('ref_alpha_pred', 1))  # 注意：target 默认 alpha=1，但 ref 可能不是

    # ===== 用临时文件保存渲染结果 =====
    tmp_ref = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_tgt = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_ref.close()
    tmp_tgt.close()

    try:
        # ===== 渲染参考图的坐标轴 =====
        axis_renderer.render_axis(az, el, ro, alpha, save_path=tmp_ref.name)
        axis_ref = Image.open(tmp_ref.name).convert("RGBA")

        # 叠加坐标轴到参考图
        # 确保尺寸一致
        if axis_ref.size != pil_ref.size:
            pil_ref = pil_ref.resize(axis_ref.size, Image.BICUBIC)
        pil_ref_rgba = pil_ref.convert("RGBA")
        overlaid_ref = Image.alpha_composite(pil_ref_rgba, axis_ref).convert("RGB")

        # ===== 处理目标图（如果有）=====
        if pil_tgt is not None:
            rel_az = safe_float(ans_dict.get('rel_az_pred', 0))
            rel_el = safe_float(ans_dict.get('rel_el_pred', 0))
            rel_ro = safe_float(ans_dict.get('rel_ro_pred', 0))

            tgt_azi, tgt_ele, tgt_rot = Get_target_azi_ele_rot(az, el, ro, rel_az, rel_el, rel_ro)
            print("Target: Azi",tgt_azi,"Ele",tgt_ele,"Rot",tgt_rot)
            
            # target 默认 alpha=1（根据你的说明）
            axis_renderer.render_axis(tgt_azi, tgt_ele, tgt_rot, alpha=1, save_path=tmp_tgt.name)
            axis_tgt = Image.open(tmp_tgt.name).convert("RGBA")

            if axis_tgt.size != pil_tgt.size:
                pil_tgt = pil_tgt.resize(axis_tgt.size, Image.BICUBIC)
            pil_tgt_rgba = pil_tgt.convert("RGBA")
            overlaid_tgt = Image.alpha_composite(pil_tgt_rgba, axis_tgt).convert("RGB")
        else:
            overlaid_tgt = None
            rel_az = rel_el = rel_ro = 0.0
    finally:
        # 安全删除临时文件（即使出错也清理）
        if os.path.exists(tmp_ref.name):
            os.remove(tmp_ref.name)
            print('cleaned {}'.format(tmp_ref.name))
        if os.path.exists(tmp_tgt.name):
            os.remove(tmp_tgt.name)
            print('cleaned {}'.format(tmp_tgt.name))

    return [
        overlaid_ref,  # 渲染+叠加后的参考图
        overlaid_tgt,  # 渲染+叠加后的目标图（可能为 None）
        f"{az:.2f}",
        f"{el:.2f}",
        f"{ro:.2f}",
        str(alpha),
        f"{rel_az:.2f}",
        f"{rel_el:.2f}",
        f"{rel_ro:.2f}",
    ]


# ====== Gradio Blocks UI ======
with gr.Blocks(title="Orient-Anything Demo") as demo:
    gr.Markdown("# Orient-Anything Demo")
    gr.Markdown("Upload a **reference image** (required). Optionally upload a **target image** for relative pose.")

    with gr.Row():
        # 左侧：输入图像（参考图 + 目标图，同一行）
        with gr.Column():
            with gr.Row():
                ref_img = gr.Image(
                    label="Reference Image (required)",
                    type="numpy",
                    height=256,
                    width=256,
                    value=None,
                    interactive=True
                )
                tgt_img = gr.Image(
                    label="Target Image (optional)",
                    type="numpy",
                    height=256,
                    width=256,
                    value=None,
                    interactive=True
                )
            rm_bkg = gr.Checkbox(label="Remove Background", value=True)
            run_btn = gr.Button("Run Inference", variant="primary")
            # === 在这里插入示例 ===
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["assets/examples/F35-0.jpg", "assets/examples/F35-1.jpg"],
                        ["assets/examples/skateboard-0.jpg", "assets/examples/skateboard-1.jpg"],
                    ],
                    inputs=[ref_img, tgt_img],
                    examples_per_page=2,
                    label="Example Inputs (click to load)"
                )
                gr.Examples(
                    examples=[
                        ["assets/examples/table-0.jpg", "assets/examples/table-1.jpg"],
                        ["assets/examples/bottle.jpg", None],
                    ],
                    inputs=[ref_img, tgt_img],
                    examples_per_page=2,
                    label=""
                )

        # 右侧：结果图像 + 文本输出
        with gr.Column():
            # 结果图像：参考结果 + 目标结果（可选）
            with gr.Row():
                res_ref_img = gr.Image(
                    label="Rendered Reference",
                    type="pil",
                    height=256,
                    width=256,
                    interactive=False
                )
                res_tgt_img = gr.Image(
                    label="Rendered Target (if provided)",
                    type="pil",
                    height=256,
                    width=256,
                    interactive=False
                )
                
            # 文本输出放在图像下方
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Absolute Pose (Reference)")
                    az_out = gr.Textbox(label="Azimuth (0~360°)")
                    el_out = gr.Textbox(label="Polar (-90~90°)")
                    ro_out = gr.Textbox(label="Rotation (-90~90°)")
                    alpha_out = gr.Textbox(label="Number of Directions (0/1/2/4)")
                with gr.Column():
                    gr.Markdown("### Relative Pose (Target w.r.t Reference)")
                    rel_az_out = gr.Textbox(label="Relative Azimuth (0~360°)")
                    rel_el_out = gr.Textbox(label="Relative Polar (-90~90°)")
                    rel_ro_out = gr.Textbox(label="Relative Rotation (-90~90°)")

    # 绑定点击事件
    run_btn.click(
        fn=run_inference,
        inputs=[ref_img, tgt_img, rm_bkg],
        outputs=[res_ref_img, res_tgt_img, az_out, el_out, ro_out, alpha_out, rel_az_out, rel_el_out, rel_ro_out],
        preprocess=True,
        postprocess=True
    )
    
    

# 启动（禁用 API 避免 schema 错误）
demo.launch(show_api=False)
