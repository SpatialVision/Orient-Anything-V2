DINO_SMALL  = "facebook/dinov2-small"
DINO_BASE   = "facebook/dinov2-base"
DINO_LARGE  = "facebook/dinov2-large"
DINO_GIANT  = "facebook/dinov2-giant"
VGGT_1B     = "facebook/VGGT-1B"

HF_CKPT_PATH = "demo_ckpts/rotmod_realrotaug_best.pt"
LOCAL_CKPT_PATH = "/root/autodl-tmp/zza_backup/OriAnyV2_ckpt/rotmod_realrotaug/rotmod_realrotaug_best.pt"

RENDER_FILE = "assets/axis_render.blend"

DATA_ROOT = ""

COCO_ROOT       = f'{DATA_ROOT}/coco_val/class_seg/white/'
COCO_META       = f'{DATA_ROOT}/coco_val/coco_anno_direction_ref.csv'

LINEMOD_META    = f'{DATA_ROOT}/POPE/data/metadatas/LM_dataset.pt'
ONEPOSE_META    = f'{DATA_ROOT}/POPE/data/metadatas/onepose.pt'
ONEPOSEPP_META  = f'{DATA_ROOT}/POPE/data/metadatas/onepose_plusplus.pt'
YCBV_META       = f'{DATA_ROOT}/POPE/data/metadatas/ycbv.pt'
LINEMOD_RANDOM_META    = f'{DATA_ROOT}/POPE/data/metadatas/LM_dataset_random.pt'
ONEPOSE_RANDOM_META    = f'{DATA_ROOT}/POPE/data/metadatas/onepose_random.pt'
ONEPOSEPP_RANDOM_META  = f'{DATA_ROOT}/POPE/data/metadatas/onepose_plusplus_random.pt'
YCBV_RANDOM_META       = f'{DATA_ROOT}/POPE/data/metadatas/ycbv_random.pt'

LINEMOD_ROOT    = f'{DATA_ROOT}/POPE/data/LM_dataset/'
ONEPOSE_ROOT    = f'{DATA_ROOT}/POPE/data/onepose/'
ONEPOSEPP_ROOT  = f'{DATA_ROOT}/POPE/data/onepose_plusplus/'
YCBV_ROOT       = f'{DATA_ROOT}/POPE/data/ycbv/'
POPE_ROOT       = f'{DATA_ROOT}/POPE/data/'

OMNI6DPOSE_META_BAL = f'{DATA_ROOT}/Omni6DPose/Meta/obj_meta_angle_balanced.csv'
OMNI6DPOSE_ROOT     = f'{DATA_ROOT}/Omni6DPose/render_output/'

OBJECTRON_META  = f'{DATA_ROOT}/objectron/objectron_meta.csv'
OBJECTRON_ROOT  = f'{DATA_ROOT}/objectron/objectron_data/test_crop/'

SUNRGBD_META  = f'{DATA_ROOT}/SUNRGBD_test/sunrgbd_meta.csv'
SUNRGBD_ROOT  = f'{DATA_ROOT}/SUNRGBD_test/test_crop/'

ARK_META  = f'{DATA_ROOT}/ARKitScenes/arkitscenes_meta.csv'
ARK_ROOT  = f'{DATA_ROOT}/ARKitScenes/test_crop/'
