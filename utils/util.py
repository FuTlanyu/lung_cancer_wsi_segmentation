
LEVEL_USED = 0

# white background
PADDING = 512
# 在最终要分割层级中
PATCH_SIZE = 512
TRAIN_STRIDE = 384
OVERLAY = 128

TEST_STRIDE = 512

program_path = 'E:/python/program/Dachuang/Unet-like-model/UnetSeg-Pytorch/'
program_data = 'E:/UnetSegPy-Data/data/'

# ==================================  TRAIN PATH  ========================================
train_wsi_path = 'F:/LungCancerData/LocalTestData/train/train_wsi'
train_mask_path = 'F:/LungCancerData/LocalTestData/train/train_mask'

THESIS_FIGURE_DIR = program_data + 'tissue_mask/' + 'level' + str(LEVEL_USED) + '/t20_wsi/'

train_path = program_data + 'patch/train/'
train_img = train_path + 'level' + str(LEVEL_USED) + '/imgs/'
train_mask = train_path + 'level' + str(LEVEL_USED) + '/masks/'

# train_path = 'H:/trainPatch50/'
# train_img = train_path + 'level' + str(LEVEL_USED) + '/imgs/'
# train_mask = train_path + 'level' + str(LEVEL_USED) + '/masks/'


dir_checkpoint = program_path + 'saved_model/checkpoints/'

# ==================================  TEST PATH  ========================================
test_wsi_path = 'H:/LungCancerData/training1_test/imgs'
test_mask_path = 'H:/LungCancerData/training1_test/masks'

test_path = program_data + 'patch/test/max/'

# test_path = 'H:/t1_20_patch/'

test_patch_dir = test_path + 'patch_image/'
gt_mask_dir = test_path + 'gt_mask/'
predicted_mask_dir = test_path + 'predicted_mask/'


model_dir = program_path + 'saved_model/checkpoints/'
log_dir = program_path + 'log/'

test_result_vis_dir = test_path + 'visualization/'


def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    filename = filename.split('\\')[1]
    return filename


def get_type(file_path):
    path_tokens = file_path.split('/')
    file_type = path_tokens[-2]
    return file_type.upper()
