# from model import *
# from data import *
import util as util
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import numpy as np
import cv2
import glob
import os
import time
import math
from scipy import misc
import imageio
# from keras.models import load_model

# level5 所对应的一系列常量



'''
wsi_image = OpenSlide(util.TUMOR_WSI_PATH + '/86.tif')
mask_image = OpenSlide(util.TUMOR_MASK_PATH + '/86.tif')

dim = wsi_image.level_dimensions[LEVEL]
print(dim) # (5578, 13049)

# 分割大小为(1024, 2048)的patch测试

# 第一个数减小往左移动，第二个数增大往下移动

patch = wsi_image.read_region((5000+1024*30, 7000+2048*22), LEVEL, (512, 512)).convert('RGB')
mask = mask_image.read_region((5000+1024*30, 7000+2048*22), LEVEL, (512, 512)).convert('RGB')

patch = np.array(patch)
mask = np.array(mask)

r, g, b = cv2.split(mask)
mask = cv2.merge([r])
# saved size (2048, 1024)
# 在(2048, 1024)的这块组织区域中非零点的个数仅占59318 2.82%

white_pixel_cnt_mask = cv2.countNonZero(mask)
print(white_pixel_cnt_mask)
misc.imsave(util.TEST_PATCH_DIR + '0' + ".png", patch)
misc.imsave(util.TEST_MASK_DIR + '0' + ".png", mask)

print()
'''



'''
算法流程：
读入WSI测试数据的image和mask；
从LEVEL中取数据测试，将mask保存在某一文件夹中；

'''



class TEST(object):
    LEVEL = 0

    x_dim = 89262
    y_dim = 208796
    row = 0
    col = 0
    img_num = 0

    # x_padding = 0
    # y_padding = 0
    # stride = 0
    size = 512

    f = open(util.log_dir + time.strftime("%Y%m%d_", time.localtime())  + 'testResultLog.txt', 'a')

    def test_patch_generate(self, test_wsi_path, test_mask_path):
        wsi_paths = glob.glob(os.path.join(test_wsi_path, '*.tif'))
        wsi_paths.sort()
        mask_paths = glob.glob(os.path.join(test_mask_path, '*.tif'))
        mask_paths.sort()

        wsi_num = len(wsi_paths)
        mask_num = len(mask_paths)
        assert wsi_num == mask_num, "the number of test wsi image and mask image is not equal"

        for i in range(wsi_num):
            wsi_path = wsi_paths[i]
            mask_path = mask_paths[i]
            wsi_filename = util.get_filename_from_path(wsi_path)
            mask_filename = util.get_filename_from_path(mask_path)
            assert wsi_filename == mask_filename, "the index of wsi image and mask image is not equal"
            print('Test image %s is being extracted.' % wsi_filename)
            self.f.write('Test image %s is being extracted.' % wsi_filename)

            wsi = OpenSlide(wsi_path)
            mask = OpenSlide(mask_path)
            # level0 (89262, 208796)
            # 在wsi与对应的mask中分割出对应的patch，存储在文件夹下
            self.x_dim = wsi.level_dimensions[self.LEVEL][0]
            self.y_dim = wsi.level_dimensions[self.LEVEL][1]

            # ========================== 不扩充进行分割，舍弃边角 ===========================

            self.row = self.x_dim // self.size
            self.col = self.y_dim // self.size
            self.img_num = self.row * self.col
            print('Extract %d patches from image %s' % (self.img_num, wsi_filename))
            self.f.write('Extract %d patches from image %s' % (self.img_num, wsi_filename))

            for index in range(int(self.img_num)):
                x = int(int(index // self.col) * self.size)
                y = int(index % self.col * self.size)
                img = wsi.read_region((x, y), self.LEVEL, (util.PATCH_SIZE, util.PATCH_SIZE)).convert('RGB')
                gt_mask = mask.read_region((x, y), self.LEVEL, (util.PATCH_SIZE, util.PATCH_SIZE)).convert('RGB')

                img = np.array(img)
                gt_mask = np.array(gt_mask)

                # =============== 将ground truth mask patch存储在文件夹下 =========================
                r, g, b = cv2.split(gt_mask)
                gt_mask_img = cv2.merge([r])

                # 保存的是0-255的真实图像
                gt_mask_img = gt_mask_img * 255

                gt_mask_save_dir = util.gt_mask_dir + wsi_filename
                if not os.path.exists(gt_mask_save_dir):
                    os.mkdir(gt_mask_save_dir)
                imageio.imsave(gt_mask_save_dir + '/' + str(index) + '.png', gt_mask_img)

                # ================== 将patch image存储在文件夹下 ================================
                patch_img_save_dir = util.test_patch_dir + wsi_filename
                if not os.path.exists(patch_img_save_dir):
                    os.mkdir(patch_img_save_dir)
                imageio.imsave(patch_img_save_dir + '/' + str(index) + '.png', img)

                print('index: ', index)




            # wsi_image = wsi.read_region((0, 0), self.LEVEL, (util.PATCH_SIZE, util.PATCH_SIZE)).convert('RGB')
            # mask_image = mask.read_region((0, 0), self.LEVEL, (util.PATCH_SIZE, util.PATCH_SIZE)).convert('RGB')
            #
            # wsi_image = np.array(wsi_image)
            # mask_image = np.array(mask_image)
            #
            # '''
            # Save whole ground truth mask image from level 0 to a directory
            # '''
            # r, g, b = cv2.split(mask_image)
            #
            # mask_image = cv2.merge([r])
            # misc.imsave(util.VALID_MASK_DIR + mask_filename + '.png', mask_image)
            #
            # # 从wsi_image中分割patch，制作数据集
            # x_dim = wsi_image.shape[0]
            # y_dim = wsi_image.shape[1]
            # print(wsi_image.shape)
            # assert x_dim == self.x_dim, "the x dimension(%d) of image %s is not equal to the default value(%d)" \
            #                                                                % (x_dim, wsi_filename, self.x_dim)
            # assert y_dim == self.y_dim, "the y dimension(%d) of image %s is not equal to the default value(%d)" \
            #                                                                % (y_dim, wsi_filename, self.y_dim)
            #
            # # 对wsi_image作padding操作
            # wsi_image = cv2.copyMakeBorder(wsi_image, 0, self.x_padding, 0, self.y_padding, cv2.BORDER_REPLICATE)
            # print(wsi_image.shape)
            # # 根据index确定顶点的位置，在wsi_image切出对应的patch，存储在wsi_filename对应的文件夹下
            # for index in range(int(self.img_num)):
            #     x = int(int(index // self.col) * 256)
            #     y = int(index % self.col * 256)
            #     img = wsi_image[x:x+self.size, y:y+self.size, :]
            #
            #     save_dir = util.VALID_PATCH_DIR + wsi_filename
            #     if not os.path.exists(save_dir):
            #         os.mkdir(save_dir)
            #
            #     misc.imsave(save_dir + '/' + str(index) + '.png', img)
            #     print('index: ', index)

    def test_inject(self):
        model = load_model(util.TRAINED_MODEL)
        dirs= os.listdir(util.VALID_PATCH_DIR)
        print(dirs)
        self.f.write(dirs)
        self.f.flush()
        for dir in dirs:
            print('inject test patches from image %s to the trained model' % dir)
            self.f.write('inject test patches from image %s to the trained model\n' % dir)
            self.f.flush()

            files = glob.glob(os.path.join(util.VALID_PATCH_DIR + dir, '*.png'))
            # assert len(files) == self.img_num, 'image number(%d) to test is not equal to the default image number(%d)' \
            #                                % (len(files), self.img_num)

            testGene = testGenerator(util.VALID_PATCH_DIR + dir, num_image=len(files), target_size=(self.size, self.size))

            results = model.predict_generator(testGene, steps=len(files), verbose=1)  # what is steps?

            predicted_patch_dir = util.VALID_PREDICTED_PATCH_DIR + dir
            if not os.path.exists(predicted_patch_dir):
                os.mkdir(predicted_patch_dir)

            saveResult(predicted_patch_dir, results)

    def combine_patch_mask(self):
        print('Combine mask patches into a whole mask image')
        dirs = os.listdir(util.TEST_PREDICTED_PATCH_DIR)

        for img_name in dirs:
            print('Combine mask image %s' % img_name)
            mask_path_dir = util.TEST_PREDICTED_PATCH_DIR + img_name

            mask_patches = glob.glob(os.path.join(mask_path_dir, '*.npy'))
            mask_patches.sort()
            assert len(mask_patches) == self.img_num, 'mask patches(%d) to combine is not equal to the default image number(%d)' \
                                               % (len(mask_patches), self.img_num)

            # 创建一个扩充后的初始值为-1的矩阵 shape = (6656, 2816) dtype = float64
            mask_img = np.ones((self.x_dim + self.x_padding, self.y_dim + self.y_padding)) * -1
            # 任取出一个patch放在对应的位置，对其中的每一个点做判断，若其下的值为-1，直接复制，否则取平均值
            for mask_patch in mask_patches:
                index = int(util.get_filename_from_path(mask_patch))
                x = int(int(index // self.col) * 256)
                y = int(index % self.col * 256)
                # patch_img = misc.imread(mask_patch)
                patch_img = np.load(mask_patch)
                for i in range(self.size):
                    for j in range(self.size):
                        patch_val = patch_img[i][j]
                        mask_val = mask_img[x+i][y+j]
                        if mask_val == -1:
                            mask_img[x + i][y + j] = patch_val
                        else:
                            mask_img[x + i][y + j] = (mask_val + patch_val)/2

            mask_img_clip = mask_img[:self.x_dim, :self.y_dim]
            assert mask_img_clip.shape[0] == self.x_dim and mask_img_clip.shape[1] == self.y_dim, 'generated mask image size is not correspond to the default mask size'

            # misc.imsave(util.TEST_PREDICTED_MASK_DIR + img_name + '.png', mask_img_clip)
            np.save(util.TEST_PREDICTED_MASK_DIR + img_name + '.npy', mask_img_clip)



    def dice_result(self):
        # =============================  对整张图像计算dice系数  ==================================
        # print('Calculate dice coefficient for each image...')
        # pred_masks = glob.glob(os.path.join(util.TEST_PREDICTED_MASK_DIR, '*.npy'))
        # pred_masks.sort()
        #
        # gt_masks = glob.glob(os.path.join(util.TEST_MASK_DIR, '*.png'))
        # gt_masks.sort()
        #
        # assert len(pred_masks) == len(gt_masks), 'predict image number(%d) is not equal to the ground truth image number(%d)'\
        #                                             %(len(pred_masks), len(gt_masks))
        #
        # dice_average = 0
        # for i in range(len(pred_masks)):
        #     pred_mask_path = pred_masks[i]
        #     gt_mask_path = gt_masks[i]
        #
        #     pred_filename = util.get_filename_from_path(pred_mask_path)
        #     gt_filename = util.get_filename_from_path(gt_mask_path)
        #     assert pred_filename == gt_filename, 'predict mask(%s) is not correspond to the ground truth mask(%s)'\
        #                                             %(pred_filename, gt_filename)
        #
        #     pred_mask = np.load(pred_mask_path)
        #     gt_mask = imageio.imread(gt_mask_path)
        #
        #     # intersection = np.sum([pred_mask == gt_mask])
        #     # dice = intersection/(self.x_dim*self.y_dim)
        #     dice = util.dice_coeff(pred_mask, gt_mask)
        #     print('For image %s, the dice coefficient is %.5f.' % (pred_filename, dice))
        #
        #     dice_average += dice
        #
        # dice_average = dice_average / len(pred_masks)
        # print('For %d images, the average dice coefficient is %.5f.' % (len(pred_masks), dice_average))


        # ========================  对patch求dice系数，并将其累加  ===============================
        print('Calculate dice coefficient for each image...')
        self.f.write('Calculate dice coefficient for each image...\n')
        self.f.flush()
        gt_dirs = os.listdir(util.gt_mask_dir)
        pred_dirs = os.listdir(util.predicted_mask_dir)
        assert gt_dirs == pred_dirs, 'ground truth masks are not equal to predicted masks'
        dice_all = 0
        count = 0
        for img_name in gt_dirs:
            intersection = 0
            pred_num = 0
            gt_num = 0

            print('Calculate dice coefficient for %s' % img_name)
            self.f.write('Calculate dice coefficient for %s\n' % img_name)
            self.f.flush()
            gt_patch_dir = util.gt_mask_dir + img_name
            pred_patch_dir = util.predicted_mask_dir + img_name

            gt_masks = glob.glob(os.path.join(gt_patch_dir, '*.png'))
            gt_masks.sort()
            pred_masks = glob.glob(os.path.join(pred_patch_dir, '*.png'))
            pred_masks.sort()
            assert len(gt_masks) == len(pred_masks), 'The number of ground truth patches is not equal to predicted patches'
            gt_pred_pairs = zip(gt_masks, pred_masks)

            for gt_pred_pair in gt_pred_pairs:
                gt_patch = gt_pred_pair[0]
                pred_patch = gt_pred_pair[1]
                gt_name = util.get_filename_from_path(gt_patch)
                pred_name = util.get_filename_from_path(pred_patch)
                assert gt_name == pred_name, 'The ground truth patch is not corresponding to the predicted patch'

                # print(gt_name)

                gt_patch = imageio.imread(gt_patch) / 255
                pred_patch = imageio.imread(pred_patch) /255

                intersection += (gt_patch * pred_patch).sum()
                gt_num += gt_patch.sum()
                pred_num += pred_patch.sum()

            print('intersection: ', intersection)
            print('gt_num: ', gt_num)
            print('pred_num: ', pred_num)
            dice = 2. * intersection / (gt_num + pred_num)
            print('For image %s, the dice coefficient is %f.' % (img_name, dice))
            self.f.write('For image %s, the dice coefficient is %f.\n' % (img_name, dice))
            self.f.flush()
            dice_all += dice
            count += 1

        print('\nFor %d image(s), the average dice coefficient is %f.' % (count, dice_all/count))
        self.f.write('\nFor %d image(s), the average dice coefficient is %f.\n' % (count, dice_all/count))
        self.f.flush()
def run():

    testProc = TEST()

    # ========================== 扩充后进行分割 ==========================
    # testProc.x_padding = math.ceil((testProc.x_dim - testProc.size) / testProc.stride) * testProc.stride - (testProc.x_dim - testProc.size)
    # testProc.y_padding = math.ceil((testProc.y_dim - testProc.size) / testProc.stride) * testProc.stride - (testProc.y_dim - testProc.size)
    #
    # testProc.row = (testProc.x_dim + testProc.x_padding - testProc.size) / testProc.stride + 1
    # testProc.col = (testProc.y_dim + testProc.y_padding - testProc.size) / testProc.stride + 1


    # testProc.test_patch_generate(util.test_wsi_path, util.test_mask_path)
    # os.system('python predict.py')
    # testProc.test_inject()
    #
    # testProc.combine_patch_mask()
    testProc.dice_result()

    testProc.f.close()
    '''Simple test'''
    # model = load_model(util.TRAINED_MODEL)
    # files = glob.glob(os.path.join('E:/python/program/Dachuang/Unet-like-model/UnetSeg/data/patch/train/level0/corresponding/patch_image', '*.png'))
    # testGene = testGenerator('E:/python/program/Dachuang/Unet-like-model/UnetSeg/data/patch/train/level0/corresponding/patch_image', num_image=len(files), target_size=(512, 512))
    #
    # results = model.predict_generator(testGene, steps=len(files), verbose=1)  # what is steps?
    #
    # predicted_patch_dir = 'E:/python/program/Dachuang/Unet-like-model/UnetSeg/data/patch/train/level0/corresponding/predicted'
    # if not os.path.exists(predicted_patch_dir):
    #     os.mkdir(predicted_patch_dir)
    #
    # # 将测试结果作为numpy数组保存
    # saveResult(predicted_patch_dir, results)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    run()


# ===================================  单独计算真实mask图像中点的个数  =====================================
# 若算openslide则无法在AI Studio下计算
# LEVEL0  (89262, 208796)
# x = 89262, 将其18等分, 89262 = 2 * 3 * 3 * 3 * 3 * 19 * 29
# y = 208796, 将其14等分, 2087962 * 2 * 7 * 7457
# gt_num = 0
# mask_path = util.test_mask_path + '/' + img_name + '.tif'
# mask = OpenSlide(mask_path)
#
# xs_num = 18
# x = mask.level_dimensions[self.LEVEL][0]
# x_area = x // xs_num
#
# ys_num = 14
# y = mask.level_dimensions[self.LEVEL][1]
# y_area = y // ys_num
#
# for i in range(xs_num):
#     for j in range(ys_num):
#         area = mask.read_region((i * x_area, j * y_area), self.LEVEL, (x_area, y_area)).convert('RGB')
#         area_image = np.array(area)
#         r, g, b = cv2.split(area_image)
#
#         area_image = cv2.merge([r])
#         gt_num += area_image.sum()