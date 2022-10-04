import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import util as util
import random
import imageio
import time
import math
from utils.data_vis import plot_img_and_mask_2, plot_img_and_mask_3

class WSI(object):
    """
        # ================================
        # Class to annotate WSIs with ROIs
        # ================================

    """
    # wsi number get extracted
    index = 0
    patch_num = 0
    wsi_paths = []
    mask_paths = []
    def_level = 7 # not used
    key = 0
    wsi_type = 'TRAIN'
    preFlag = 0
    vis = False
    bpatches = []

    f = open(util.log_dir + time.strftime("%Y%m%d_", time.localtime()) + 'level' + str(util.LEVEL_USED)
             + '_' + 'patchLog.txt', 'a')

    def read_normal_wsi(self, wsi_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            self.wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)

            level = min(self.def_level, self.wsi_image.level_count - 1)
            print('level used: %d' % level)
            print(self.wsi_image.level_dimensions[level])

            self.rgb_image_pil = self.wsi_image.read_region((0, 0), level,
                                                            self.wsi_image.level_dimensions[level])
            self.rgb_image = np.array(self.rgb_image_pil)

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True



    def read_tumor_wsi(self, wsi_path, mask_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            self.wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)
            self.mask_image = OpenSlide(mask_path)

            level_used = 4
            self.rgb_image_pil = self.wsi_image.read_region((0, 0), level_used,
                                                            self.wsi_image.level_dimensions[level_used])
            # shape = (13049, 5578, 4)
            self.rgb_image = np.array(self.rgb_image_pil)

            # mask_level = self.mask_image.level_count - 1
            mask_level = level_used
            self.rgb_mask_pil = self.mask_image.read_region((0, 0), mask_level,
                                                            self.mask_image.level_dimensions[mask_level])

            resize_factor = float(1.0 / pow(2, level_used - mask_level))
            # 为什么需要resize而不直接取出某一层级的mask和wsi
            # resize后mask与rgb_image大小并不相同
            self.mask = cv2.resize(np.array(self.rgb_mask_pil), (0, 0), fx=resize_factor, fy=resize_factor)

            r, g, b, a = cv2.split(self.mask)
            r = r*255
            g = g*255
            b = b*255
            self.mask = cv2.merge([r, g, b, a])

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True

    def find_roi_normal(self):
        # self.mask = cv2.cvtColor(self.mask, cv2.CV_32SC1)
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        # [20, 20, 20]
        lower_red = np.array([30, 30, 30])
        # [255, 255, 255]
        upper_red = np.array([200, 200, 200])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(self.rgb_image, self.rgb_image, mask=mask)

        # (50, 50)
        close_kernel = np.ones((50, 50), dtype=np.uint8)
        close_kernel_tmp = np.ones((30, 30), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        image_close_tmp = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel_tmp))
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        open_kernel_tmp = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        image_open_tmp = Image.fromarray(cv2.morphologyEx(np.array(image_close_tmp), cv2.MORPH_OPEN, open_kernel_tmp))
        contour_rgb, bounding_boxes, contour_rgb_tmp = self.get_normal_image_contours(np.array(image_open),
                                                                                      self.rgb_image,
                                                                                      np.array(image_open_tmp))
        self.draw_bbox(bounding_boxes)

        self.display(contour_rgb, contour_rgb_tmp)

    def find_roi_tumor(self, flag, tumor_patches):
        # self.mask = cv2.cvtColor(self.mask, cv2.CV_32SC1)
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 40, 50])
        upper_red = np.array([200, 200, 200])
        # mask 为组织区域的掩码图像   tissue_mask
        tissue_mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(self.rgb_image, self.rgb_image, mask=tissue_mask)

        # (50, 50)
        close_kernel = np.ones((20, 20), dtype=np.uint8)
        close_kernel_tmp = np.ones((50, 50), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(tissue_mask), cv2.MORPH_CLOSE, close_kernel))
        image_close_tmp = Image.fromarray(cv2.morphologyEx(np.array(tissue_mask), cv2.MORPH_CLOSE, close_kernel_tmp))
        # (30, 30)
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        open_kernel_tmp = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        image_open_tmp = Image.fromarray(cv2.morphologyEx(np.array(image_close_tmp), cv2.MORPH_OPEN, open_kernel_tmp))
        contour_rgb, contour_mask, bounding_boxes, contour_rgb_tmp, contour_mask_tmp = self.get_tumor_image_contours(
            np.array(image_open), self.rgb_image,
            self.mask, np.array(image_open_tmp))

        # image_to_process = self.wsi_image
        # mask_to_process = self.gt_mask
        wsi_name = util.get_filename_from_path(self.wsi_path)
        cv2.imwrite(os.path.join(util.THESIS_FIGURE_DIR, wsi_name) + '_hsv_mask.png', tissue_mask)
        # cv2.imwrite(os.path.join(util.THESIS_FIGURE_DIR, wsi_name) + '_contour_rgb.png', contour_rgb)
        # cv2.imwrite(os.path.join(util.THESIS_FIGURE_DIR, wsi_name) + '_contour_mask.png', contour_mask)
        # cv2.imwrite(os.path.join(util.THESIS_FIGURE_DIR, wsi_name) + '_contour_rgb_tmp.png', contour_rgb_tmp)
        # cv2.imwrite(os.path.join(util.THESIS_FIGURE_DIR, wsi_name) + '_contour_mask_tmp.png', contour_mask_tmp)
        # cv2.imwrite(os.path.join(util.THESIS_FIGURE_DIR, wsi_name) + '_bbox_rgb_image.png', np.array(self.rgb_image_pil))
        cv2.imwrite(os.path.join(util.THESIS_FIGURE_DIR, wsi_name) + '_tumor_mask_image.png', self.mask)


        # x_list, y_list = self.get_list_by_step(bounding_boxes, 16)
        # print(x_list, y_list)
        vflag, vTumor = self.get_patch_and_mask(bounding_boxes, tissue_mask, flag, tumor_patches)
        return vflag, vTumor



    def get_normal_image_contours(self, cont_img, rgb_image, cont_img_tmp):
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours_tmp, _ = cv2.findContours(cont_img_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        # print(boundingBoxes)
        contours_rgb_image_array = np.array(rgb_image)
        contours_rgb_image_array_tmp = np.array(rgb_image)

        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 3)
        cv2.drawContours(contours_rgb_image_array_tmp, contours_tmp, -1, line_color, 3)
        # cv2.drawContours(gt_mask, contours_mask, -1, line_color, 3)
        return contours_rgb_image_array, boundingBoxes, contours_rgb_image_array_tmp



    def process_level(self, coor_list, level, tissue_mask, flag, tumor_patches):
        for coord in coor_list:
            # level4
            coor_x = coord[0]
            coor_y = coord[1]
            # coor_x_res = coor_x*(2**(4-level))  # magic number ???
            # coor_y_res = coor_y*(2**(4-level))
            patch = self.wsi_image.read_region((coor_x * 2**4, coor_y * 2**4), level, (util.PATCH_SIZE, util.PATCH_SIZE)).convert('RGB')
            # mask_image为肿瘤mask，之前已经存储过
            mask_pt = self.mask_image.read_region((coor_x * 2**4, coor_y * 2**4), level, (util.PATCH_SIZE, util.PATCH_SIZE)).convert('RGB')

            level4_psize = int(util.PATCH_SIZE / 2**(4-util.LEVEL_USED))

            mask_ts = tissue_mask[coor_y:coor_y + level4_psize, coor_x:coor_x + level4_psize]
            patch_array = np.array(patch)
            mask_array = np.array(mask_pt)
            # print(patch_array.shape)
            # print(mask_array.shape)

            if self.wsi_type == 'TRAIN':
                # ====================   Train Patch Extract  ======================
                white_pixel_cnt_ts = cv2.countNonZero(mask_ts)

                tissue_area = 0.1
                if white_pixel_cnt_ts > (level4_psize * level4_psize * tissue_area):

                    self.bpatches.append((coor_x, coor_y, level4_psize, level4_psize))

                    # r, g, b, a = cv2.split(patch_array)
                    # merged = cv2.merge([r, g, b])
                    r, g, b = cv2.split(mask_array)
                    mask_array = cv2.merge([r])

                    white_pixel_cnt_mask = cv2.countNonZero(mask_array)
                    # 记录下所有的patch中正样本的数量
                    if white_pixel_cnt_mask > util.PATCH_SIZE * util.PATCH_SIZE * 0.5:
                        print('tumor patch: ', tumor_patches)
                        tumor_patches += 1


                    imageio.imwrite(util.train_img + str(flag) + ".png", patch_array)
                    imageio.imwrite(util.train_mask + str(flag) + ".png", mask_array)

                    if self.vis:
                        plot_img_and_mask_2(patch_array, mask_array)
                        # plot_img_and_mask_3(str(self.index), str(flag), patch_array, mask_array, mask_ts)

                    print('flag: ', flag)
                    flag += 1

            else:
                # ====================   Test Patch Extract  ======================
                # 组织区域占比 > 0.1 才作为test patch
                white_pixel_cnt_ts = cv2.countNonZero(mask_ts)
                if white_pixel_cnt_ts > (level4_psize * level4_psize * 0.05):

                    self.bpatches.append((coor_x, coor_y, level4_psize, level4_psize))
                    num = flag - self.preFlag
                    # =============== 将ground truth mask patch存储在文件夹下 =========================
                    r, g, b = cv2.split(mask_array)
                    gt_mask_img = cv2.merge([r])

                    gt_mask_save_dir = util.gt_mask_dir + str(self.index)
                    if not os.path.exists(gt_mask_save_dir):
                        os.mkdir(gt_mask_save_dir)

                    imageio.imsave(gt_mask_save_dir + '/' + str(num) + '.png', gt_mask_img * 255)

                    # ================== 将patch image存储在文件夹下 ================================
                    patch_img_save_dir = util.test_patch_dir + str(self.index)
                    if not os.path.exists(patch_img_save_dir):
                        os.mkdir(patch_img_save_dir)

                    imageio.imsave(patch_img_save_dir + '/' + str(num) + '.png', patch_array)
                    print('index: ', num)
                    flag += 1
                    if self.vis:
                        plot_img_and_mask_2(patch_array, gt_mask_img)

        return flag, tumor_patches

    def get_list(self, bounding_boxes):
        x_list = []
        y_list = []

        for i, bounding_box in enumerate(bounding_boxes):
            x_start = int(bounding_box[0])
            y_start = int(bounding_box[1])
            x_end = x_start + bounding_box[2]
            y_end = y_start + bounding_box[3]

            num_x = int(bounding_box[2] / 10) + 1
            num_y = int(bounding_box[3] / 10) + 1

            for j in range(min(num_x, num_y)):
                x_list.append(random.randint(x_start, x_end))
                y_list.append(random.randint(y_start, y_end))

        return x_list, y_list

    def get_list_by_step(self, bounding_boxes, step):
        x_list = []
        y_list = []
        for i, bounding_box in enumerate(bounding_boxes):
            x = bounding_box[0]
            y = bounding_box[1]
            step_num_x = int(bounding_box[2]/step)
            step_num_y = int(bounding_box[3]/step)

            if step_num_x > 0 and step_num_y > 0:
                for m in range(step_num_x):
                    for n in range(step_num_y):
                        x_list.append(x + m * step)
                        y_list.append(y + n * step)

        return x_list, y_list

    def get_patch_and_mask(self, bounding_boxes, tissue_mask, flag, tumor_patches):
        new_bboxes = []
        if self.wsi_type == 'VALID':
            x_list = []
            y_list = []
            x_end_list = []
            y_end_list = []
            for bounding_box in bounding_boxes:
                x = bounding_box[0]
                y = bounding_box[1]
                x_end = int(x + bounding_box[2])
                y_end = int(y + bounding_box[3])

                if bounding_box[2] * bounding_box[3] <= 32 * 32:
                    continue
                elif x >= 0 and x <= 600 and y >= 0 and y <= 1500:
                    continue
                elif x >= 4100 and x <= 5578 and y >= 0 and y <= 1500:
                    continue

                x_list.append(x)
                y_list.append(y)
                x_end_list.append(x_end)
                y_end_list.append(y_end)

                x_list.sort()
                y_list.sort()
                x_end_list.sort(reverse=True)
                y_end_list.sort(reverse=True)

            xl = x_list[0]
            yu = y_list[0]
            xr = x_end_list[0]
            yd = y_end_list[0]
            # print(f'Left-Top: ({xl}, {yu})')
            # print(f'Right-Down: ({xr}, {yd})')
            img_max = self.wsi_image.read_region((xl * 2 ** 4, yu * 2 ** 4), 4, (xr - xl, yd - yu)).convert('RGB')
            img_max = np.array(img_max)
            imageio.imwrite('E:/UnetSegPy-Data/data/demo/imgs/' + f'{str(self.index)}max_{str(x)}_{str(y)}.png', img_max)
            new_bboxes.append((xl, yu, xr - xl, yd - yu))
            self.draw_bbox(new_bboxes, (0, 0, 255), 3) # Red max bounding box

        elif self.wsi_type == 'TRAIN':
            new_bboxes = bounding_boxes

        for bounding_box in new_bboxes:
            # 获取level4下对应的坐标
            x = bounding_box[0]
            y = bounding_box[1]
            # No Padding
            end_x = int(x + bounding_box[2])
            end_y = int(y + bounding_box[3])

            # 获取patch, 当level_used > 4时patch会增大，level_used < 4时patch会减小
            patch_size = int(util.PATCH_SIZE / 2**(4-util.LEVEL_USED))  # 64
            # No Overlay
            stride = int(util.TEST_STRIDE / 2**(4 - util.LEVEL_USED))

            coor_list = []

            x_inter = int((end_x - x - patch_size) / stride) + 1
            y_inter = int((end_y - y - patch_size) / stride) + 1

            # 当框出的区域不足1个patch时也要将其割出
            if x_inter < 0:
                x_inter = 0
            if y_inter < 0:
                y_inter = 0
            for i in range(x_inter + 1):
                for j in range(y_inter + 1):
                    coor_list.append((x + i * stride, y + j * stride))

            flag, tumor_patches = self.process_level(coor_list, util.LEVEL_USED, tissue_mask, flag, tumor_patches)



        return flag, tumor_patches
            # process_level(x_list, y_list, 1, self.mask, self.wsi_image, self.gt_mask)
            # process_level(x_list, y_list, 2, self.mask, self.wsi_image, self.gt_mask)
            # process_level(x_list, y_list, 3, self.mask, self.wsi_image, self.gt_mask)

    # self.get_tumor_image_contours(np.array(image_open), self.rgb_image,self.mask, np.array(image_open_tmp))
    def get_tumor_image_contours(self, cont_img, rgb_image, mask_image, cont_img_tmp):
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours_tmp, _ = cv2.findContours(cont_img_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # _, contours_mask, _ = cv2.findContours(gt_mask, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        # print(boundingBoxes)
        contours_rgb_image_array = np.array(rgb_image)
        contours_rgb_image_array_tmp = np.array(rgb_image)

        # contours_mask_image_array = np.array(mask_image)
        # contours_mask_image_array_tmp = np.array(mask_image)
        contours_mask_image_array = mask_image
        contours_mask_image_array_tmp = mask_image


        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 3)
        cv2.drawContours(contours_rgb_image_array_tmp, contours_tmp, -1, line_color, 3)
        cv2.drawContours(contours_mask_image_array, contours, -1, line_color, 3)
        cv2.drawContours(contours_mask_image_array_tmp, contours_tmp, -1, line_color, 3)
        # cv2.drawContours(gt_mask, contours_mask, -1, line_color, 3)
        self.draw_bbox(boundingBoxes, (0, 255, 0), 5) # green bounding boxes

        return contours_rgb_image_array, contours_mask_image_array, boundingBoxes, contours_rgb_image_array_tmp, \
               contours_mask_image_array_tmp

    def display(self, contour_rgb, contour_rgb_tmp, contour_mask=None):
        # cv2.imshow('rgb', self.rgb_image)
        # cv2.imshow('mask', mask)
        # cv2.imshow('image_close',np.array(image_close))
        # cv2.imshow('image_open', np.array(image_open))

        contour_rgb = cv2.resize(contour_rgb, (0, 0), fx=0.60, fy=0.60)
        cv2.imshow('contour_rgb', np.array(contour_rgb))
        contour_rgb_tmp = cv2.resize(contour_rgb_tmp, (0, 0), fx=0.60, fy=0.60)
        cv2.imshow('contour_rgb_tmp', np.array(contour_rgb_tmp))
        if contour_mask is not None:
            contour_mask = cv2.resize(contour_mask, (0, 0), fx=0.60, fy=0.60)
            cv2.imshow('contour_mask', np.array(contour_mask))



    def draw_bbox(self, bounding_boxes, line_color, thickness):
        draw = ImageDraw.Draw(self.rgb_image_pil)
        for i, bounding_box in enumerate(bounding_boxes):
            x = int(bounding_box[0])
            y = int(bounding_box[1])
            for offset in range(thickness):
                draw.rectangle([x + offset, y + offset, x + offset + bounding_box[2], y + offset + bounding_box[3]],
                               outline=line_color)




# 1
def run_on_tumor_data(wsi_path, mask_path):
    wsi = WSI()
    wsi.wsi_paths = glob.glob(os.path.join(wsi_path, '*.tif'))
    wsi.wsi_paths.sort()
    wsi.mask_paths = glob.glob(os.path.join(mask_path, '*.tif'))
    wsi.mask_paths.sort()
    wsi.wsi_type = util.get_type(wsi_path)
    wsi.vis = False

    print('Patch Type: %s' % wsi.wsi_type)
    wsi.f.write('Patch Type: %s\n' % wsi.wsi_type)
    '''
    flag：统计每张wsi可以分割多少patch
    tumor_patches: patch中白色部分占比达40%以上就认定其为tumor patch，计算数据集中tumor patch的个数
    '''
    flag = 0
    tumor_patches = 0
    wsi_num = len(wsi.wsi_paths)
    mask_num = len(wsi.mask_paths)
    assert wsi_num == mask_num, "the amount of wsi image and mask image is not equal"

    print("Level %d is used for extract patches" % util.LEVEL_USED)
    wsi.f.write('LEVEL USED: %d\n\n' % util.LEVEL_USED)
    wsi.f.flush()
    for i in range(wsi_num):

        wsi.bpatches = []
        wsi_path = wsi.wsi_paths[i]
        mask_path = wsi.mask_paths[i]

        wsi_file = util.get_filename_from_path(wsi_path)
        mask_file = util.get_filename_from_path(mask_path)

        assert wsi_file == mask_file, "the index of wsi image and mask image is not equal"
        print("Extract patches from image %s" % wsi_file)
        wsi.index = int(wsi_file)

        wsi.preFlag = flag
        preTumor = tumor_patches
        if wsi.read_tumor_wsi(wsi_path, mask_path):
            flag, tumor_patches = wsi.find_roi_tumor(flag, tumor_patches)

        patch_per_wsi = flag - wsi.preFlag
        tumor_per_wsi = tumor_patches - preTumor

        print("Extract %d patches from image %s" % (patch_per_wsi, wsi_file))
        print("Extract %d tumor patches from image %s, ratio %.3f" % (tumor_per_wsi, wsi_file, tumor_per_wsi/patch_per_wsi))
        wsi.f.write("Extract %d patches from image %s\n" % (patch_per_wsi, wsi_file))
        wsi.f.write("Extract %d tumor patches from image %s, ratio %.3f\n" % (tumor_per_wsi, wsi_file, tumor_per_wsi/patch_per_wsi))
        # if flag > wsi.patch_num:
        #     break

        wsi.draw_bbox(wsi.bpatches, (255, 0, 0), 2)  # blue bounding patches
        cv2.imwrite(os.path.join(util.THESIS_FIGURE_DIR, wsi_file) + '_bpatches_rgb_image.png', np.array(wsi.rgb_image_pil))

    print("Extract patches has finished successfully.\nTotal patch number: %d\nTotal tumor number: %d\nRatio: %.3f" % (flag - 1, tumor_patches, tumor_patches/flag))
    wsi.f.write("Extract patches has finished successfully.\nTotal patch number: %d\nTotal tumor number: %d\nRatio: %.3f\n" % (flag - 1, tumor_patches, tumor_patches/flag))
    wsi.f.close()
    # print("In %d patches, there are %d patches contain tumor area" % (flag - preFlag, has_white_areas))

def run_on_normal_data():
    wsi = WSI()
    wsi.wsi_paths = glob.glob(os.path.join(util.NORMAL_WSI_PATH, '*.tif'))
    wsi.wsi_paths.sort()

    wsi.index = 0
    # wsi_paths = ops.wsi_paths[30:]
    # ops.read_wsi(WSI_FILE_NAME)
    # ops.find_roi()
    while True:
        wsi_path = wsi.wsi_paths[wsi.index]
        print(wsi_path)
        if wsi.read_normal_wsi(wsi_path):
            wsi.find_roi_normal()
            if not wsi.wait():
                break
        else:
            if wsi.key == 81:
                wsi.index -= 1
                if wsi.index < 0:
                    wsi.index = len(wsi.wsi_paths) - 1
            elif wsi.key == 83:
                wsi.index += 1
                if wsi.index >= len(wsi.wsi_paths):
                    wsi.index = 0


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    run_on_tumor_data(util.train_wsi_path, util.train_mask_path)

