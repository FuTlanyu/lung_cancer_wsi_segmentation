import matplotlib.pyplot as plt
import os

def plot_img_and_mask_3(wsi, index, img, predicted_mask, gt_mask, save_dir = None, isShow = False):
    classes = predicted_mask.shape[2] if len(predicted_mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 2)
    # 坐标原点位于左下角
    plt.suptitle(f'WSI number: {wsi}  Patch number: {index}', y=0.2, fontsize=16)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(predicted_mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(predicted_mask)
        ax[2].set_title(f'Ground truth mask')
        ax[2].imshow(gt_mask)
    plt.xticks([]), plt.yticks([])
    if save_dir:
        save_path = save_dir + wsi
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path + '/' + index + '.png')
    if isShow:
        plt.show()

    plt.close()

def plot_img_and_mask_2(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

    plt.close()
