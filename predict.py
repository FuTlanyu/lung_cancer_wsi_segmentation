import argparse
import logging
import os
import glob
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask_2, plot_img_and_mask_3
from utils.dataset import BasicDataset
import util

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))
    # return Image.fromarray(mask.astype(np.uint8))


if __name__ == "__main__":

    MODEL_PATH = util.model_dir + 'level1/CP_epoch14_0.4660.pth'
    scale = 0.25
    mask_threshold = 0.5
    no_save = False
    viz = False

    net = UNet(n_channels=3, n_classes=1)
    print("Loading model {}".format(MODEL_PATH))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded !")

    dirs = os.listdir(util.test_patch_dir)
    print(dirs)

    for dir in dirs:
        print('Inject test patches from image %s to the trained model' % dir)
        patch_num = len(os.listdir(util.test_patch_dir + dir))
        for index in range(patch_num):
            index = str(index)
            print("Predicting image {} ...".format(index))
            fn = util.test_patch_dir + dir + '/' + index + '.png'
            img = Image.open(fn)

            predicted_mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=scale,
                               out_threshold=mask_threshold,
                               device=device)

            if not no_save:
                # 保存的是0-255的预测图像
                result = mask_to_image(predicted_mask)
                save_dir = util.predicted_mask_dir + dir
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                save_path = save_dir + '/' + index + '.png'
                result.save(save_path)

                # print("Mask saved to {}".format(save_path))

            if viz:
                # print("Visualizing results for image {}, close to continue ...".format(fn))
                gt_mask_path = util.gt_mask_dir + dir + '/' + index + '.png'
                gt_mask = imageio.imread(gt_mask_path)
                gt_mask = np.array(gt_mask)

                plot_img_and_mask_3(dir, index, img, predicted_mask, gt_mask, util.test_result_vis_dir, False)
