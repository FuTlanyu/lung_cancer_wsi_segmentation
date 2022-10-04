import numpy as np
import torch
from torchvision import transforms
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('checkpoints3/fcn_model_11.pt')  # 加载模型
model = model.to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def dice_coefficient(a, b):
    """dice coefficient 2nt/na + nb."""
    a_bigrams = set(a)
    b_bigrams = set(b)
    overlap = len(a_bigrams & b_bigrams)
    return overlap * 2.0/(len(a_bigrams) + len(b_bigrams))


if __name__ == '__main__':
    root_img = "../../data/data27255/packet1/patch_img/88"
    root_mask = "../../data/data27255/packet1/gt_mask/88"
    filelist = os.listdir(root_img)

    count = 0
    dices = []
    
    XY_total = 0.0000
    X_total = 0.0000
    Y_total = 0.0000

    for file in filelist:
        # if count == 1000:
        #     break
        img_name = root_img + "/" + file
        mask_name = root_mask + "/" + file
        imgA = cv2.imread(img_name)
        imgB = cv2.imread(mask_name)
        imgB = imgB / 255

        imgA = transform(imgA)
        imgA = imgA.to(device)
        imgA = imgA.unsqueeze(0)
        output = model(imgA)
        output = torch.sigmoid(output)

        output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        output_np = np.argmin(output_np, axis=1)

        a = np.squeeze(output_np[0, ...])

        # dice_coef_theoretical()
        # a = a * 255

        # cor = 0
        # incor = 0

        XY = 0.0000
        X = 0.0000
        Y = 0.0000
        rows, cols = a.shape[:2]
        for row in range(rows):
            for col in range(cols):
                if a[row, col] == 1:
                    X += 1
                    X_total += 1
                    if imgB[row, col][0] == 1:
                        XY += 1
                        XY_total += 1
                if imgB[row, col][0] == 1:
                    Y += 1
                    Y_total += 1

        dice = (2 * XY) / (X + Y + 0.0001)
        print(file, "\t", dice)
        # print("incor:", incor / (cor + incor))
        if Y > 0:
            dices.append(dice)
        count += 1

    # img_name = r'D:/ChromeDownload/patch2/patch_image/23640.png'  # 预测的图片
    # mask_name = r'D:/ChromeDownload/patch2/mask_image/23640.png'
    print(np.mean(dices))
    print(len(dices))
    
    dice_total = (2 * XY_total) / (X_total + Y_total + 0.0001)
    print("total dice:", dice_total)
