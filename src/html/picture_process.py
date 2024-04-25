import numpy as np
import cv2
import os, os.path
import sys
from PIL import Image, ImageEnhance
from skimage.morphology import area_closing, area_opening
from scipy.signal import convolve2d
from tqdm import tqdm

def rotate_image(image, angle, center=None):
    if center is not None:
        image_center = center
    else:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def triangle_mask(filename, convs):
    img = cv2.imread(filename)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,70,0]), np.array([15,255,150]))
    
    mask = area_opening(mask, 2300)
    mask = area_closing(mask, 2000)

    conv_res = []
    for conv_num in tqdm(range(len(convs))):
        conv_res.append(convolve2d(mask.astype('int64'), convs[conv_num].astype('int64'), mode='valid'))

    conv_res = sum(conv_res)
    conv_res = (conv_res > 600000000)
    conv_res = conv_res.astype('uint8')

    num_triags, _ = cv2.connectedComponents(conv_res)
    scale = (mask.sum() / (num_triags - 1))
    mask = area_opening(mask, scale / 380)

    return mask

def get_info(filename, mask, output):
    original_stdout = sys.stdout
    sys.stdout = output

    img = Image.open(filename)
    enhancer = ImageEnhance.Brightness(img)
    img = np.array(enhancer.enhance(1.5))[:, :, ::-1]

    num_triags, labels = cv2.connectedComponents(mask)
    print(num_triags - 1)

    size = np.prod(labels.shape)

    for t in range(1, num_triags):
        cords = np.where(labels == t)
        print(int(cords[1].mean()), int(cords[0].mean()), sep=', ', end='; ')

        triag_mask = (labels == t).astype('uint8')
        triag = cv2.bitwise_and(img, img, mask = triag_mask)

        numbers = []
        
        hsv = cv2.cvtColor(triag, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (20,0,120), (30, 120, 255)).sum()
        green = cv2.inRange(hsv, (36,0,0), (70, 255, 255)).sum()
        yellow = cv2.inRange(hsv, (15,200,200), (36, 255, 255)).sum()
        blue = cv2.inRange(hsv, (110,0,0), (130, 255, 255)).sum()
        red = cv2.inRange(hsv, (0,190,200), (10, 255, 255)).sum() + cv2.inRange(hsv, (350,190,200), (360, 255, 255)).sum()
        
        if red > 1000:
            numbers.append(5)

        colors = [blue, yellow, green, white]
        th1 = np.array([1100, 1000, 200, 1000]) / 623808 * size
        th2 = np.array([60000, 25000, 9000, 4000]) / 623808 * size

        for idx, val in enumerate(colors):
            if val > th1[idx]:
                numbers.append(4 - idx)
                if val > th2[idx]:
                    numbers.append(4 - idx)

        if len(numbers) == 0:
            numbers.append(5)
            numbers.append(5)
        if len(numbers) > 3:
            numbers = numbers[:3]
        numbers += [0] * (3 - len(numbers))

        for i in range(len(numbers) - 1):
            print(numbers[i], end=', ')
        print(numbers[-1])
    
    print()
    sys.stdout = original_stdout
