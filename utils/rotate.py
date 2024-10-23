from PIL import Image
import numpy as np
import cv2

def img_to_cv(image):
    imgarray = np.array(image)
    r,g,b,a = cv2.split(imgarray)
    cvarray = cv2.merge([b, g, r, a])
    return cvarray

def rotate_bound_white_bg(imagecv, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = imagecv.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
 
    # perform the actual rotation and return the image
    rotated = cv2.warpAffine(imagecv, M, (w, h),borderValue=(255,255,255,0))
    b, g, r, a = cv2.split(rotated)
    rotated_array = cv2.merge([r, g, b, a])
    rt_sticker = Image.fromarray(np.uint8(rotated_array))
    return rt_sticker