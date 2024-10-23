from PIL import Image
import numpy as np
import cv2

def make_stick(backimg,sticker,x,y,factor=1):
    backimg = np.array(backimg)
    r,g,b = cv2.split(backimg)
    background = cv2.merge([b, g, r])
    
    base,_ = make_basemap(background.shape[1],background.shape[0],sticker,x=x,y=y)
    r,g,b,a = cv2.split(base)
    foreGroundImage = cv2.merge([b, g, r,a])

    b,g,r,a = cv2.split(foreGroundImage)
    foreground = cv2.merge((b,g,r))
    
    alpha = cv2.merge((a,a,a))

    foreground = foreground.astype(float)
    background = background.astype(float)
    
    alpha = alpha.astype(float)/255
    alpha = alpha * factor
    
    foreground = cv2.multiply(alpha,foreground)
    background = cv2.multiply(1-alpha,background)
    
    outarray = foreground + background

    b, g, r = cv2.split(outarray)
    outarray = cv2.merge([r, g, b])
    outImage = Image.fromarray(np.uint8(outarray))
    return outImage

def change_sticker(sticker,scale):
    new_weight = int(sticker.size[0]/scale)
    new_height = int(sticker.size[1]/scale)
    sticker = sticker.resize((new_weight,new_height), Image.Resampling.LANCZOS)
    return sticker

def make_basemap(width,height,sticker,x,y):
    layer = Image.new('RGBA',(width,height),(255,255,255,0)) # white and transparent
    layer.paste(sticker,(x,y))
    base = np.array(layer)
    alpha_matrix = base[:,:,3]
    basemap = np.where(alpha_matrix!=0,1,0)
    return base,basemap