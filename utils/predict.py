from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import GoogLeNet
import numpy as np
import torch
from models import *
import warnings

from utils import rotate
from utils import stick
from utils import mapping3d
warnings.filterwarnings("ignore")

""" perturb the image """
def perturb_image(xs, backimg, sticker,opstickercv,magnification, z_buffer, searchspace, mask):
    xs = np.array(xs)
    d = xs.ndim
    if(d==1):
        xs = np.array([xs])
    w,h = backimg.size
    
    imgs = []
    valid = []
    l = len(xs)
    for i in range(l):
        sid = int(xs[i][0])
        x = int(searchspace[sid][0])
        y = int(searchspace[sid][1])
        angle = xs[i][2]
        rt_sticker = rotate.rotate_bound_white_bg(opstickercv, angle)
        nsticker,_ = mapping3d.deformation3d(sticker,rt_sticker,magnification,z_buffer,x,y)
        outImage = stick.make_stick(backimg=backimg, sticker=nsticker, x=x, y=y, factor=xs[i][1])
        imgs.append(outImage)
        
        check_result = int(check_valid(w, h, nsticker, x, y, mask))
        valid.append(check_result)
            
    return imgs,valid

def check_valid(w, h, sticker, x, y, mask):
    _,basemap = stick.make_basemap(width=w, height=h, sticker=sticker, x=x, y=y)
    area = np.sum(basemap)
    overlap = mask * basemap
    retain = np.sum(overlap)
    if(abs(area - retain) > 15):
        return 0
    else:
        return 1

def simple_perturb(xs, backimg, sticker, searchspace, mask):
    xs = np.array(xs)
    d = xs.ndim
    if(d==1):
        xs = np.array([xs])
    w,h = backimg.size
    
    imgs = []
    valid = []
    l = len(xs)
    for i in range(l):
        print('making {}-th perturbed image'.format(i),end='\r')
        sid = int(xs[i][0])
        x = int(searchspace[sid][0])
        y = int(searchspace[sid][1])
        angle = xs[i][2]
        stickercv = rotate.img_to_cv(sticker)
        rt_sticker = rotate.rotate_bound_white_bg(stickercv, angle)
        outImage = stick.make_stick(backimg=backimg, sticker=rt_sticker, x=x, y=y, factor=xs[i][1])
        imgs.append(outImage)
        
        check_result = int(check_valid(w, h, rt_sticker, x, y, mask))
        valid.append(check_result)
    
    return imgs,valid

# """ query the model for image's classification """
# """
# def predict_type_xxx(image_perturbed):
#     top_5_cls_all = [[top-1,...,top-5],...], as a list
#     prob_all = [[vector of probability in percentage],...], as a tensor
#     return top_5_cls_all, prob_all
# """
@torch.no_grad()
def predict_type_googlenet(image_perturbed):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    predictor = GoogLeNet(init_weights=False).to(device)
    state_dict = torch.load('./models/googlenet-1378be20.pth', map_location=device)
    predictor.load_state_dict(state_dict, strict=True)
    predictor.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    def collate_fn(x):
        x = [preprocess(sample) for sample in x]
        return torch.stack(x)
    loader = DataLoader(image_perturbed, batch_size=42, shuffle=False, collate_fn=collate_fn)

    top_5_cls_all, prob_all = [], []
    for input_batch in loader:
        output = predictor(input_batch)
        prob = torch.nn.functional.softmax(output, dim=1) * 100
        _, top_5_cls = torch.topk(prob, 5, dim=1)
        top_5_cls_all.append(top_5_cls)
        prob_all.append(prob)
    return torch.cat(top_5_cls_all, dim=0).tolist(), torch.cat(prob_all, dim=0)

@torch.no_grad()
def initial_predict_googlenet(image_perturbed):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    predictor = GoogLeNet(init_weights=False).to(device)
    state_dict = torch.load('./models/googlenet-1378be20.pth', map_location=device)
    predictor.load_state_dict(state_dict, strict=True)
    predictor.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image_perturbed[0])
    input_batch = input_tensor.unsqueeze(0)

    output = predictor(input_batch)
    prob = torch.nn.functional.softmax(output[0], dim=0) * 100
    _, top_5_cls = torch.topk(prob, 5)

    return top_5_cls.tolist()