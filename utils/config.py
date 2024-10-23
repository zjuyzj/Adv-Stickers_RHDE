from torchvision import datasets

class Config(object):
    bound = 15             # critical value of the gap between label1 and label2
    id_threat = 0          # id of threat model in ['googlenet']
    sticker_name = 'bs12'
    scale = 12             # The scale of the sticker

    data_dir = './datasets/ImageNet-1K'
    idx = 0
    dataset = datasets.ImageFolder(data_dir)
    pic, gtlabel = dataset[idx]
    lbl_mapping = dataset.class_to_idx
    gtlabel = int(list(lbl_mapping.keys())[list(lbl_mapping.values()).index(gtlabel)])