import copy, os, torch, cv2
from torch import nn
import torch.optim as optim
from torchvision.models.googlenet import GoogLeNet
from torchvision import transforms, datasets

# import json
# with open('./datasets/ImageNet_class_index.json') as f:
#     cls_dict = json.loads(f.read())
# with open('./datasets/ImageNet_val_label.txt') as f:
#     ground_truth, ground_truth_dict = f.readlines(), dict()
#     for sample in ground_truth:
#         filename, label = sample.split()
#         ground_truth_dict[filename] = label
# pred_cls = 2 # index of predicted class, starting from 0
# print(ground_truth_dict['ILSVRC2012_val_00002338.JPEG'] == cls_dict[str(pred_cls)][0])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
def train(num_cls=20, dataset_name='cls_dataset', total_epoch=100, batch_size=64, lr=0.001, momentum=0.9):
    predictor = GoogLeNet(num_classes=num_cls, aux_logits=False, init_weights=False).to(device)
    state_dict = torch.load('./models/googlenet-1378be20.pth', map_location=device)
    state_dict_new = copy.deepcopy(state_dict)
    for key in state_dict.keys():
        if key.startswith('fc.'): del state_dict_new[key]
    predictor.load_state_dict(state_dict_new, strict=False)
    predictor.train()

    dataset = datasets.ImageFolder(f'./datasets/{dataset_name}', transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(predictor.parameters(), lr=lr, momentum=momentum)

    for epoch in range(total_epoch):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            img, lbl = data
            optimizer.zero_grad()
            output = predictor(img.to(device))
            loss = criterion(output, lbl.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[{epoch}, {i:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
        os.makedirs('./models', exist_ok=True)
        torch.save(predictor.state_dict(), f'./models/googlenet-{dataset_name}-{epoch}.pth')
        
@torch.no_grad()
def evaluate(ckpt_name, num_cls=20, dataset_name='clean_cls_samples', batch_size=64):
    predictor = GoogLeNet(num_classes=num_cls, aux_logits=False, init_weights=False).to(device)
    state_dict = torch.load(f'./models/{ckpt_name}', map_location=device)
    predictor.load_state_dict(state_dict, strict=True)
    predictor.eval()
    
    dataset = datasets.ImageFolder(f'./datasets/{dataset_name}', transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    total_acc_cnt, total_cnt = 0.0, 0.0
    for sample in loader:
        img, lbl = sample
        pred = predictor(img.to(device))
        pred_prob = torch.nn.functional.softmax(pred, dim=1)
        pred_prob, pred_cls = torch.topk(pred_prob, 1, dim=1)
        pred_prob, pred_cls = pred_prob.squeeze(), pred_cls.squeeze()
        acc_cnt = torch.sum(pred_cls == lbl.to(device)).item()
        total_acc_cnt = total_acc_cnt + acc_cnt
        total_cnt = total_cnt + img.shape[0]
    accuracy = total_acc_cnt/total_cnt*100
    return accuracy

class DatasetWrapper(datasets.VisionDataset):
    def __init__(self, dataset, *args, **kwargs):
        super(self.__class__, self).__init__(root=args[0])
        self.dataset = dataset(*args, **kwargs)
        
    def __getitem__(self, index: int):
        file_path = self.dataset.imgs[index][0]
        return file_path, index, *self.dataset.__getitem__(index)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def get_filename(self, idx):
        return 

@torch.no_grad()
def attack_evaluate(ckpt_name, num_cls=20, adv_dataset_name='adv_samples', clean_dataset_name='clean_cls_samples'):
    predictor = GoogLeNet(num_classes=num_cls, aux_logits=False, init_weights=False).to(device)
    state_dict = torch.load(f'./models/{ckpt_name}', map_location=device)
    predictor.load_state_dict(state_dict, strict=True)
    predictor.eval()
    
    clean_dataset = DatasetWrapper(datasets.ImageFolder, f'./datasets/{clean_dataset_name}', transform=preprocess)
    adv_dataset = datasets.ImageFolder(f'./datasets/{adv_dataset_name}', transform=preprocess)
    clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    actc_score, ald_score = 0.0, 0.0
    for clean_sample in clean_loader:
        clean_img_path, idx, clean_img, lbl = clean_sample
        adv_img, adv_lbl = adv_dataset[idx]
        clean_img_filename = os.path.basename(clean_img_path[0])
        adv_img_filename = os.path.basename(adv_dataset.imgs[idx][0])
        assert lbl == adv_lbl and (clean_img_filename == adv_img_filename)
        pred = predictor(adv_img.unsqueeze(dim=0).to(device))
        pred_prob = torch.nn.functional.softmax(pred, dim=1)
        _, pred_cls = torch.topk(pred_prob, 1, dim=1)
        attack_succeed = pred_cls.squeeze().item() != lbl.item()
        diff_map = (clean_img[0]-adv_img) != 0 # True or False on three channels
        diff_map = torch.sum(diff_map, dim=0) # Combine the difference to single channel
        diff_cnt = torch.sum(diff_map != 0) # Count on the single channel bool map
        diff_map[diff_map != 0] = 255 # Convert to black/white mask with only 0 and 255
        total_area = adv_img.shape[-1]*adv_img.shape[-2]
        diff_ratio = (diff_cnt/total_area).item()
        diff_map_cv = diff_map.to(device='cpu', dtype=torch.uint8).numpy()
        num_region, _, _, _ = cv2.connectedComponentsWithStats(diff_map_cv)
        is_punished = diff_ratio > 0.02 or num_region-1 > 5 # num_region includes background
        if not is_punished:
            ald_score = ald_score + (0.02 if not attack_succeed else diff_ratio)
            actc_score = actc_score + pred_prob.squeeze()[lbl].item()
    actc_score = actc_score / len(clean_dataset)
    ald_score = ald_score / (0.02*len(clean_dataset))
    total_score = 0.5*((1-actc_score)+(1-ald_score))*100
    return total_score

if __name__ == '__main__':
    # train()
    # for i in range(80):
    #     acc = evaluate(ckpt_name=f'googlenet-cls_dataset-{i}.pth')
    #     print(f'Accuracy of epoch {i}: {acc:.2f}%')
    # print('Best accuracy: {:.3f}%'.format(evaluate(ckpt_name='googlenet-cls_dataset-best.pth')))
    total_score = attack_evaluate(ckpt_name='googlenet-cls_dataset-best.pth')
    print(f'Total score: {total_score}')