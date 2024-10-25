from PIL import Image
import numpy as np
import copy, os
import random

from heuristicsDE import differential_evolution
from utils import rotate
from utils import stick
from utils import predict
from utils import tools

from torchvision import datasets

class Attacker():
    def _reset_state(self):
        self.convert = False          # indicate whether DE needs to re-compute energies to compare with target result
        self.start = False            # whether start target attack from untarget style
        self.latter = 0               # target class
        self.generate_rank, self.generate_score, self.best_rank, self.best_score = [],[],[],[]
        self.timess = 0               # Record the query times based on batch

    def _init_sticker(self, sticker_path, sticker_scale):
        sticker_img = Image.open(sticker_path)
        scale1, scale2 = sticker_img.size[0]//23, sticker_scale
        self.magnification = scale2/scale1
        operate_sticker = stick.change_sticker(sticker_img, scale1)
        self.sticker = stick.change_sticker(sticker_img, scale2)
        self.opstickercv = rotate.img_to_cv(operate_sticker)

    def __init__(self, threat_model: str='googlenet', sticker_path='./stickers/bs12.png', sticker_scale=12, bound=15):
        self.pinf, self.ninf = 99.9999999, 0.0000001
        self.threat_model = threat_model
        self.bound = bound # critical value of the gap between label1 and label2
        self._init_sticker(sticker_path, sticker_scale)
        self._reset_state()

    def attack(self, image, gt_label, maxiter=30):
        self._reset_state()
        rank = eval('predict.initial_predict_{}([image])'.format(self.threat_model))
        if(rank[0] == gt_label):
            z_buffer = np.zeros((image.height, image.width))
            return self._attack(gt_label, image, self.sticker, self.opstickercv, self.magnification, z_buffer, maxiter=maxiter)
        else: return None, None

    def _attack(self, true_label, initial_pic, sticker, opstickercv, magnification, z_buffer, target=None, maxiter=30, popsize=40):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else true_label
        mask = np.zeros((initial_pic.height, initial_pic.width)) # valid=1, unvalid=0
        mask[0:initial_pic.height-sticker.height, 0:initial_pic.width-sticker.width] = 1
        num_space = np.sum(mask).astype(int)
        searchspace = np.zeros((num_space,2))         # store the coordinate(Image style)
        pack_searchspace = copy.deepcopy(mask)-2      # record the id, unvalid=-2
        trace_searchspace = []                        # mark whether it has been accessed
        for i in range(mask.shape[0]):
            col = [[-1] for j in range(mask.shape[1])]
            trace_searchspace.append(col)
        k = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if(mask[i][j] == 1):
                    searchspace[k] = (j,i)
                    k = k + 1
        np.random.shuffle(searchspace)
        for i in range(len(searchspace)):
            x = int(searchspace[i][0])
            y = int(searchspace[i][1])
            pack_searchspace[y][x] = int(i)
        bounds = [(0,num_space), (0.8,0.98),(0,359)]
        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs,gorb): return self.__predict_classes(xs, gorb, initial_pic, target_class, searchspace, sticker,opstickercv,magnification, z_buffer, mask, target is None)
        def callback_fn(x, convergence): return self.__attack_success(x, initial_pic, target_class, searchspace, sticker,opstickercv,magnification, z_buffer, mask, targeted_attack)
        def region_fn(xs): return self.__region_produce(xs, true_label, searchspace, pack_searchspace, trace_searchspace, initial_pic, sticker,opstickercv,magnification, z_buffer, mask)
        def ct_energy(ranks, pred_ps, valids): return self.__convert_energy(ranks, pred_ps, valids, target_class)
        raw_result = differential_evolution(predict_fn, region_fn, ct_energy, bounds, maxiter=maxiter, popsize=popsize, recombination=1, atol=-1, callback=callback_fn, polish=False)
        sid, factor, angle = int(raw_result.x[0]), raw_result.x[1], raw_result.x[2]
        x, y = int(searchspace[sid][0]), int(searchspace[sid][1])
        # Calculate some useful statistics to return from this function
        attack_image, valid = predict.perturb_image(raw_result.x, initial_pic, sticker, opstickercv, magnification, z_buffer, searchspace, mask)
        return attack_image, [x, y, factor, angle, sid]

    """ Perturb the image with the given individual(xs) and get the prediction of the model """
    def __predict_classes(self, xs, gorb, initial_pic, target_class, searchspace, sticker, opstickercv, magnification, z_buffer, mask, minimize=True):
        imgs_perturbed, valid = predict.perturb_image(xs, initial_pic, sticker, opstickercv, magnification, z_buffer, searchspace, mask)
        predictions = []
        le = len(imgs_perturbed)
        rank, pred_p = eval('predict.predict_type_{}(imgs_perturbed)'.format(self.threat_model))
        self.timess=self.timess+1
        print(f'iter = {self.timess}, start = {self.start}, convert = {self.convert}')
        for i in range(le):
            if(rank[i][0] != target_class):   # untarget
                probab = -1 * self.pinf
            else:
                label2 = rank[i][1]
                probab1 = pred_p[i][target_class].item()
                if(self.start == False):
                    probab2 = pred_p[i][label2].item()
                    a,b = 1,0
                    probab = a * probab1 - b * probab2
                elif(self.start == True):
                    probab2 = pred_p[i][self.latter].item()
                    #a,b = 0.3,0.7
                    #probab = a * probab1 - b * probab2
                    beta = 20 if self.threat_model != 'arcface' else 1
                    probab = beta*(probab1 - probab2)/probab1 + (probab1 - probab2)
            if(valid[i] == 0):
                probab = self.pinf
            predictions.append(np.array(probab))
        predictions = np.array(predictions)
        duplicate = copy.deepcopy(predictions)
        current_optimal = int(duplicate.argsort()[0])
        mingap = pred_p[current_optimal][rank[current_optimal][0]].item() - pred_p[current_optimal][rank[current_optimal][1]].item()
        if(gorb == 0):
            self.generate_rank.append([rank[current_optimal][0],rank[current_optimal][1]])
            self.generate_score.append([pred_p[current_optimal][rank[current_optimal][0]].item(),pred_p[current_optimal][rank[current_optimal][1]].item()])
        elif(gorb == 1):
            self.best_rank.append([rank[current_optimal][0],rank[current_optimal][1]])
            self.best_score.append([pred_p[current_optimal][rank[current_optimal][0]].item(),pred_p[current_optimal][rank[current_optimal][1]].item()])
        if(self.start==False and rank[current_optimal][0] == target_class and mingap <= self.bound):
            self.start = True
            self.latter = rank[current_optimal][1]
            self.convert = True
        return predictions, rank, self.convert, pred_p, valid

    def __convert_energy(self, rank, pred_p, valid, target_class):
        self.convert = False
        predictions = []
        for i in range(len(rank)):
            if(rank[i][0] != target_class):   # untarget
                probab = -1 * self.pinf
            else:
                label2 = rank[i][1]
                probab1 = pred_p[i][target_class].item()
                probab2 = pred_p[i][self.latter].item()
                #a,b = 0.3,0.7
                #probab = a * probab1 - b * probab2
                beta = 20 if self.threat_model != 'arcface' else 1
                probab = beta*(probab1 - probab2)/probab1 + (probab1 - probab2)
            if(valid[i] == 0):
                probab = self.pinf
            predictions.append(np.array(probab))
        predictions = np.array(predictions)
        return predictions    

    def ___single_predict(self, xs, initial_pic, true_label, searchspace, sticker,opstickercv,magnification, z_buffer, mask):
        imgs_perturbed, valid = predict.simple_perturb(xs, initial_pic, sticker, searchspace, mask)
        rank, pred_p = eval('predict.predict_type_{}(imgs_perturbed)'.format(self.threat_model))
        predictions = []
        for i in range(len(imgs_perturbed)):
            if(rank[i][0] != true_label):   # untarget
                probab = -1 * self.pinf
            else:
                probab = pred_p[i][true_label].item()
            if(valid[i] == 0):
                probab = self.pinf
            predictions.append(probab)
        predictions = np.array(predictions)
        return predictions

    """  If the prediction is what we want (misclassification or targeted classification), return True """
    def __attack_success(self, x, initial_pic, target_class, searchspace, sticker,opstickercv,magnification, z_buffer, mask, targeted_attack=False):
        attack_image, valid = predict.perturb_image(x, initial_pic, sticker, opstickercv, magnification, z_buffer, searchspace, mask)
        rank, _ = eval('predict.predict_type_{}(attack_image)'.format(self.threat_model))
        predicted_class = rank[0][0]
        if ((targeted_attack and predicted_class == target_class and valid[0]==1) or
            (not targeted_attack and predicted_class != target_class and valid[0]==1)):
            return True
        # NOTE: return None otherwise (not False), due to how Scipy handles its callback function

    def __region_produce(self, xs, true_label, searchspace, pack_searchspace, trace_searchspace, initial_pic, sticker, opstickercv, magnification, z_buffer, mask):
        h, w = int(mask.shape[0]), int(mask.shape[1])
        len_relative = len(xs)
        len_per = np.zeros((len_relative,1))  # the number of valid dots around the current dot
        pots = []                             # The whole set of perturbation vectors considered in inbreeding
        inbreeding = []
        for i in list(range(len_relative)):   # for each individual
            cur = int(xs[i][0])
            alp = xs[i][1]
            angle = xs[i][2]
            x = int(searchspace[cur][0])
            y = int(searchspace[cur][1])
            neighbors = tools.adjacent_coordinates(x,y,s=1)
            temp = 0
            for j in range(len(neighbors)):
                p = tools.num_clip(0,w-1,int(neighbors[j][0]))
                q = tools.num_clip(0,h-1,int(neighbors[j][1]))
                if(alp in trace_searchspace[q][p]):              # if this dot has been visited
                    judge = random.random()
                    if(judge <= 0.5):                            # change the step
                        slide = 2
                        while(1):
                            far_neighbors = tools.adjacent_coordinates(x,y,s=slide)
                            pn = int(far_neighbors[j][0])
                            qn = int(far_neighbors[j][1])
                            if(alp in trace_searchspace[qn][pn]):
                                slide = slide + 1
                            else:
                                break
                        trace_searchspace[qn][pn].append(alp)
                        attribute = pack_searchspace[qn][pn]
                        if(attribute >= 0):
                            temp = temp + 1
                            pots.append([attribute,alp,angle])
                    else:                                        # change the alpha using random
                        attribute = pack_searchspace[q][p]
                        alp_ex = random.uniform(0.8,0.98)
                        if(attribute >= 0):
                            temp = temp + 1
                            pots.append([attribute,alp_ex,angle])
                        trace_searchspace[q][p].append(alp_ex)
                else:
                    trace_searchspace[q][p].append(alp)
                    attribute = pack_searchspace[q][p]
                    if(attribute >= 0):
                        temp = temp + 1
                        pots.append([attribute,alp,angle])
            len_per[i][0] = temp
        predictions = self.___single_predict(pots, initial_pic, true_label, searchspace, sticker,opstickercv,magnification, z_buffer, mask)
        cursor = 0
        for i in range(len_relative):
            sublen = len_per[i][0]
            if(sublen != 0):
                upper = int(cursor + sublen)
                subset = predictions[int(cursor):upper]
                better = np.argsort(subset)[0]
                inbreeding.append(pots[int(cursor+better)])
            else:
                inbreeding.append(xs[i])
            cursor = cursor + sublen
        return inbreeding
    
def attack_all(dataset_name = 'clean_cls_samples', use_raw_label=False, maxiter=30):
    generated_path = './datasets/generated'
    dataset = datasets.ImageFolder(f'./datasets/{dataset_name}')
    attacker = Attacker()
    for idx in range(len(dataset)):
        img, lbl = dataset[idx]
        lbl_mapping = dataset.class_to_idx
        lbl_k, lbl_v = list(lbl_mapping.keys()), list(lbl_mapping.values())
        img_cls_name = lbl_k[lbl_v.index(lbl)]
        if use_raw_label: lbl = int(img_cls_name)
        img_filename = os.path.basename(dataset.imgs[idx][0])
        print(f'[START] {img_filename} ({idx+1}/{len(dataset)})')
        attack_img, attack_param = attacker.attack(img, lbl, maxiter=maxiter)
        print(f'[END] {img_filename} ({attack_param})')
        attack_img_folder_path = os.path.join(generated_path, img_cls_name)
        os.makedirs(attack_img_folder_path, exist_ok=True)
        if attack_img is not None: attack_img[0].save(os.path.join(attack_img_folder_path, img_filename))
        else: img.save(os.path.join(attack_img_folder_path, img_filename))

if __name__=="__main__":
    attack_all(maxiter=8)