import os
import glob
import random
import numpy as np
import torch
import torch.utils.data
import torch.backends.cudnn
from tqdm import tqdm
from util.data_util import index_to_time
import pickle
import json

def set_th_config(seed):
    if not seed:
        return 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def filter_checkpoints(model_dir, suffix='t7', max_to_keep=5):
    model_paths = glob.glob(os.path.join(model_dir, '*.{}'.format(suffix)))
    if len(model_paths) > max_to_keep:
        model_file_dict = dict()
        suffix_len = len(suffix) + 1
        for model_path in model_paths:
            step = int(os.path.basename(model_path).split('_')[1][0:-suffix_len])
            model_file_dict[step] = model_path
        sorted_tuples = sorted(model_file_dict.items())
        unused_tuples = sorted_tuples[0:-max_to_keep]
        for _, model_path in unused_tuples:
            os.remove(model_path)


def get_last_checkpoint(model_dir, suffix='t7'):
    model_filenames = glob.glob(os.path.join(model_dir, '*.{}'.format(suffix)))
    model_file_dict = dict()
    suffix_len = len(suffix) + 1
    for model_filename in model_filenames:
        step = int(os.path.basename(model_filename).split('_')[1][0:-suffix_len])
        model_file_dict[step] = model_filename
    sorted_tuples = sorted(model_file_dict.items())
    last_checkpoint = sorted_tuples[-1]
    print('testing: ', last_checkpoint[1])
    return last_checkpoint[1]


def convert_length_to_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.size()[0], max_len) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def calculate_iou_accuracy(ious, threshold):
    total_size = float(len(ious))
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / total_size * 100.0


def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)

def eval_test(model, data_loader, device, mode='test', epoch=None, global_step=None, configs=None):
    ious = []
    iousa=[]
    iousv=[]
    rec = []
    # lines=[]
    count=0
    count_v=0
    sum_a=0
    sum_v=0
    # dic=dict()

    with torch.no_grad():
        result_line=[]
        for idx, (records, vfeats, vfeat_lens, afeats,tfeats, tfeat_lens, word_ids, char_ids, s_labels, e_labels ) in tqdm(
                enumerate(data_loader), total=len(data_loader), desc='evaluate {}'.format(mode)):
            # prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
            afeats= afeats.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            # generate mask
            query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
            video_mask = convert_length_to_mask(vfeat_lens).to(device)
            
            s_labels, e_labels = s_labels.to(device), e_labels.to(device)
            if tfeats==None:
                start_logits_av, end_logits_av,start_logits_v, end_logits_v,start_logits_a, end_logits_a,v_score,a_score,av_score,parama = model(word_ids, char_ids, vfeats, video_mask, afeats,query_mask,1)
            else:
                tfeats, tfeat_lens = tfeats.to(device), tfeat_lens.to(device)
                text_mask = convert_length_to_mask(tfeat_lens).to(device)
                start_logits_av, end_logits_av,start_logits_v, end_logits_v,start_logits_a, end_logits_a,v_score,a_score,av_score,parama = model(vfeats, video_mask, afeats,tfeats, text_mask,1)
    

            count+=start_logits_av.shape[0]

            start_logits = start_logits_av
            end_logits = end_logits_av

            start_logitsa = start_logits_a
            end_logitsa = end_logits_a

            start_logitsv = start_logits_v
            end_logitsv = end_logits_v

            start_indices, end_indices = model.extract_index(start_logits, end_logits)
            start_indices = start_indices.cpu().numpy()
            end_indices = end_indices.cpu().numpy()
            
            start_indicesa, end_indicesa = model.extract_index(start_logitsa, end_logitsa)
            start_indicesa = start_indicesa.cpu().numpy()
            end_indicesa = end_indicesa.cpu().numpy()

            start_indicesv, end_indicesv = model.extract_index(start_logitsv, end_logitsv)
            start_indicesv = start_indicesv.cpu().numpy()
            end_indicesv = end_indicesv.cpu().numpy()
            
            for record, start_index, end_index,start_indexa, end_indexa,start_indexv, end_indexv in zip(records, start_indices, end_indices, start_indicesa, end_indicesa, start_indicesv, end_indicesv):
                
                start_time, end_time = index_to_time(start_index, end_index, record["v_len"], record["duration"])
                iou = calculate_iou(i0=[start_time, end_time], i1=[record["s_time"], record["e_time"]])

                start_timea, end_timea = index_to_time(start_indexa, end_indexa, record["v_len"], record["duration"])
                ioua = calculate_iou(i0=[start_timea, end_timea], i1=[record["s_time"], record["e_time"]])

                start_timev, end_timev = index_to_time(start_indexv, end_indexv, record["v_len"], record["duration"])
                iouv = calculate_iou(i0=[start_timev, end_timev], i1=[record["s_time"], record["e_time"]])

                
                ious.append(iou)
                rec.append(record.copy())
                iousa.append(ioua)
                iousv.append(iouv)

                rec[-1]['v_pre_s'] = start_time
                rec[-1]['v_pre_e'] = end_time

    if configs != None and configs.save_predictions != None:
        with open(configs.save_predictions, 'wb') as f:
            pickle.dump(rec, f)

    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    mi = np.mean(ious) * 100.0

    r1i3a = calculate_iou_accuracy(iousa, threshold=0.3)
    r1i5a = calculate_iou_accuracy(iousa, threshold=0.5)
    r1i7a = calculate_iou_accuracy(iousa, threshold=0.7)
    mia = np.mean(iousa) * 100.0

    r1i3v = calculate_iou_accuracy(iousv, threshold=0.3)
    r1i5v = calculate_iou_accuracy(iousv, threshold=0.5)
    r1i7v = calculate_iou_accuracy(iousv, threshold=0.7)
    miv = np.mean(iousv) * 100.0

    score_str = "Epoch {}, Step {}:\n".format(epoch, global_step)
    score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
    score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
    score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
    score_str += "mean IoU: {:.2f}\n".format(mi)

    score_stra = "Epoch {}, Step {}:\n".format(epoch, global_step)
    score_stra += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3a)
    score_stra += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5a)
    score_stra += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7a)
    score_stra += "mean IoU: {:.2f}\n".format(mia)

    score_strv = "Epoch {}, Step {}:\n".format(epoch, global_step)
    score_strv += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3v)
    score_strv += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5v)
    score_strv += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7v)
    score_strv += "mean IoU: {:.2f}\n".format(miv)

    return r1i3, r1i5, r1i7, mi, score_str, score_stra, score_strv

def grader(start_logits_av, end_logits_av, \
        start_logits_v, end_logits_v, \
        s_labels, e_labels, ext):
    soft_start_logits_av = torch.softmax(start_logits_av, dim=-1)
    soft_end_logits_av = torch.softmax(end_logits_av, dim=-1)
    soft_start_logits_v = torch.softmax(start_logits_v, dim=-1)
    soft_end_logits_v = torch.softmax(end_logits_v, dim=-1)
    batch_size = s_labels.shape[0]
    
    s_av, s_v, rho = torch.zeros(batch_size).type_as(start_logits_av), \
            torch.zeros(batch_size).type_as(start_logits_av), \
            torch.zeros(batch_size).type_as(start_logits_av)
    seq_len = start_logits_av.shape[1]
    eps = 1e-5
    for i in range(batch_size):
        s_label = s_labels[i]
        e_label = e_labels[i]
        s_ind_s = max(0, s_label-ext)
        s_ind_e = min(seq_len, s_label+ext+1)
        e_ind_s = max(0, e_label-ext)
        e_ind_e = min(seq_len, e_label+ext+1)
        s_av[i] = (sum(soft_start_logits_av[i, s_ind_s:s_ind_e])+sum(soft_end_logits_av[i, e_ind_s:e_ind_e])) / 2
        s_v[i] = (sum(soft_start_logits_v[i, s_ind_s:s_ind_e])+sum(soft_end_logits_v[i, e_ind_s:e_ind_e])) / 2
        rho[i] = torch.sigmoid(torch.log((s_av[i] + eps) / (s_v[i] + eps)))
    
    return rho, s_av, s_v

