import os
import glob
import json
import pickle
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as F
import torch
def rescale(src, tgt):
    tgt_len, src_len = tgt.shape[0], src.shape[0]
    src_rescale = np.zeros((tgt.shape[0],src.shape[1]))
    idxs = np.arange(0, tgt_len + 1, 1.0) / tgt_len * src_len
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > src_len - 1] = src_len - 1

    for i in range(tgt.shape[0]):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            src_rescale[i,:]=np.mean(src[s_idx:e_idx], axis=0)
        else:
            src_rescale[i,:]=src[s_idx]

    return src_rescale
def load_json(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, mode='w', encoding='utf-8') as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_lines(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        return [e.strip("\n") for e in f.readlines()]


def save_lines(data, filename):
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write("\n".join(data))


def load_pickle(filename):
    with open(filename, mode='rb') as handle:
        data = pickle.load(handle)
        return data


def save_pickle(data, filename):
    with open(filename, mode='wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_video_features(root, max_position_length,audio_feature):
    video_features = dict()
    vid_list=[]
    filenames = glob.glob(os.path.join(root, "*.npy"))
    for filename in tqdm(filenames, total=len(filenames), desc="load video features"):
        video_id = filename.split("/")[-1].split(".")[0]
        feature = np.load(filename)
        if max_position_length is None:
            video_features[video_id] = feature
        else:
            new_feature = visual_feature_sampling(feature, max_num_clips=max_position_length)
            if video_id in audio_feature:
                audio_feature[video_id]=rescale(audio_feature[video_id],new_feature)
            video_features[video_id] = new_feature
    return video_features,audio_feature # ['id': (numpy)feature]


def load_i3d_video_features(root, max_position_length,audio_feature):
    video_features = dict()
    vid_list=[]
    filenames = glob.glob(os.path.join(root, "*.npy"))
    for filename in tqdm(filenames, total=len(filenames), desc="load video features"):
        video_id = filename.split("/")[-1].split(".")[0]
        feature = np.load(filename)
        if max_position_length is None:
            video_features[video_id] = feature
        else:
            new_feature = visual_feature_sampling(feature, max_num_clips=max_position_length)
            if video_id in audio_feature:
                audio_feature[video_id]=rescale(audio_feature[video_id],new_feature)
            video_features[video_id] = new_feature
    return video_features,audio_feature # ['id': (numpy)feature]

def load_video_clip_sf_features(root, max_position_length,audio_feature):
    sf_features = dict()
    sf="slowfast_features"
    sf_file=os.listdir(os.path.join(root,sf))
    for i in tqdm(sf_file, total=len(sf_file), desc="load slow fast features"):
        vid=i.split(".")[0]
        filename=os.path.join(root,sf,i)
        sf_features[vid]=np.load(filename)['features']

    video_features = dict()
    vroot=root+"/visual_features"
    filenames = glob.glob(os.path.join(vroot, "*.npy"))
    for filename in tqdm(filenames, total=len(filenames), desc="load clip video features"):
        video_id = filename.split("/")[-1].split(".")[0]
        feature = np.load(filename)
        new_feature = visual_feature_sampling(feature, max_num_clips=max_position_length)
        sf_features[video_id]=rescale(sf_features[video_id],new_feature)
        new_feature = np.concatenate((sf_features[video_id],new_feature),axis=-1)
        if video_id in audio_feature:
            audio_feature[video_id]=rescale(audio_feature[video_id],new_feature)
            video_features[video_id] = new_feature
        else:
            print("missing",video_id)
        

    text_features = dict()
    troot=root+"/text_features"
    filenames = glob.glob(os.path.join(troot, "*.npy"))
    for filename in tqdm(filenames, total=len(filenames), desc="load clip text features"):
        video_id = filename.split("/")[-1].split(".")[0]
        feature = np.load(filename)
        text_features[video_id] = feature

    return video_features,audio_feature,text_features 

def load_internvideo_features(root, max_position_length,audio_feature):
    video_features = dict()

    filenames = os.listdir(os.path.join(root,"visual_features_6b"))
    for filename in tqdm(filenames, total=len(filenames), desc="load video features"):
        filename = os.path.join(root,"visual_features_6b",filename)
        video_id = filename.split("/")[-1].split(".")[0]
        feature = torch.load(filename).numpy()
        if max_position_length is None:
            video_features[video_id] = feature
        else:
            new_feature = visual_feature_sampling(feature, max_num_clips=max_position_length)
            if video_id in audio_feature:
                audio_feature[video_id]=rescale(audio_feature[video_id],new_feature)
            else:
                print("missing",video_id)
            video_features[video_id] = new_feature

    text_features = dict()
    fl = os.listdir(os.path.join(root,'llama2_txt'))
    for filename in tqdm(fl, total=len(fl), desc="load llama2 text features"):
        filename = os.path.join(root,"llama2_txt",filename)
        tid = filename.split("/")[-1].split(".")[0]
        tid = tid[:3]+"_"+tid[3:]
        feature = torch.load(filename).numpy()
        text_features[tid]=feature

    return video_features,audio_feature,text_features 


def load_audio_features(root, max_position_length, mode):
    if mode == 'VGGish':
        return load_audio_features_VGGish(root, max_position_length)
    else:
        return load_audio_features_PANN(root, max_position_length)

def fix_json(shape_json_v,shape_json_a):
    vfeat_lens = load_json(shape_json_v)
    afeat_lens = load_json(shape_json_a)
    new_shape_json={}
    for i in vfeat_lens.keys():
        if i not in afeat_lens.keys():
            continue
        new_shape_json[i]=vfeat_lens[i]
    return new_shape_json,vfeat_lens

    


def load_audio_features_PANN(root, max_position_length):
    audio_features = dict()
    filenames = glob.glob(os.path.join(root, "*.npy"))
    for filename in tqdm(filenames, total=len(filenames), desc="load audio features"):
        video_id = filename.split("/")[-1].split(".")[0]
        feature = np.load(filename)
        if max_position_length is None:
            audio_features[video_id] = feature
        else:
            new_feature = visual_feature_sampling(feature, max_num_clips=max_position_length)
            audio_features[video_id] = new_feature
    return audio_features

def load_audio_features_VGGish(root, max_position_length):
    with open(root, 'rb') as f: 
        audio_features = pickle.load(f)
    for k in audio_features.keys():
        audio_features[k] = audio_features[k].astype(np.float32)
        audio_features[k] = audio_features[k] / 255
    if max_position_length is None:
        return audio_features
    else:
        for k in audio_features.keys():
            new_feature = visual_feature_sampling(audio_features[k], max_num_clips=max_position_length)
            audio_features[k] = new_feature
        return audio_features
    
def visual_feature_sampling(visual_feature, max_num_clips):
    num_clips = visual_feature.shape[0]
    if num_clips <= max_num_clips:
        return visual_feature
    idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
    
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_visual_feature = []
    for i in range(max_num_clips):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))
        else:
            new_visual_feature.append(visual_feature[s_idx])
    new_visual_feature = np.asarray(new_visual_feature)
    return new_visual_feature


def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]
    return overlap


def time_to_index(start_time, end_time, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) / float(num_units) * duration
    e_times = np.arange(1, num_units + 1).astype(np.float32) / float(num_units) * duration
    candidates = np.stack([np.repeat(s_times[:, None], repeats=num_units, axis=1),
                           np.repeat(e_times[None, :], repeats=num_units, axis=0)], axis=2).reshape((-1, 2))
    overlaps = compute_overlap(candidates.tolist(), [start_time, end_time]).reshape(num_units, num_units)
    start_index = np.argmax(overlaps) // num_units
    end_index = np.argmax(overlaps) % num_units
    return start_index, end_index, overlaps


def index_to_time(start_index, end_index, num_units, duration):
    
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time


def pad_seq(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
        pad_tok = 0  # 0: "PAD" for words and chars, "PAD" for tags
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def pad_char_seq(sequences, max_length=None, max_length_2=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))
    if max_length_2 is None:
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
        sp, sl = pad_seq(seq, max_length=max_length_2)
        sequence_padded.append(sp)
        sequence_length.append(sl)
    sequence_padded, _ = pad_seq(sequence_padded, pad_tok=[0] * max_length_2, max_length=max_length)
    sequence_length, _ = pad_seq(sequence_length, max_length=max_length)
    return sequence_padded, sequence_length


def pad_video_seq(sequences, max_length=None):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])
    feature_length = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])
        if add_length > 0:
            add_feature = np.zeros(shape=[add_length, feature_length], dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length
