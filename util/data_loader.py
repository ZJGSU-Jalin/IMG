import numpy as np
import torch
import torch.utils.data
from util.data_util import pad_seq, pad_char_seq, pad_video_seq
import random
import math

def get_saliency(s,e,length,max_n=2):
    if s>e:
        s=e
    if s!=e:
        pos=random.sample(range(s,e+1),k=max_n)
    else:
        pos=[s,s]
    neg_list=list(range(0,s))+list(range(e+1,length))
    if len(neg_list)<max_n:
        neg=[0,0]
    else:
        neg=random.sample(neg_list,k=max_n)

    return pos,neg
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_features, audio_features,text_features, mode='test'):
        super(Dataset, self).__init__()
        self.dataset = dataset       
        for key in video_features.keys():
            if len(video_features[key].shape) == 4:
                video_features[key] = video_features[key].squeeze()
        
        self.video_features = video_features
        self.audio_features = audio_features
        self.text_features = text_features

    def __getitem__(self, index):
        record = self.dataset[index]
        video_feature = self.video_features[record['vid']]
        audio_feature = self.audio_features[record['vid']]
        if self.text_features is None:
            text_features=None
        else:
            text_features=self.text_features[record['qid']]

        s_ind, e_ind = int(record['s_ind']), int(record['e_ind'])
        pos_clip_indices,neg_clip_indices=get_saliency(s_ind,e_ind,len(video_feature))
        word_ids, char_ids = record['w_ids'], record['c_ids']
        return record, video_feature, audio_feature,text_features, word_ids, char_ids, s_ind, e_ind,pos_clip_indices,neg_clip_indices

    def __len__(self):
        return len(self.dataset)

def train_collate_fn(data):
    
    records, video_features, audio_features,text_features, word_ids, char_ids, s_inds, e_inds,pos_clip_indices,neg_clip_indices = zip(*data)
    # process word ids
    word_ids, _ = pad_seq(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)
    if text_features[0] is None:
        tfeats=tfeat_lens=None
    else:
        tfeats, tfeat_lens = pad_video_seq(text_features)
        tfeats = np.asarray(tfeats, dtype=np.float32)
        tfeat_lens = np.asarray(tfeat_lens, dtype=np.int32)
        tfeats = torch.tensor(tfeats, dtype=torch.float32)
        tfeat_lens = torch.tensor(tfeat_lens, dtype=torch.int64)
    # process audio features
    afeats, afeat_lens = pad_video_seq(audio_features)
    afeats = np.asarray(afeats, dtype=np.float32)
    afeat_lens = np.asarray(afeat_lens, dtype=np.int32)  
    # process labels
    max_len = np.max(vfeat_lens)
    batch_size = vfeat_lens.shape[0]
    s_labels = np.asarray(s_inds, dtype=np.int64)
    e_labels = np.asarray(e_inds, dtype=np.int64)
    h_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
    extend = 0
    for idx in range(batch_size):
        st, et = s_inds[idx], e_inds[idx]
        cur_max_len = vfeat_lens[idx]
        extend_len = round(extend * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, cur_max_len - 1)
            h_labels[idx][st_:(et_ + 1)] = 1
        else:
            h_labels[idx][st:(et + 1)] = 1
    



    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    afeats = torch.tensor(afeats, dtype=torch.float32)

    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)
    h_labels = torch.tensor(h_labels, dtype=torch.int64)
    pos = torch.tensor(pos_clip_indices, dtype=torch.int64)
    neg = torch.tensor(neg_clip_indices, dtype=torch.int64)
    return records, vfeats, vfeat_lens, afeats,tfeats,tfeat_lens, word_ids, char_ids, s_labels, e_labels,h_labels,pos,neg


def test_collate_fn(data):
    records, video_features, audio_features,text_features, word_ids, char_ids, s_inds, e_inds,pos_clip_indices,neg_clip_indices = zip(*data)

    # process word ids
    word_ids, _ = pad_seq(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)
    # process audio features
    afeats, afeat_lens = pad_video_seq(audio_features)
    afeats = np.asarray(afeats, dtype=np.float32)
    afeat_lens = np.asarray(afeat_lens, dtype=np.int32)
    if text_features[0] is None:
        tfeats=tfeat_lens=None
    else:
        tfeats, tfeat_lens = pad_video_seq(text_features)
        tfeats = np.asarray(tfeats, dtype=np.float32)
        tfeat_lens = np.asarray(tfeat_lens, dtype=np.int32)
        tfeats = torch.tensor(tfeats, dtype=torch.float32)
        tfeat_lens = torch.tensor(tfeat_lens, dtype=torch.int64)
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    afeats = torch.tensor(afeats, dtype=torch.float32)
    afeat_lens = torch.tensor(afeat_lens, dtype=torch.int64)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    s_labels = np.asarray(s_inds, dtype=np.int64)
    e_labels = np.asarray(e_inds, dtype=np.int64)
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)
    return records, vfeats, vfeat_lens, afeats,tfeats,tfeat_lens, word_ids, char_ids, s_labels, e_labels

def get_train_loader(dataset, video_features, audio_features,text_features, configs):
    train_set = Dataset(dataset=dataset, video_features=video_features, audio_features=audio_features,text_features=text_features, mode='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=True,
                                               collate_fn=train_collate_fn, num_workers=0)
    return train_loader

def get_test_loader(dataset, video_features, audio_features,text_features, configs):
    test_set = Dataset(dataset=dataset, video_features=video_features, audio_features=audio_features,text_features=text_features,mode='test')
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=configs.batch_size, shuffle=False,
                                              collate_fn=test_collate_fn, num_workers=0)
    return test_loader