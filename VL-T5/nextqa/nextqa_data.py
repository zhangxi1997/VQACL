from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
import re
import os, pandas

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast
from metrix import get_wups
from Question_type import All_V as Category_splits
from Question_type import All_Q


# from pywsd.utils import lemmatize_sentence
import nltk


project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = coco_dir.joinpath('images/')
coco_feature_dir = coco_dir.joinpath('features')
vqa_dir = dataset_dir.joinpath('vqa')

def load_file(file_name):
    annos = None
    if os.path.splitext(file_name)[-1] == '.csv':
        return pandas.read_csv(file_name, delimiter=',')
    with open(file_name, 'r') as fp:
        if os.path.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if os.path.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos

#use stopwords tailored for NExT-QA
stopwords = load_file('./src/stopwords.txt')


class VQAFineTuneDataset(Dataset):
    def __init__(self, split, Examplar_set, raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train', task='DO', cates=[0,1,2]):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        # self.sources = split.split(',')
        self.sources = mode

        # if self.verbose:
        #     print('Data sources: ', self.sources)

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)

        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            if args.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)


        sample_list_file = os.path.join('/data/zhangxi/nextqa/Partition_Q_v2', '{}_{}.csv'.format(mode, task))
        all_data = load_file(sample_list_file)

        cate_data = []
        self.cate_set = set()

        for idx in range(len(all_data)):
            current = all_data.loc[idx]
            if int(current['bigCate']) in cates:
                cate_data.append(current)
                self.cate_set.add(current['bigCate'])

        if len(Examplar_set) > 0:
            for _d in Examplar_set: # memory
                if int(_d['bigCate']) in cates:
                    cate_data.append(_d)
                    self.cate_set.add(_d['bigCate'])

        self.data = cate_data


        if self.verbose:
            print(task, "   # all sentences:", len(self.data))
            if mode == 'train':
                print("    cate set:", self.cate_set, ', miss cate:', set(cates).difference(self.cate_set))

        self.frame_feats = {}
        self.mot_feats = {}
        vid_feat_file = os.path.join('/data/zhangxi/nextqa', 'vid_feat/app_mot_{}.h5'.format(mode))
        with h5py.File(vid_feat_file, 'r') as fp:
            vids = fp['ids']
            feats = fp['feat']
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                self.frame_feats[str(vid)] = feat[:, :2048]  # (16, 2048)
                self.mot_feats[str(vid)] = feat[:, 2048:]  # (16, 2048)

        if mode == 'train':
            print("==== Only use motion features! ====")



    def __len__(self):
        return len(self.data)

    def get_video_feature(self, video_name):
        app_feat = self.frame_feats[video_name]
        video_feature = app_feat # (16, 2048)
        mot_feat = self.mot_feats[video_name]
        # video_feature = np.concatenate((video_feature, mot_feat), axis=1) #(16, 4096)

        return torch.from_numpy(mot_feat).type(torch.float32)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        # datum = self.data.loc[idx]

        out_dict['ques_label'] = All_Q.index(datum['type'])
        out_dict['img_cate'] = int(datum['bigCate']) - 1  # -1 !!!!



        ###### Image ######
        if self.args.use_vision:
            video_name = datum['video']
            video_name = str(video_name)
            video_feature = self.get_video_feature(video_name)

            out_dict['vis_feats'] = video_feature

            boxes = np.zeros([16,4], dtype=float)
            for i in range(16):
                # boxes[i][0] = 0
                # boxes[i][0] = 0
                boxes[i][2] = 1
                boxes[i][3] = 1
                    # = np.array([0,0,1,1], dtype=float)
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

            out_dict['boxes'] = boxes

        ###### Text #####

        sent = datum['question']
        qtype =  datum['type']
        sent, qtype = str(sent), str(qtype)

        input_ids = self.tokenizer.encode(f'vqa: {sent}', max_length=23, truncation=True) # max_length

        question_id = str(datum['video']) +'_'+ str(datum['qid'])
        out_dict['question_id'] = question_id


        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)

        answer = datum['answer']
        out_dict['answer'] = answer

        target_ids = self.tokenizer.encode(answer, max_length=6, truncation=True)

        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)
        out_dict['score'] = 1.0

        return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        args = batch[0]['args']

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if args.use_vision:
            V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        if 'target' in batch[0]:
            # targets = []
            targets = torch.zeros(B, len(batch[0]['target']), dtype=torch.float)
        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        sentences = []
        question_ids = []
        answers = []
        all_answers = []
        img_ids = []
        img_paths = []
        labels = []
        scores = []
        is_topk_optimal = []
        cate_labels = []
        ques_labels = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            if args.use_vision:
                boxes[i] += entry['boxes']
                vis_feats[i] += entry['vis_feats']
                # img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'target' in entry:
                targets[i] += entry['target']
                # targets.append(entry['target'])

            sentences.append(entry['sent'])
            question_ids.append(entry['question_id'])
            if 'answer' in entry:
                answers.append(entry['answer'])
            if 'all_answers' in entry:
                all_answers.append(entry['all_answers'])
            if 'score' in entry:
                scores.append(entry['score'])

            if 'label' in entry:
                labels.append(entry['label'])

            if 'is_topk_optimal' in entry:
                is_topk_optimal.append(entry['is_topk_optimal'])

            if 'img_cate' in entry: #-------------
                cate_labels.append(entry['img_cate'])
            if 'ques_label' in entry:
                ques_labels.append(entry['ques_label'])

        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        if 'target' in batch[0]:
            # targets = torch.stack(targets, dim=0)
            batch_entry['targets'] = targets

        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            # batch_entry['img_id'] = img_ids
            # batch_entry['img_paths'] = img_paths

        batch_entry['sent'] = sentences
        batch_entry['question_ids'] = question_ids
        batch_entry['answers'] = answers
        batch_entry['all_answers'] = all_answers
        batch_entry['scores'] = torch.FloatTensor(scores)
        batch_entry['labels'] = labels

        batch_entry['args'] = args
        batch_entry['task'] = 'vqa'

        cate_labels_ = torch.LongTensor(cate_labels).unsqueeze(1)  # [bs, 1]
        batch_entry['cate_labels'] = torch.zeros(cate_labels_.shape[0], 80).scatter_(1, cate_labels_, 1)  # [bs, 80]

        ques_labels_ = torch.LongTensor(ques_labels).unsqueeze(1)
        batch_entry['ques_labels'] = torch.zeros(cate_labels_.shape[0], len(All_Q)).scatter_(1, ques_labels_, 1)

        return batch_entry

class VQAFineTuneDataset_memory(Dataset):
    def __init__(self, split, Examplar_set, raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train', task='DO', cates=[0,1,2]):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        # self.sources = split.split(',')
        self.sources = mode

        # if self.verbose:
        #     print('Data sources: ', self.sources)

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)

        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            if args.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)


        # all_data = Examplar_set
        print('load from memory buffer')

        cate_data = []
        self.cate_set = set()

        # if len(Examplar_set) > 0:
        for _d in Examplar_set: # memory
            if int(_d['bigCate']) in cates:
                cate_data.append(_d)
                self.cate_set.add(_d['bigCate'])

        self.data = cate_data


        if self.verbose:
            print(task, "   # all sentences:", len(self.data))
            if mode == 'train':
                print("    cate set:", self.cate_set, ', miss cate:', set(cates).difference(self.cate_set))

        self.frame_feats = {}
        self.mot_feats = {}
        vid_feat_file = os.path.join('/data/zhangxi/nextqa', 'vid_feat/app_mot_{}.h5'.format(mode))
        with h5py.File(vid_feat_file, 'r') as fp:
            vids = fp['ids']
            feats = fp['feat']
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                self.frame_feats[str(vid)] = feat[:, :2048]  # (16, 2048)
                self.mot_feats[str(vid)] = feat[:, 2048:]  # (16, 2048)

        if mode == 'train':
            print("==== Only use motion features! ====")


    def __len__(self):
        return len(self.data)

    def get_video_feature(self, video_name):
        app_feat = self.frame_feats[video_name]
        video_feature = app_feat # (16, 2048)
        mot_feat = self.mot_feats[video_name]

        return torch.from_numpy(mot_feat).type(torch.float32)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        # datum = self.data.loc[idx]

        out_dict['ques_label'] = All_Q.index(datum['type'])
        out_dict['img_cate'] = int(datum['bigCate']) - 1  # -1 !!!!



        ###### Image ######
        if self.args.use_vision:
            video_name = datum['video']
            video_name = str(video_name)
            video_feature = self.get_video_feature(video_name) # 原本维度是36, 2048

            out_dict['vis_feats'] = video_feature


            boxes = np.zeros([16,4], dtype=float)
            for i in range(16):
                # boxes[i][0] = 0
                # boxes[i][0] = 0
                boxes[i][2] = 1
                boxes[i][3] = 1
                    # = np.array([0,0,1,1], dtype=float)
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

            out_dict['boxes'] = boxes

        ###### Text #####
        # caption = datum['caption']
        # if 'sent' in datum:
        #     sent = datum['sent']
        # elif 'question' in datum:
        #     sent = datum['question']
        sent = datum['question']
        qtype =  datum['type']
        sent, qtype = str(sent), str(qtype)

        input_ids = self.tokenizer.encode(f'vqa: {sent}', max_length=23, truncation=True) # max_length

        question_id = str(datum['video']) +'_'+ str(datum['qid'])
        out_dict['question_id'] = question_id


        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        # out_dict['target_ids'] = torch.LongTensor(target_ids)
        # out_dict['target_length'] = len(target_ids)


        answer = datum['answer']
        out_dict['answer'] = answer
        # out_dict['score'] = score
        # out_dict['all_answers'] = [answer]

        target_ids = self.tokenizer.encode(answer, max_length=6, truncation=True)

        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)
        out_dict['score'] = 1.0
        # ================================

        return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        args = batch[0]['args']

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if args.use_vision:
            V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        if 'target' in batch[0]:
            # targets = []
            targets = torch.zeros(B, len(batch[0]['target']), dtype=torch.float)
        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        sentences = []
        question_ids = []
        answers = []
        all_answers = []
        img_ids = []
        img_paths = []
        labels = []
        scores = []
        is_topk_optimal = []
        cate_labels = []
        ques_labels = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            if args.use_vision:
                boxes[i] += entry['boxes']
                vis_feats[i] += entry['vis_feats']
                # img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'target' in entry:
                targets[i] += entry['target']
                # targets.append(entry['target'])

            sentences.append(entry['sent'])
            question_ids.append(entry['question_id'])
            if 'answer' in entry:
                answers.append(entry['answer'])
            if 'all_answers' in entry:
                all_answers.append(entry['all_answers'])
            if 'score' in entry:
                scores.append(entry['score'])

            if 'label' in entry:
                labels.append(entry['label'])

            if 'is_topk_optimal' in entry:
                is_topk_optimal.append(entry['is_topk_optimal'])

            if 'img_cate' in entry: #-------------
                cate_labels.append(entry['img_cate'])
            if 'ques_label' in entry:
                ques_labels.append(entry['ques_label'])

        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        if 'target' in batch[0]:
            # targets = torch.stack(targets, dim=0)
            batch_entry['targets'] = targets

        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            # batch_entry['img_id'] = img_ids
            # batch_entry['img_paths'] = img_paths

        batch_entry['sent'] = sentences
        batch_entry['question_ids'] = question_ids
        batch_entry['answers'] = answers
        batch_entry['all_answers'] = all_answers
        batch_entry['scores'] = torch.FloatTensor(scores)
        batch_entry['labels'] = labels

        batch_entry['args'] = args
        batch_entry['task'] = 'vqa'

        cate_labels_ = torch.LongTensor(cate_labels).unsqueeze(1)  # [bs, 1]
        batch_entry['cate_labels'] = torch.zeros(cate_labels_.shape[0], 80).scatter_(1, cate_labels_, 1)  # [bs, 80]

        ques_labels_ = torch.LongTensor(ques_labels).unsqueeze(1)
        batch_entry['ques_labels'] = torch.zeros(cate_labels_.shape[0], len(All_Q)).scatter_(1, ques_labels_, 1)

        return batch_entry



def get_loader(args, Examplar_set, _dset, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1, task='DO'):

    verbose = (gpu == 0)

    # _dset = VQADataset(split, verbose)

    cate_loader = {}

    for idx, CateGroup in enumerate(Category_splits):

        dataset = VQAFineTuneDataset(
            split,
            Examplar_set,
            task=task,
            raw_dataset=None,
            rank=gpu,
            topk=topk,
            verbose=verbose,
            args=args,
            mode=mode,
            cates=Category_splits[CateGroup],)

        if distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = None

        if mode == 'train':
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=(sampler is None),
                num_workers=workers, pin_memory=True, sampler=sampler,
                collate_fn=dataset.collate_fn)
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=workers, pin_memory=True,
                sampler=sampler,
                shuffle=None if (sampler is not None) else False,
                collate_fn=dataset.collate_fn,
                drop_last=False)

        if verbose:
            loader.evaluator = VQAEvaluator(_dset)

        loader.task = 'vqa'
        cate_loader[CateGroup] = loader

    return cate_loader


def get_loader_vis(args, Examplar_set, _dset, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1, task='DO'):

    verbose = (gpu == 0)

    # _dset = VQADataset(split, verbose)

    cate_loader = {}

    for idx, CateGroup in enumerate(range(5)):

        dataset = VQAFineTuneDataset(
            split,
            Examplar_set,
            task=task,
            raw_dataset=None,
            rank=gpu,
            topk=topk,
            verbose=verbose,
            args=args,
            mode=mode,
            cates=[CateGroup],)

        if distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = None

        if mode == 'train':
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=(sampler is None),
                num_workers=workers, pin_memory=True, sampler=sampler,
                collate_fn=dataset.collate_fn)
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=workers, pin_memory=True,
                sampler=sampler,
                shuffle=None if (sampler is not None) else False,
                collate_fn=dataset.collate_fn,
                drop_last=False)

        if verbose:
            loader.evaluator = VQAEvaluator(_dset)

        loader.task = 'vqa'
        cate_loader[CateGroup] = loader

    return cate_loader



def get_loader_memory(args, Examplar_set, _dset, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1, task='DO'):

    verbose = (gpu == 0)

    # _dset = VQADataset(split, verbose)

    cate_loader = {}

    for idx, CateGroup in enumerate(Category_splits):

        dataset = VQAFineTuneDataset_memory(
            split,
            Examplar_set,
            task=task,
            raw_dataset=None,
            rank=gpu,
            topk=topk,
            verbose=verbose,
            args=args,
            mode=mode,
            cates=Category_splits[CateGroup],)

        if distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = None

        if mode == 'train':
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=(sampler is None),
                num_workers=workers, pin_memory=True, sampler=sampler,
                collate_fn=dataset.collate_fn)
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=workers, pin_memory=True,
                sampler=sampler,
                shuffle=None if (sampler is not None) else False,
                collate_fn=dataset.collate_fn,
                drop_last=False)

        if verbose:
            loader.evaluator = VQAEvaluator(_dset)

        loader.task = 'vqa'
        cate_loader[CateGroup] = loader

    return cate_loader



def get_loader_test(args, Examplar_set, _dset, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1, task='DO'):

    verbose = (gpu == 0)

    # _dset = VQADataset(split, verbose)

    dataset = VQAFineTuneDataset(
        split,
        Examplar_set,
        task=task,
        raw_dataset=None,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode,
        cates=[i for i in range(1, 81)],)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = VQAEvaluator(_dset)

    loader.task = 'vqa'

    return loader


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """

    def __init__(self, splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')

        sample_list_file = os.path.join('/data/zhangxi/nextqa', 'train.csv')
        train2014_data = load_file(sample_list_file)
        sample_list_file = os.path.join('/data/zhangxi/nextqa', 'val.csv')
        val2014_data = load_file(sample_list_file)
        train2014_id2datum = {}
        for i in range(len(train2014_data)):
            datum = train2014_data.loc[i]
            qid = str(datum['video']) +'_'+ str(datum['qid'])
            train2014_id2datum[qid] = datum
        val2014_id2datum = {}
        for i in range(len(val2014_data)):
            datum = val2014_data.loc[i]
            qid = str(datum['video']) +'_'+ str(datum['qid'])
            val2014_id2datum[qid] = datum

        self.id2datum_gt = {**train2014_id2datum, **val2014_id2datum}

        # Loading datasets

        self.data = []
        for split in self.splits:
            sample_list = load_file(os.path.join('/data/zhangxi/nextqa', split+'.csv'))
            for i in range(len(sample_list)):
                self.data.append(sample_list.loc[i])


        # Convert list to dict (for evaluation)
        self.id2datum = {
            str(datum['video']) +'_'+ str(datum['qid']): datum
            for datum in self.data
        }


    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class VQAEvaluator:
    def __init__(self, dataset: VQADataset = None):
        self.dataset = dataset

        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
							 "youll": "you'll", "youre": "you're", "youve": "you've"}

        self.manualMap    = { 'none': '0',
							  'zero': '0',
							  'one': '1',
							  'two': '2',
							  'three': '3',
							  'four': '4',
							  'five': '5',
							  'six': '6',
							  'seven': '7',
							  'eight': '8',
							  'nine': '9',
							  'ten': '10'
							}

        self.articles     = ['a',
							 'an',
							 'the'
							]

        self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip   = re.compile("(\d)(\,)(\d)")
        self.punct        = [';', r"/", '[', ']', '"', '{', '}',
							 '(', ')', '=', '+', '\\', '_', '-',
							 '>', '<', '@', '`', ',', '?', '!']

        self.n = 2

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }
        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)

    def remove_stop(self, sentence):

        # words = lemmatize_sentence(sentence)
        words = nltk.word_tokenize(sentence)
        words = [w for w in words if not w in stopwords]
        return ' '.join(words)


    def evaluate_raw(self, quesid2ans: dict, is_topk_optimal=None):
        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        gts = self.dataset.id2datum_gt

        self.accuracy     = {}
        self.evalQA       = {}
        self.evalQuesType = {}
        self.evalAnsType  = {}
        self.add_ref = load_file('/data/zhangxi/nextqa/add_reference_answer_test.json')

        # accQA = []
        accQuesType = {}
        accAnsType = {}

        # print("Computing accuracy")
        score = 0

        for quesId, resAns in tqdm(quesid2ans.items(), total=len(quesid2ans), ncols=80):

            # quesId = int(quesId)

            datum = self.dataset.id2datum[quesId]

            if is_topk_optimal is None:
                pass
            elif 'is_topk_optimal' in datum:
                if datum['is_topk_optimal'] != is_topk_optimal:
                    continue

            resAns      = resAns.replace('\n', ' ')
            resAns      = resAns.replace('\t', ' ')
            resAns      = resAns.strip()
            resAns      = self.processPunctuation(resAns)
            resAns      = self.processDigitArticle(resAns)

            gtAnswers = datum['answer']
            quesType = datum['type']

            resAns = self.remove_stop(resAns) # for nextqa
            gtAnswers = self.remove_stop(gtAnswers)
            if datum['video'] in self.add_ref:
                gt_ans_add = self.add_ref[datum['video']][datum['qid']]
                gt_ans_add = self.remove_stop(gt_ans_add)
                if quesType in ['CC', 'CB']:
                    if resAns == gtAnswers or resAns == gt_ans_add:
                        cur_s = 1
                    else:
                        cur_s = 0
                else:
                    cur_s = max(get_wups(resAns, gtAnswers, 0), get_wups(resAns, gt_ans_add, 0))
            else:
                if quesType in ['CC', 'CB']:
                    if resAns == gtAnswers:
                        cur_s = 1
                    else:
                        cur_s = 0
                else:
                    cur_s = get_wups(resAns, gtAnswers, 0)

            score += cur_s

            # if cur_s < 0.5:
            #     print('@wrong,', cur_s, ',', quesId, ',', quesType, ',', self.dataset.id2datum[quesId]['question'], ',',
            #     self.dataset.id2datum[quesId]['video'], ', Predict:', resAns, ", GT:", gtAnswers)

        # if len(accQA) == 0:
        #     return {
        #         'overall': 0,
        #         # 'perQuestionType': {},
        #         # 'perAnswerType': {}
        #     }
        # else:
            # self.setAccuracy(accQA, accQuesType, accAnsType)
        self.accuracy['overall'] = (float(score)/len(quesid2ans)) * 100

        return self.accuracy

    def normalize_answer(self, resAns):
        resAns      = resAns.replace('\n', ' ')
        resAns      = resAns.replace('\t', ' ')
        resAns      = resAns.strip()
        resAns      = self.processPunctuation(resAns)
        resAns      = self.processDigitArticle(resAns)
        resAns = resAns.replace(',', '')
        return resAns

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                        outText,
                                        re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100*acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100*acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100*acc, self.n)

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy['overall']   = round(100*float(sum(accQA))/len(accQA), self.n)
        self.accuracy['perQuestionType'] = {quesType: round(100*float(sum(accQuesType[quesType]))/len(accQuesType[quesType]), self.n) for quesType in accQuesType}
        self.accuracy['perAnswerType']   = {ansType:  round(100*float(sum(accAnsType[ansType]))/len(accAnsType[ansType]), self.n) for ansType in accAnsType}

