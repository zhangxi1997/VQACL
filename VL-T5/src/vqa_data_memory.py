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

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

import sys
sys.path.append("..")
from Question_type import Category_splits, ImgId_cate_map, QuesId_task_map, All_task_list


project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = coco_dir.joinpath('images/')
coco_feature_dir = coco_dir.joinpath('features')
vqa_dir = dataset_dir.joinpath('vqa')


class VQAFineTuneDataset(Dataset):
    def __init__(self, coco_Ours, Examplar_set, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train', task='q_what', cates=[0,1,2]):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.sources = split.split(',')
        # if self.verbose:
        #     print('Data sources: ', self.sources,'_' + task)

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

        self.answer_normalizer = VQAEvaluator()

        self.img_ids_to_source = {}
        data_info_dicts_cate = []
        self.cate_set = set()
        for source in self.sources:
            data_info_path = dataset_dir.joinpath(f'vqa/Partition_Q/{source}_'+f'{task}.json')

            with open(data_info_path) as f:
                _data_info_dicts = json.load(f)
                _data_info_dicts.extend(Examplar_set)
                for _d in _data_info_dicts:
                    img_id = _d['img_id']
                    try:
                        if ImgId_cate_map[img_id] in cates:
                            self.cate_set.add(ImgId_cate_map[img_id])
                            data_info_dicts_cate.append(_d)
                            if 'vg_qa_full' == source:
                                self.img_ids_to_source[_d['img_id']] = 'vg'
                            elif 'train2014' in _d['img_id']:
                                self.img_ids_to_source[_d['img_id']] = 'train2014'
                            elif 'val2014' in _d['img_id']:
                                self.img_ids_to_source[_d['img_id']] = 'val2014'
                            else:
                                self.img_ids_to_source[_d['img_id']] = source
                                _d['source'] = source
                    except:
                        continue

        data = data_info_dicts_cate

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data), 'with Examplers')
            if self.sources[0] == 'karpathy_train':
                print("    cate set:", self.cate_set, ', miss cate:', set(cates).difference(self.cate_set))

        self.n_boxes = args.n_boxes
        self.source_to_h5 = {
            'train': coco_feature_dir.joinpath(f'train2014_obj36.h5'),
            'minival': coco_feature_dir.joinpath(f'val2014_obj36.h5'),
            'nominival': coco_feature_dir.joinpath(f'val2014_obj36.h5'),
            'test': coco_feature_dir.joinpath(f'test2015_obj36.h5'),

            'vg': dataset_dir.joinpath('VG/features').joinpath('vg_gqa_obj36.h5'),

            'train2014': coco_feature_dir.joinpath(f'train2014_obj36.h5'),
            'val2014': coco_feature_dir.joinpath(f'val2014_obj36.h5'),
        }



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['img_id']
            out_dict['img_id'] = img_id

            out_dict['img_cate'] = ImgId_cate_map[img_id]

            source = self.img_ids_to_source[img_id] # source: val2014

            f = self.source_to_h5[source]

            if isinstance(f, Path):
                # path = self.data_source_to_h5_path[source]
                f = h5py.File(f, 'r')
                # self.split_to_h5_features[split_i] = f
                self.source_to_h5[source] = f

            feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
            try:
                f[f'{img_id}/features'].read_direct(feats)
            except KeyError:
                print('img_id', img_id)
                print(datum)
                exit()

            feats = torch.from_numpy(feats)
            out_dict['vis_feats'] = feats

            # Normalize the boxes (to 0 ~ 1)
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

            out_dict['boxes'] = boxes

        ###### Text #####
        # caption = datum['caption']
        if 'sent' in datum:
            sent = datum['sent']
        elif 'question' in datum:
            sent = datum['question']

        input_ids = self.tokenizer.encode(f'vqa: {sent}', max_length=20, truncation=True)

        question_id = datum['question_id']
        out_dict['question_id'] = question_id

        out_dict['ques_label'] = QuesId_task_map[str(question_id)] # ------


        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        # out_dict['target_ids'] = torch.LongTensor(target_ids)
        # out_dict['target_length'] = len(target_ids)

        if 'is_topk_optimal' in datum:
            out_dict['is_topk_optimal'] = datum['is_topk_optimal']

        if 'label' in datum:
            label = datum['label']
            out_dict['label'] = label

            # 3129 topk answers
            if self.args.classifier:
                target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in label.items():
                    target[self.raw_dataset.ans2label[ans]] = score
                out_dict['target'] = target

            elif self.args.raw_label:

                # 10 raw answers
                # ex) 'answers': [{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 2},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 3},
                #     {'answer': 'netting', 'answer_confidence': 'yes', 'answer_id': 4},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 5},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 6},
                #     {'answer': 'mesh', 'answer_confidence': 'maybe', 'answer_id': 7},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 8},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 9},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 10}],

                answers = datum['answers']
                answer = random.choice(answers)['answer']

                if self.args.answer_normalize:
                    answer = self.answer_normalizer.normalize_answer(answer)

                score = int(len(answers) > 0)

                out_dict['answer'] = answer
                out_dict['score'] = score
                out_dict['all_answers'] = [a['answer'] for a in answers]

                target_ids = self.tokenizer.encode(answer, max_length=10, truncation=True)

                out_dict['target_ids'] = torch.LongTensor(target_ids)
                out_dict['target_length'] = len(target_ids)

            else:
                # https://github.com/airsplay/lxmert/blob/master/src/pretrain/lxmert_pretrain.py#L191

                answers = []
                scores = []
                for a, s in label.items():
                    answers.append(a)
                    scores.append(s)

                score_sum = sum(scores)

                if score_sum == 0:
                    answer = ''
                    score = 0.
                else:
                    prob = [score / score_sum for score in scores]
                    choice = np.random.multinomial(1, prob).argmax()
                    answer = answers[choice]
                    score = scores[choice]
                    assert len(answer) > 0, (sent, label, choice, answer)

                out_dict['answer'] = answer
                out_dict['score'] = score
                out_dict['all_answers'] = answers


                target_ids = self.tokenizer.encode(answer, max_length=10, truncation=True)

                out_dict['target_ids'] = torch.LongTensor(target_ids)
                out_dict['target_length'] = len(target_ids)

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

        cate_labels_ = torch.LongTensor(cate_labels).unsqueeze(1) #[bs, 1]
        batch_entry['cate_labels'] = torch.zeros(cate_labels_.shape[0], 80).scatter_(1, cate_labels_, 1 ) # [bs, 80]

        ques_labels_ = torch.LongTensor(ques_labels).unsqueeze(1)
        batch_entry['ques_labels'] = torch.zeros(cate_labels_.shape[0], len(All_task_list)).scatter_(1, ques_labels_, 1 ) # [bs, 10]

        return batch_entry

class VQAFineTuneDataset_memory(Dataset):
    def __init__(self, coco_Ours, Examplar_set, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train', cates=[0,1,2]):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.sources = split.split(',')
        # if self.verbose:
        #     print('Data sources: ', self.sources,'_' + task)

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

        self.answer_normalizer = VQAEvaluator()

        self.img_ids_to_source = {}
        data_info_dicts_cate = []

        _data_info_dicts = Examplar_set # ---- from the memory

        for _d in _data_info_dicts:
            img_id = _d['img_id']
            try:
                if ImgId_cate_map[img_id] in cates:
                    data_info_dicts_cate.append(_d)
                    if 'vg_qa_full' == self.sources[0]:
                        self.img_ids_to_source[_d['img_id']] = 'vg'
                    elif 'train2014' in _d['img_id']:
                        self.img_ids_to_source[_d['img_id']] = 'train2014'
                    elif 'val2014' in _d['img_id']:
                        self.img_ids_to_source[_d['img_id']] = 'val2014'
                    else:
                        self.img_ids_to_source[_d['img_id']] = self.sources[0]
                        _d['source'] = self.sources[0]
            except:
                continue


        data = data_info_dicts_cate

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data), 'with Examplers')

        self.n_boxes = args.n_boxes
        self.source_to_h5 = {
            'train': coco_feature_dir.joinpath(f'train2014_obj36.h5'),
            'minival': coco_feature_dir.joinpath(f'val2014_obj36.h5'),
            'nominival': coco_feature_dir.joinpath(f'val2014_obj36.h5'),
            'test': coco_feature_dir.joinpath(f'test2015_obj36.h5'),

            'vg': dataset_dir.joinpath('VG/features').joinpath('vg_gqa_obj36.h5'),

            'train2014': coco_feature_dir.joinpath(f'train2014_obj36.h5'),
            'val2014': coco_feature_dir.joinpath(f'val2014_obj36.h5'),
        }



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['img_id'] # img_id: COCO_val2014_.....
            out_dict['img_id'] = img_id

            out_dict['img_cate'] = ImgId_cate_map[img_id]

            source = self.img_ids_to_source[img_id]

            f = self.source_to_h5[source]

            if isinstance(f, Path):
                # path = self.data_source_to_h5_path[source]
                f = h5py.File(f, 'r')
                # self.split_to_h5_features[split_i] = f
                self.source_to_h5[source] = f

            feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
            try:
                f[f'{img_id}/features'].read_direct(feats)
            except KeyError:
                print('img_id', img_id)
                print(datum)
                exit()

            feats = torch.from_numpy(feats)
            out_dict['vis_feats'] = feats

            # Normalize the boxes (to 0 ~ 1)
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

            out_dict['boxes'] = boxes

        ###### Text #####
        # caption = datum['caption']
        if 'sent' in datum:
            sent = datum['sent']
        elif 'question' in datum:
            sent = datum['question']

        input_ids = self.tokenizer.encode(f'vqa: {sent}', max_length=20, truncation=True)

        question_id = datum['question_id']
        out_dict['question_id'] = question_id

        out_dict['ques_label'] = QuesId_task_map[str(question_id)]


        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        # out_dict['target_ids'] = torch.LongTensor(target_ids)
        # out_dict['target_length'] = len(target_ids)

        if 'is_topk_optimal' in datum:
            out_dict['is_topk_optimal'] = datum['is_topk_optimal']

        if 'label' in datum:
            label = datum['label']
            out_dict['label'] = label

            # 3129 topk answers
            if self.args.classifier:
                target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in label.items():
                    target[self.raw_dataset.ans2label[ans]] = score
                out_dict['target'] = target

            elif self.args.raw_label:

                # 10 raw answers
                # ex) 'answers': [{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 2},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 3},
                #     {'answer': 'netting', 'answer_confidence': 'yes', 'answer_id': 4},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 5},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 6},
                #     {'answer': 'mesh', 'answer_confidence': 'maybe', 'answer_id': 7},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 8},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 9},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 10}],

                answers = datum['answers']
                answer = random.choice(answers)['answer']

                if self.args.answer_normalize:
                    answer = self.answer_normalizer.normalize_answer(answer)

                score = int(len(answers) > 0)

                out_dict['answer'] = answer
                out_dict['score'] = score
                out_dict['all_answers'] = [a['answer'] for a in answers]

                target_ids = self.tokenizer.encode(answer, max_length=10, truncation=True)

                out_dict['target_ids'] = torch.LongTensor(target_ids)
                out_dict['target_length'] = len(target_ids)

            else:
                # https://github.com/airsplay/lxmert/blob/master/src/pretrain/lxmert_pretrain.py#L191

                answers = []
                scores = []
                for a, s in label.items():
                    answers.append(a)
                    scores.append(s)

                score_sum = sum(scores)

                if score_sum == 0:
                    answer = ''
                    score = 0.
                else:
                    prob = [score / score_sum for score in scores]
                    choice = np.random.multinomial(1, prob).argmax()
                    answer = answers[choice]
                    score = scores[choice]
                    assert len(answer) > 0, (sent, label, choice, answer)

                out_dict['answer'] = answer
                out_dict['score'] = score
                out_dict['all_answers'] = answers


                target_ids = self.tokenizer.encode(answer, max_length=10, truncation=True)

                out_dict['target_ids'] = torch.LongTensor(target_ids)
                out_dict['target_length'] = len(target_ids)

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

            if 'img_cate' in entry:
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

        cate_labels_ = torch.LongTensor(cate_labels).unsqueeze(1)
        batch_entry['cate_labels'] = torch.zeros(cate_labels_.shape[0], 80).scatter_(1, cate_labels_, 1 ) # [bs, 80]

        ques_labels_ = torch.LongTensor(ques_labels).unsqueeze(1)
        batch_entry['ques_labels'] = torch.zeros(cate_labels_.shape[0], len(All_task_list)).scatter_(1, ques_labels_, 1 ) # [bs, 10]

        return batch_entry



def get_loader_memory(args, coco_Ours, Examplar_set, _dset, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = (gpu == 0)

    # _dset = VQADataset(split, verbose, task) # 这里不用改动？

    cate_loader = {}

    for idx, CateGroup in enumerate(Category_splits):
        dataset = VQAFineTuneDataset_memory(
            coco_Ours,
            Examplar_set,
            split,
            raw_dataset=_dset,
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




def get_loader_test(args, coco_Ours, Examplar_set, _dset, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1, task='what'):

    verbose = (gpu == 0)
    # cate_loader = {}

    dataset = VQAFineTuneDataset(
        coco_Ours,
        Examplar_set,
        split,
        raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode,
        task=task,
        cates=[i for i in range(80)],) # all categories

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
    # cate_loader[CateGroup] = loader
    return loader



def get_loader(args, coco_Ours, Examplar_set, _dset, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1, task='what'):

    verbose = (gpu == 0)

    cate_loader = {}
    total_num = 0

    for idx, CateGroup in enumerate(Category_splits):
        print(CateGroup, end=',')
        dataset = VQAFineTuneDataset(
            coco_Ours,
            Examplar_set,
            split,
            raw_dataset=_dset,
            rank=gpu,
            topk=topk,
            verbose=verbose,
            args=args,
            mode=mode,
            task=task,
            cates=Category_splits[CateGroup],)

        total_num += len(dataset)

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

    return cate_loader, total_num



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

        with open(dataset_dir.joinpath(f'vqa/v2_mscoco_train2014_annotations.json')) as f:
            train2014_data = json.load(f)

        with open(dataset_dir.joinpath(f'vqa/v2_mscoco_val2014_annotations.json')) as f:
            val2014_data = json.load(f)

        train2014_id2datum = {}
        for datum in train2014_data['annotations']:
            qid = datum['question_id']
            train2014_id2datum[qid] = datum
        val2014_id2datum = {}
        for datum in val2014_data['annotations']:
            qid = datum['question_id']
            val2014_id2datum[qid] = datum
        self.id2datum_gt = {**train2014_id2datum, **val2014_id2datum}

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(
                json.load(open(vqa_dir.joinpath("%s.json" % (split)))))

        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Topk Answers
        self.ans2label = json.load(
            open(vqa_dir.joinpath("trainval_ans2label.json")))
        self.label2ans = json.load(
            open(vqa_dir.joinpath("trainval_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)

        if verbose:
            print('# All Answers:', len(self.ans2label))

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

    def evaluate_raw(self, quesid2ans: dict, is_topk_optimal=None):
        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        gts = self.dataset.id2datum_gt

        self.accuracy     = {}
        self.evalQA       = {}
        self.evalQuesType = {}
        self.evalAnsType  = {}

        accQA = []
        accQuesType = {}
        accAnsType = {}

        # print("Computing accuracy")

        for quesId, resAns in tqdm(quesid2ans.items(), total=len(quesid2ans), ncols=80):

            quesId = int(quesId)

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

            gtAcc  = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]
            if len(set(gtAnswers)) > 1:
                for ansDic in gts[quesId]['answers']:
                    ansDic['answer'] = self.processPunctuation(ansDic['answer'])
            for gtAnsDatum in gts[quesId]['answers']:
                otherGTAns = [item for item in gts[quesId]['answers'] if item!=gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item['answer']==resAns]
                acc = min(1, float(len(matchingAns))/3)
                gtAcc.append(acc)

            quesType    = gts[quesId]['question_type']
            ansType     = gts[quesId]['answer_type']
            avgGTAcc = float(sum(gtAcc))/len(gtAcc)
            accQA.append(avgGTAcc)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)

            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)

            # if avgGTAcc == 0.0:
            #     print(quesId,',', quesType,',', self.dataset.id2datum[quesId]['sent'],',', self.dataset.id2datum[quesId]['img_id'],', Predict:',resAns, ", GT:", gtAnswers,"wrong-----")

            # if avgGTAcc == 1.0:
            # print('@,',avgGTAcc,',', quesId,',', quesType,',', self.dataset.id2datum[quesId]['sent'],',', self.dataset.id2datum[quesId]['img_id'],', Predict:',resAns, ", GT:", gtAnswers)


        if len(accQA) == 0:
            return {
                'overall': 0,
                'perQuestionType': {},
                'perAnswerType': {}
            }
        else:
            self.setAccuracy(accQA, accQuesType, accAnsType)

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
        self.accuracy['overall']         = round(100*float(sum(accQA))/len(accQA), self.n)
        self.accuracy['perQuestionType'] = {quesType: round(100*float(sum(accQuesType[quesType]))/len(accQuesType[quesType]), self.n) for quesType in accQuesType}
        self.accuracy['perAnswerType']   = {ansType:  round(100*float(sum(accAnsType[ansType]))/len(accAnsType[ansType]), self.n) for ansType in accAnsType}

