
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint

from param import parse_args

from vqa_data_memory import get_loader_test, VQADataset, get_loader
from utils import load_state_dict, LossMeter, set_global_logging_level
import dist_utils
from Question_type import Category_splits, random_dic
import json
import random



import wandb
# try:
#     import wandb
#     from wandb import init, finish
# except ImportError:
#     wandb=None
# wandb = None

# set_global_logging_level(logging.ERROR, ["transformers"])

proj_dir = Path(__file__).resolve().parent.parent


_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase

#  10个 task
from Question_type import All_task, Comp_task




def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class Trainer(TrainerBase):
    def __init__(self, args, coco_Ours, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.result_matrix = {}
        self.task_list = []
        for task in coco_Ours:
            self.result_matrix[task] = {}
            self.task_list.append(task)

        self.test_loader_dict_all = {}
        self.test_loader_dict = {}
        self.test_dset = VQADataset(args.test, True)

        super().__init__(
            args,
            train=train)

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        from vqa_model import VLT5VQA

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5VQA


        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        # --------------------- random seed --------------------#
        if self.args.from_scratch:
            if args.ifseed:
                self.init_weights(seed=args.seed, ifseed=True)
            else:
                self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')
        self.iftrain = train
        self.coco_Ours = coco_Ours

        self.composition_test_cate = args.comp_cate



    def Test(self, load=False):

        for task_idx, task in enumerate(self.task_list):
            print('======================== Now is task "', task, '" ========================')

            test_loader = get_loader_test(
                args,
                self.coco_Ours,
                [],
                self.test_dset,
                split=args.test, mode='val', batch_size=args.valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=4,
                topk=args.valid_topk,
                task=task,
            )
            self.test_loader_dict_all[task] = test_loader

            test_loader, _ = get_loader(
                args,
                self.coco_Ours,
                [],
                self.test_dset,
                split=args.test, mode='val', batch_size=args.valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=4,
                topk=args.valid_topk,
                task=task,
            )
            self.test_loader_dict[task] = test_loader

        # ========= Testing =========
        self.test(self.task_list[-1], comp=True)



    def test(self, task, comp=False):
        # Test Set
        if not os.path.exists(self.args.output):
            os.mkdir(self.args.output)
        last_path = os.path.join(self.args.output, task + '_LAST')
        self.load(last_path)
        self.model.module.Q_prototype = torch.load(self.args.output+'/Q_prototype.pt')
        self.model.module.V_prototype = torch.load(self.args.output+'/V_prototype.pt')


        # =========== test for all previous tasks
        for test_task in self.coco_Ours:
            if test_task != self.task_list[0]:
                cate_comp = self.composition_test_cate
                self.test_loader = self.test_loader_dict[test_task][cate_comp]
                print(' ===== Test for the task "' + test_task + '"  ====== for composition ' + cate_comp)
            else:
                continue

            quesid2ans = self.predict(self.test_loader)

            if self.verbose:
                evaluator = self.test_loader.evaluator
                score_dict = evaluator.evaluate(quesid2ans)

                acc_dict_all = evaluator.evaluate_raw(quesid2ans)
                acc_dict_answerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=True)
                acc_dict_unanswerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=False)

                wandb_log_dict = {}
                wandb_log_dict['Test/overall'] = acc_dict_all['overall']
                wandb_log_dict['Test/topk_optimal'] = acc_dict_answerable['overall']
                wandb_log_dict['Test/topk_not_optimal'] = acc_dict_unanswerable['overall']

                for qtype, score in acc_dict_all['perQuestionType'].items():
                    wandb_log_dict[f'Test_Qtypes/{qtype}'] = score
                for atype, score in acc_dict_all['perAnswerType'].items():
                    if atype == 'yes/no':
                        atype = 'yes_no'
                    wandb_log_dict[f'Test_Atypes/{atype}'] = score

                print(test_task, wandb_log_dict)

            self.result_matrix[task][test_task] = acc_dict_all['overall']

            if self.args.distributed:
                dist.barrier()



    def predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}
            if self.verbose:
                pbar = tqdm(total=len(loader), ncols=120, desc="Prediction---")
            for i, batch in enumerate(loader):
                if self.args.distributed:
                    results = self.model.module.test_step(batch)
                else:
                    results = self.model.test_step(batch)

                pred_ans = results['pred_ans']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

                if self.verbose:
                    pbar.update(1)

            if self.verbose:
                pbar.close()

        if self.args.distributed:
            dist.barrier()

        qid2ans_list = dist_utils.all_gather(quesid2ans)
        if self.verbose:
            quesid2ans = {}
            for qid2ans in qid2ans_list:
                for k, v in qid2ans.items():
                    quesid2ans[k] = v

            if dump_path is not None:
                evaluator = loader.evaluator
                evaluator.dump_result(quesid2ans, dump_path)

        return quesid2ans

    def evaluate(self, loader, dump_path=None):
        quesid2ans = self.predict(loader, dump_path)

        if self.verbose:
            evaluator = loader.evaluator
            acc_dict = evaluator.evaluate_raw(quesid2ans)
            topk_score = evaluator.evaluate(quesid2ans)
            acc_dict['topk_score'] = topk_score

            return acc_dict

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')

    coco_Ours = All_task


    trainer = Trainer(args, coco_Ours, train=True)

    if args.checkpoint!='None':
        trainer.Test(load=True)
    else:
        trainer.Test(load=False)
    try:
        print('#------------------ Final Performance --------------------#')
        print(trainer.result_matrix['q_causal'])
        acc = 0
        for key in trainer.result_matrix['q_causal']:
            if key in Comp_task:
                acc += trainer.result_matrix['q_causal'][key]
        print('AP:', round(acc/len(Comp_task), 4))

    except:
        pass

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = 1
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)

        else:
            ckpt_str = 'scrach'
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
