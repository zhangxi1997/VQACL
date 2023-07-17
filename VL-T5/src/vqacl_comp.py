
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

from vqa_data_memory import get_loader, get_loader_test, VQADataset, get_loader_memory
from utils import load_state_dict, LossMeter, set_global_logging_level
import dist_utils
import json
import random

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
#  10 task
from Question_type import All_task, Comp_task, show_results_matrix, evaluate_metric, Category_splits, ImgId_cate_map, random_dic


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class Trainer(TrainerBase):
    def __init__(self, args, coco_Ours, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.result_matrix = {}
        self.result_matrix_comp = {}
        self.result_matrix_noncomp = {}
        self.task_list = []
        for task in coco_Ours:
            self.result_matrix[task] = {}
            self.result_matrix_comp[task] = {}
            self.result_matrix_noncomp[task] = {}
            self.task_list.append(task)

        self.train_loader_dict = {}
        self.val_loader_dict = {}
        self.test_loader_dict = {}
        self.test_loader_dict_all = {}

        self.train_dset = VQADataset(args.train, True)
        self.val_dset = VQADataset(args.valid, True)
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

        self.task_iftrain = {}
        for task in self.coco_Ours:
            self.task_iftrain[task] = 0

        self.task_total_num = torch.zeros(len(self.task_list))

        self.M = args.m_size
        self.Examplar_set = {'G1':[], 'G2':[], 'G3':[], 'G4':[], 'G5':[]}
        self.composition_test_cate = args.comp_cate



    def train(self, load=False):
        if load == True:
            latest_task = args.checkpoint
            latest_task_idx = self.task_list.index(latest_task)
            for idx, task in enumerate(self.task_list):
                if idx <=latest_task_idx:
                    self.task_iftrain[task] = 1

            checkpoint_model = self.args.output + '/' + args.checkpoint+'_LAST'
            # if not os.path.exists(self.args.output):
            #     os.mkdir(self.args.output)
            # best_path = os.path.join(self.args.output, task+'_BEST') # BEST or LAST?
            self.load(checkpoint_model)
            print('Success to load the checkpoint from the task ', latest_task)

        else:
            latest_task_idx = -1

        for task_idx, task in enumerate(self.task_list[latest_task_idx+1:]):  # for each task, train for several epochs
            print('======================== Now is task "', task, '" ========================')
            self.task_iftrain[task] = 1

            # Memory
            if args.memory:
                if task_idx != latest_task_idx + 1:
                    each_memory = int(self.M / task_idx)
                    data_info_path = (
                                'datasets/vqa/Partition_Q/karpathy_train_' + f'{self.task_list[task_idx - 1]}.json')
                    with open(data_info_path) as f:
                        data_info_dicts = json.load(f)

                    random.shuffle(data_info_dicts)  # shuffle
                    each_memory_for_cate = int(each_memory / len(Category_splits))

                    for cate in Category_splits:
                        num = 0
                        self.Examplar_set[cate].append([])
                        for _d in data_info_dicts:
                            img_id = _d['img_id']
                            if img_id in ImgId_cate_map:
                                if ImgId_cate_map[img_id] in Category_splits[cate]:
                                    self.Examplar_set[cate][task_idx - 1].append(_d)
                                    num += 1
                                    if num >= each_memory_for_cate:
                                        break

                    print('Load from Partition_Q_v3......')

                    for cate in Category_splits:
                        for i in range(task_idx):
                            self.Examplar_set[cate][i] = self.Examplar_set[cate][i][: each_memory_for_cate]

                    All_examplar = []
                    for E_set in self.Examplar_set:
                        for task_set in self.Examplar_set[E_set]:
                            All_examplar += task_set
                    # assert len(All_examplar) == M
                    print("# The size of the cate Memory:", len(All_examplar))
                else:
                    All_examplar = []
                    each_memory = 0
            else:
                All_examplar = []
                each_memory = 0

            # Load the data
            print("#Loading ", task)

            train_loader, total_num_Q = get_loader(
                args,
                self.coco_Ours,
                [],
                self.train_dset,
                split=args.train, mode='train', batch_size=args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
                task=task,
            )

            self.task_total_num[task_idx] = total_num_Q


            if args.valid_batch_size is not None:
                self.valid_batch_size = args.valid_batch_size
            else:
                self.valid_batch_size = args.batch_size
            print(f'Building val loader at GPU {args.gpu}')
            val_loader, _ = get_loader(
                args,
                self.coco_Ours,
                [],
                self.val_dset,
                split=args.valid, mode='val', batch_size=self.valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=4,
                topk=args.valid_topk,
                task=task,
            )

            print(f'Building test loader at GPU {args.gpu}')
            test_loader, _ = get_loader(
                args,
                self.coco_Ours,
                [],
                self.test_dset,
                split=args.test, mode='val', batch_size=self.valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=4,
                topk=args.valid_topk,
                task=task,
            )
            self.test_loader_dict[task] = test_loader

            test_loader = get_loader_test(
                args,
                self.coco_Ours,
                [],
                self.test_dset,
                split=args.test, mode='val', batch_size=self.valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=4,
                topk=args.valid_topk,
                task=task,
            )
            self.test_loader_dict_all[task] = test_loader

            print("#Loading ", task)
            memory_loader = get_loader_memory(
                args,
                self.coco_Ours,
                All_examplar,
                self.train_dset,
                split=args.train, mode='train', batch_size=args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )  # G1-G5

            if self.verbose:
                loss_meter = LossMeter()
                loss_meter_mem = LossMeter()
                # loss_mem_V = LossMeter()
                # loss_mem_Q_new = LossMeter()
                # loss_mem_V_new = LossMeter()
                best_valid = 0.
                best_epoch = 0

                if 't5' in self.args.backbone:
                    if self.args.use_vision:
                        project_name = "VLT5_VQA"
                    else:
                        project_name = "T5_VQA"
                elif 'bart' in self.args.backbone:
                    if self.args.use_vision:
                        project_name = "VLBart_VQA"
                    else:
                        project_name = "Bart_VQA"

                src_dir = Path(__file__).resolve().parent
                base_path = str(src_dir.parent)
                src_dir = str(src_dir)
                # wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

            if self.args.distributed:
                dist.barrier()

            global_step = 0
            Category_splits_random = random_dic(Category_splits)

            for idx, cateGroup in enumerate(Category_splits_random):
                print('-------- Training the cate group ', cateGroup,' of task ', task,'------')

                self.train_loader_cate = train_loader[cateGroup]
                self.val_loader_cate = val_loader[cateGroup]
                self.memory_loader_cate = memory_loader[cateGroup]

                # Optimizer
                if self.iftrain:
                    if len(self.memory_loader_cate.dataset) > 0:
                        total_train_num = 2 * len(self.train_loader_cate.dataset)
                    else:
                        total_train_num = len(self.train_loader_cate.dataset)
                    self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler(total_train_num)

                    if self.args.fp16 and _use_native_amp:
                        self.scaler = torch.cuda.amp.GradScaler()
                    elif _use_apex:
                        self.model, self.optim = amp.initialize(
                            self.model, self.optim, opt_level='O1', verbosity=self.verbose)

                if cateGroup == self.composition_test_cate and task != self.task_list[latest_task_idx+1]:
                    print("-------- Pass the training for", cateGroup, 'for after composition testing.--------')
                    continue


                for epoch in range(self.args.epochs):
                    if self.start_epoch is not None:
                        epoch += self.start_epoch
                    self.model.train()

                    if self.args.distributed:
                        self.train_loader_cate.sampler.set_epoch(epoch)
                    if self.verbose:
                        pbar = tqdm(total=len(self.train_loader_cate), ncols=120)

                    epoch_results = {
                        'loss': 0.,
                    }

                    quesid2ans = {}

                    if len(self.memory_loader_cate.dataset) > 0:
                        now_loader = zip(self.train_loader_cate, cycle(self.memory_loader_cate))
                        print('Use memory loader')
                    else:
                        now_loader = self.train_loader_cate

                    for now_batch in now_loader:
                        if len(now_batch) == 2:
                            batch, mem_batch = now_batch
                        else:
                            batch = now_batch
                            mem_batch = None

                        results, lr = self.train_step(batch, epoch_results, task_idx, each_memory)
                        if mem_batch:
                            results_mem, lr = self.train_step(mem_batch, epoch_results, task_idx, each_memory)

                        if self.verbose:
                            loss_meter.update(results['loss'].item())
                            desc_str = f'Epoch {epoch} | LR {lr:.6f}'
                            desc_str += f' | Loss {loss_meter.val:4f}'
                            if mem_batch:
                                loss_meter_mem.update(results_mem['loss'].item())
                                desc_str += f' | Loss_mem {loss_meter_mem.val:4f}'
                            else:
                                loss_meter_mem.update(-1)


                            pbar.set_description(desc_str)
                            pbar.update(1)

                        if self.args.distributed:
                            dist.barrier()
                        # break

                    if self.verbose:
                        pbar.close()

                    print("Loss:",loss_meter.val,' Loss_mem:', loss_meter_mem.val)

                    # Validation
                    score_dict = self.evaluate(self.val_loader_cate)

                    if self.verbose:
                        valid_score = score_dict['topk_score'] * 100.
                        valid_score_raw = score_dict['overall']

                        log_str = ''
                        log_str += "\nGroup %s Epoch %d: Valid Raw %0.2f Topk %0.2f" % (cateGroup, epoch, valid_score_raw, valid_score)

                        print(log_str)

                    if self.args.distributed:
                        dist.barrier()

            if self.verbose:
                self.save(task + "_LAST")

            # ========= Testing =========
            self.test(task)
            if self.composition_test_cate in Category_splits:
                self.test(task, comp=True)
                self.test_nocomp(task)

        try:
            Q_prototype = self.model.module.Q_prototype
            V_prototype = self.model.module.V_prototype
            torch.save(Q_prototype, args.output + "/Q_prototype.pt")
            torch.save(V_prototype, args.output + "/V_prototype.pt")
            print(" ======= Saved the learned prototypes ======= ")
        except:
            print('save prototype error')


    def train_step(self, batch, epoch_results, task_idx, each_memory):
        if self.args.fp16 and _use_native_amp:
            with autocast():
                if self.args.distributed:
                    results = self.model.module.train_step(batch)
                else:
                    results = self.model.train_step(batch)
        else:  # this
            if self.args.distributed:
                results = self.model.module.train_step(batch, task_idx, self.args.proto_alpha, self.args.proto_beta, each_memory, self.task_total_num)
            else:
                results = self.model.train_step(batch)

        loss = results['loss']
        lambda_Q = self.args.lambda_Q
        lambda_V = self.args.lambda_V
        lambda_Q_new = self.args.lambda_Q_new
        lambda_V_new = self.args.lambda_V_new

        if 'loss_memory' in results:
            (loss_memory_Q, loss_memory_V) = results['loss_memory']
            loss = loss + lambda_Q * loss_memory_Q + lambda_V * loss_memory_V
        if 'loss_memory_new' in results:
            (loss_memory_Q_new, loss_memory_V_new) = results['loss_memory_new']
            loss = loss + lambda_Q_new * loss_memory_Q_new + lambda_V_new * loss_memory_V_new

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optim) as scaled_loss:
                scaled_loss.backward()
        else:  # this
            loss.backward()

        loss = loss.detach()

        # Update Parameters
        if self.args.clip_grad_norm > 0:
            if self.args.fp16 and _use_native_amp:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.clip_grad_norm)
            elif self.args.fp16 and _use_apex:
                torch.nn.utils.clip_grad_norm_(amp.master_params(
                    self.optim), self.args.clip_grad_norm)
            else:  # this
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.clip_grad_norm)

        if self.args.fp16 and _use_native_amp:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:  # this
            self.optim.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()
        for param in self.model.parameters():
            param.grad = None

        # ----------
        # self.model.module.Q_prototype_param.data.clamp_(0, 1)
        # self.model.module.V_prototype_param.data.clamp_(0, 1)

        # global_step += 1

        for k, v in results.items():
            if k in epoch_results:
                epoch_results[k] += v.item()

        if self.lr_scheduler:
            if version.parse(torch.__version__) >= version.parse("1.4"):
                lr = self.lr_scheduler.get_last_lr()[0]
            else:
                lr = self.lr_scheduler.get_lr()[0]
        else:
            try:
                lr = self.optim.get_lr()[0]
            except AttributeError:
                lr = self.args.lr
        return results, lr

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
        if not self.args.now_train:
            self.model.module.Q_prototype = torch.load(self.args.output + '/Q_prototype.pt')
            self.model.module.V_prototype = torch.load(self.args.output + '/V_prototype.pt')

        # =========== test for all previous tasks
        Flag = 1
        for test_task in self.coco_Ours:
            if self.args.now_train:
                if self.task_iftrain[test_task] == 0:
                    Flag = 0
            if Flag == 1:
                if comp == True:
                    if test_task != self.task_list[0]:
                        cate_comp = self.composition_test_cate
                        self.test_loader = self.test_loader_dict[test_task][cate_comp]
                        print(' ===== Test for the task "' + test_task + '"  ====== for composition ' + cate_comp)
                    else:
                        continue
                else:
                    self.test_loader = self.test_loader_dict_all[test_task]
                    print(' ===== Test for the task "' + test_task + '"  ======')

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

                if comp:
                    self.result_matrix_comp[task][test_task] = acc_dict_all['overall']
                else:
                    self.result_matrix[task][test_task] = acc_dict_all['overall']

                if self.args.distributed:
                    dist.barrier()

    def test_nocomp(self, task):
        # Test Set
        if not os.path.exists(self.args.output):
            os.mkdir(self.args.output)
        last_path = os.path.join(self.args.output, task + '_LAST')
        self.load(last_path)

        # =========== test for all previous tasks
        for test_task in self.coco_Ours:
            if self.task_iftrain[test_task] == 1:
                if test_task != self.task_list[0]:
                    cate_comp = self.composition_test_cate
                    correct_num, sum_num = 0, 0
                    print(' ===== Test for the task "' + test_task + '"  ====== for non composition ' + cate_comp)

                    for cate in Category_splits:
                        if cate == cate_comp:
                            continue
                        self.test_loader = self.test_loader_dict[test_task][cate]
                        quesid2ans = self.predict(self.test_loader)

                        if self.verbose:
                            evaluator = self.test_loader.evaluator
                            score_dict = evaluator.evaluate(quesid2ans)

                            acc_dict_all = evaluator.evaluate_raw(quesid2ans)  # here

                            correct_num += acc_dict_all['overall'] * len(self.test_loader.dataset)
                            sum_num += len(self.test_loader.dataset)

                    print(test_task, 'non composition', 'Test/Overall:', round(float(correct_num/sum_num),2))

                    self.result_matrix_noncomp[task][test_task] = round(float(correct_num/sum_num),2)


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

                pred_ans = results['pred_ans'] # generated_sents
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

    coco_Ours = All_task #['q_causal', 'q_color','q_judge']#All_task


    trainer = Trainer(args, coco_Ours, train=True)

    if args.now_train:
        if args.checkpoint != 'None':
            trainer.train(load=True)
        else:
            trainer.train(load=False)

        print('#------------------ result_matrix --------------------#')
        show_results_matrix(trainer.result_matrix)
        path = args.output + 'results_matrix.json'
        # save_results_matrix(trainer.result_matrix, path)
        metric_dict = evaluate_metric(trainer.result_matrix)
        print('#------  Metric  ------#')
        print('Incremental avg accuracy:', metric_dict['Incre_avg_acc'])
        print('*** Avg accuracy ***', metric_dict['Avg_acc'])
        print('Incremental avg forget:', metric_dict['Incre_avg_forget'])
        print('*** Avg forget ***', metric_dict['Avg_forget'])
        print('6Q Incremental avg accuracy:', metric_dict['Incre_avg_acc_6Q'])
        print('*** _6Q Avg accuracy ***', metric_dict['Avg_acc_6Q'])
        print('_6Q Incremental avg forget:', metric_dict['Incre_avg_forget_6Q'])
        print('*** _6Q Avg forget ***', metric_dict['Avg_forget_6Q'])

        print('#------------------ result_matrix_comp --------------------#')
        show_results_matrix(trainer.result_matrix_comp, start=1)
        # save_results_matrix(trainer.result_matrix_comp, path)
        metric_dict_comp = evaluate_metric(trainer.result_matrix_comp, start=1)
        print('#------ Metric ------#')
        print('Incremental avg accuracy:', metric_dict_comp['Incre_avg_acc'])
        print('*** Avg accuracy ***', metric_dict_comp['Avg_acc'])
        print('Incremental avg forget:', metric_dict_comp['Incre_avg_forget'])
        print('*** Avg forget ***', metric_dict_comp['Avg_forget'])
        print('6Q Incremental avg accuracy:', metric_dict_comp['Incre_avg_acc_6Q'])
        print('*** _6Q Avg accuracy ***', metric_dict_comp['Avg_acc_6Q'])
        print('_6Q Incremental avg forget:', metric_dict_comp['Incre_avg_forget_6Q'])
        print('*** _6Q Avg forget ***', metric_dict_comp['Avg_forget_6Q'])

        print('#------------------ result_matrix_noncomp --------------------#')
        show_results_matrix(trainer.result_matrix_noncomp, start=1)
        # save_results_matrix(trainer.result_matrix_noncomp, path)
        metric_dict_noncomp = evaluate_metric(trainer.result_matrix_noncomp, start=1)
        print('#------ Metric ------#')
        print('Incremental avg accuracy:', metric_dict_noncomp['Incre_avg_acc'])
        print('*** Avg accuracy ***', metric_dict_noncomp['Avg_acc'])
        print('Incremental avg forget:', metric_dict_noncomp['Incre_avg_forget'])
        print('*** Avg forget ***', metric_dict_noncomp['Avg_forget'])
        print('6Q Incremental avg accuracy:', metric_dict_noncomp['Incre_avg_acc_6Q'])
        print('*** _6Q Avg accuracy ***', metric_dict_noncomp['Avg_acc_6Q'])
        print('_6Q Incremental avg forget:', metric_dict_noncomp['Incre_avg_forget_6Q'])
        print('*** _6Q Avg forget ***', metric_dict_noncomp['Avg_forget_6Q'])

    else:
        if args.checkpoint!='None':
            trainer.Test(load=True)
        else:
            trainer.Test(load=False)

        try:
            print('#------------------ Final Performance with 6 question types --------------------#')
            acc = 0
            dict_final = {}
            for key in trainer.result_matrix_comp['q_causal']:
                if key in Comp_task:
                    dict_final[key] = trainer.result_matrix_comp['q_causal'][key]
                    acc += trainer.result_matrix_comp['q_causal'][key]
            print(dict_final)
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
