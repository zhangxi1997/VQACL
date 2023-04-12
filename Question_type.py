import random
seed = 66666

random.seed(seed)
print('random seed', seed)

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dict = {}
    for key in dict_key_ls:
        new_dict[key] = dicts.get(key)
    return new_dict

#  10 lingustic-driven task for VQA v2
All_task = ['q_recognition', 'q_location', 'q_judge', 'q_commonsense', 'q_count','q_action', 'q_color', 'q_type', 'q_subcategory','q_causal']
Comp_task = ['q_location', 'q_count', 'q_action', 'q_color', 'q_type', 'q_subcategory']

# visual-driven task for VQA v2
Category_splits = {'G1': [58, 48, 55, 36, 64, 1, 70, 73, 42, 15, 6, 18, 49, 59, 31, 2],\
                   'G2': [19, 77, 22, 9, 24, 53, 12, 13, 78, 50, 47, 41, 32, 28, 54, 23],\
                   'G3': [60, 8, 34, 25, 67, 4, 14, 68, 3, 79, 0, 5, 65, 20, 71, 39], \
                   'G4': [35, 29, 66, 40, 43, 26, 72, 10, 38, 61, 76, 44, 75, 69, 16, 57], \
                   'G5': [45, 33, 63, 56, 21, 11, 62, 74, 17, 52, 46, 30, 27, 51, 37, 7]}


import json

with open('datasets/QuesId_task_map.json') as fp:
    QuesId_task_map = json.load(fp)

with open('datasets/ImgId_cate_map.json') as fp:
    ImgId_cate_map = json.load(fp)

print("Success to load the QuesId_task_map and QuesId_task_map")


_6Q_idx = []

_key_list = []
for key in All_task:
    _key_list.append(key)

for key in Comp_task:
    idx = _key_list.index(key)
    _6Q_idx.append(idx)

All_task_list = []
for key in All_task:
    All_task_list.append(key)


import matplotlib.pyplot as plt
import numpy

def show_results_matrix(results, start=0):
    matrix = numpy.zeros([len(results), len(results)], dtype=float)
    key_list = []
    for key in results:
        print(key, end='\t')
        key_list.append(key)
    print('\n')
    for i in range(start,len(results)):
        avg = 0
        # print('T1   ', end='\t')
        for j in range(start,len(results)):
            if j < i+1:
                matrix[i][j] = results[key_list[i]][key_list[j]]
                avg += matrix[i][j]
            print(round(matrix[i][j], 2), end='\t')

        print("Avg:", round(avg / (len(results)-start),2))
    # print(matrix)
    # print('Average for each step:')
    # print(numpy.mean(matrix, axis=1))



def Update_memory(M, task_idx, task):
    Examplar_set = {'G0': [], 'G1': [], 'G2': [], 'G3': [], 'G4': []}
    each_memory = int(M / (task_idx + 1))
    data_info_path = ('/data/zhangxi/vqa/Partition_Q_v2/karpathy_train_' + f'{task}.json')
    with open(data_info_path) as f:
        data_info_dicts = json.load(f)
    # data_info_dicts = dataset.items()

    random.shuffle(data_info_dicts)  # shuffle
    each_memory_for_cate = int(each_memory / len(Category_splits))

    for cate in Category_splits:
        num = 0
        Examplar_set[cate].append([])
        for _d in data_info_dicts:
            img_id = _d['img_id']
            if img_id in ImgId_cate_map:
                if ImgId_cate_map[img_id] in Category_splits[cate]:
                    Examplar_set[cate][task_idx].append(_d)
                    num += 1
                    if num >= each_memory_for_cate:
                        break

    for cate in Category_splits:
        for i in range(task_idx):
            Examplar_set[cate][i] = Examplar_set[cate][i][: each_memory_for_cate]
    return Examplar_set

def evaluate_metric(results, start=0):
    matrix = numpy.zeros([len(results), len(results)], dtype=float)-1
    key_list = []
    for key in results:
        key_list.append(key)
    for i in range(start, len(results)):
        avg = 0
        for j in range(start, len(results)):
            if j < i + 1:
                matrix[i][j] = results[key_list[i]][key_list[j]]

    # Incremental Acc performance
    Incre_avg_accuracy = []
    Incre_avg_accuracy_6Q = []
    for t in range(start, len(results)):
        now_acc = matrix[t]
        all_acc = 0
        num = 0
        for acc in now_acc:
            if acc != -1:
                all_acc += acc
                num += 1
        avg_acc = all_acc / num
        Incre_avg_accuracy.append(avg_acc)

        all_acc = 0
        num = 0
        for i in range(len(now_acc)):
            if i in _6Q_idx:
                if now_acc[i] != -1:
                    all_acc += now_acc[i]
                    num += 1
        if num!=0:
            avg_acc = all_acc / num
        else:
            avg_acc = -1
        Incre_avg_accuracy_6Q.append(avg_acc)

    # Incre_avg_accuracy = numpy.mean(matrix, axis=1)
    # Avg Accuracy
    Avg_accuracy = Incre_avg_accuracy[-1]
    Avg_accuracy_6Q = Incre_avg_accuracy_6Q[-1]

    # Forget
    Incre_avg_forget = [0]
    Incre_avg_forget_6Q = [0]
    for t in range(1+start,len(results)):
        results_now = matrix[:t+1, :t+1]
        t_forget = []
        for idx in range(start, len(results_now)-1):
            task_list = results_now[:-1,idx]
            final = results_now[-1,idx]
            pre_max = max(task_list)
            if pre_max == -1:
                t_forget.append(0)
            else:
                t_forget.append(pre_max - final)
        Avg_forget = sum(t_forget)/len(t_forget)
        Incre_avg_forget.append(Avg_forget)


        t_forget_6Q = []
        for i_ in range(len(t_forget)):
            if i_+1 in _6Q_idx:
                t_forget_6Q.append(t_forget[i_])
        if len(t_forget_6Q) > 0:
            Avg_forget = sum(t_forget_6Q) / len(t_forget_6Q)
        else:
            Avg_forget = -1
        Incre_avg_forget_6Q.append(Avg_forget)

    Avg_forget = Incre_avg_forget[-1]
    Avg_forget_6Q = Incre_avg_forget_6Q[-1]



    # # BWT
    # for t in range(len(results)):
    #     t_acc = matrix[:,t]
    #     final = t_acc[-1]
    #     sum = 0
    #     for i in range(len(t_acc)-1):
    #         sum += final-t_acc[i]


    output_dict = {'Incre_avg_acc': Incre_avg_accuracy,
                   'Avg_acc': Avg_accuracy,
                   'Incre_avg_forget': Incre_avg_forget,
                   'Avg_forget':Avg_forget,
                   'Incre_avg_acc_6Q': Incre_avg_accuracy_6Q,
                   'Avg_acc_6Q': Avg_accuracy_6Q,
                   'Incre_avg_forget_6Q': Incre_avg_forget_6Q,
                   'Avg_forget_6Q': Avg_forget_6Q,
                   }
    return output_dict





if __name__ == "__main__":
    result_matrix = {}
