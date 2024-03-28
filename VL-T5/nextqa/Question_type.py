
import json
import pandas as pd
import random

All_Q = ['CW', 'TN', 'TC', 'DL', 'DB', 'DC', 'DO', 'CH'] # CW
All_Q2 = ['DL', 'DB', 'CW', 'TN', 'TC',  'DC', 'DO', 'CH'] # begin with DL
All_Q3 = ['TN', 'DB', 'CW', 'DL', 'TC',  'DC', 'DO', 'CH'] # begin with TN




All_V = {'G0': [50, 60, 15, 32, 79, 27, 53, 20, 36, 28, 24, 80, 41, 33, 18, 10], \
'G1': [54, 5, 57, 49, 69, 62, 7, 1, 14, 35, 56, 66, 58, 51, 46, 6], \
'G2': [59, 61, 74, 37, 47, 34, 19, 72, 75, 23, 63, 40, 67, 21, 73, 29],\
'G3': [22, 2, 48, 64, 68, 9, 65, 26, 45, 12, 8, 76, 55, 4, 77, 44],\
'G4': [78, 17, 52, 11, 30, 13, 38, 70, 25, 3, 43, 42, 39, 16, 71, 31]}


cate = {"cat": 1, "stingray": 2, "cellphone": 3, "stop sign": 24, "panda": 4, "camera": 5, "stool": 6, "baby walker": 7, "turtle": 8, "duck": 9, "racket": 10, "bottle": 11, "cake": 12, "aircraft": 13, "squirrel": 14, "bat": 15, \
 "chair": 16, "faucet": 17, "toilet": 18, "suitcase": 19, "hamster/rat": 20, "snowboard": 21, "ski": 22, "bench": 23, "stop_sign": 24, "baby_seat": 25, "dish": 26, "sofa": 27, "oven": 28, "handbag": 29, "bus/truck": 30,\
 "baby seat": 25, "refrigerator": 31, "microwave": 32, "bird": 33, "pig": 34, "frisbee": 35, "chicken": 36, "train": 37, "baby": 38, "backpack": 39, "motorcycle": 40, "skateboard": 41, "rabbit": 42, "sink": 43, "cup": 44, \
 "baby_walker": 7, "fish": 45, "electric fan": 46, "fruits": 47, "antelope": 48, "ball/sports ball": 49, "bicycle": 50, "ball/sports_ball": 49, "scooter": 51, "car": 52, "traffic light": 53, "crab": 54, "laptop": 55, "cattle/cow": 56, \
 "lion": 57, "adult": 58, "piano": 59, "camel": 60, "watercraft": 61, "screen/monitor": 62, "elephant": 63, "toy": 64, "guitar": 65, \
 "sheep/goat": 66, "horse": 67, "child": 68, "electric_fan": 46, "crocodile": 69, "bread": 70, "dog": 71, "bear": 72, "surfboard": 73, "kangaroo": 74, "tiger": 75, "leopard": 76, "table": 77, "penguin": 78, "snake": 79, "vegetables": 80, "traffic_light": 53}




All_V2 = {'G0': [50, 60, 15, 32, 79, 27, 53, 20, 36, 28, 24, 80, 41, 33, 18, 10], \
'G1': [54, 5, 57, 49, 69, 62, 7, 1, 14, 35, 56, 66, 58, 51, 46, 6]}
# , \
# 'G2': [59, 61, 74, 37, 47, 34, 19, 72, 75, 23, 63, 40, 67, 21, 73, 29],\
# 'G3': [22, 2, 48, 64, 68, 9, 65, 26, 45, 12, 8, 76, 55, 4, 77, 44],\
# 'G4': [78, 17, 52, 11, 30, 13, 38, 70, 25, 3, 43, 42, 39, 16, 71, 31]}


import random
def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dict = {}
    for key in dict_key_ls:
        new_dict[key] = dicts.get(key)
    return new_dict

import numpy
import numpy
def show_results_matrix(results, start=0):
    # results = {}
    # results['what'] = {'what':31.0}
    # results['color'] = {'what': 1.95,'color':67.05}
    # results['is'] = {'what': 1.02,'color':0.39, 'is': 71.6}
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

def save_results_matrix(results, path):
    dict_json = json.dumps(results)
    with open(path, 'w') as file:
        file.write(dict_json)
    print('The result matrix has been wrote into ', path)

_6Q = ['CW', 'TN', 'TC', 'DL', 'DB', 'DC', 'DO', 'CH']
_6Q_idx = []
for key in _6Q:
    idx = _6Q.index(key)
    _6Q_idx.append(idx)

def evaluate_metric(results, start=0):
    matrix = numpy.zeros([len(results), len(results)], dtype=float)-1
    key_list = []
    for key in results:
        # print(key, end='\t')
        key_list.append(key)
    # print('\n')
    for i in range(start, len(results)):
        avg = 0
        # print('T1   ', end='\t')
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
        for idx in range(start, len(results_now)-1): # T-1
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
        for i_ in range(len(t_forget)):#len(results_now)-1):
            if i_+1 in _6Q_idx:
                t_forget_6Q.append(t_forget[i_])
        if len(t_forget_6Q) > 0:
            Avg_forget = sum(t_forget_6Q) / len(t_forget_6Q)
        else:
            Avg_forget = -1
        Incre_avg_forget_6Q.append(Avg_forget)

    Avg_forget = Incre_avg_forget[-1]
    Avg_forget_6Q = Incre_avg_forget_6Q[-1]


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

