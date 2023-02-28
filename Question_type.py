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

#  10 task
All_task = ['q_recognition', 'q_location', 'q_judge', 'q_commonsense', 'q_count','q_action', 'q_color', 'q_type', 'q_subcategory','q_causal']
Comp_task = ['q_location', 'q_count', 'q_action', 'q_color', 'q_type', 'q_subcategory']


# All_Q_v4_list = []
# for key in All_Q_v4:
#     All_Q_v4_list.append(key)


Category_splits = {'G1': [58, 48, 55, 36, 64, 1, 70, 73, 42, 15, 6, 18, 49, 59, 31, 2],\
                   'G2': [19, 77, 22, 9, 24, 53, 12, 13, 78, 50, 47, 41, 32, 28, 54, 23],\
                   'G3': [60, 8, 34, 25, 67, 4, 14, 68, 3, 79, 0, 5, 65, 20, 71, 39], \
                   'G4': [35, 29, 66, 40, 43, 26, 72, 10, 38, 61, 76, 44, 75, 69, 16, 57], \
                   'G5': [45, 33, 63, 56, 21, 11, 62, 74, 17, 52, 46, 30, 27, 51, 37, 7]}


import json

with open('/data/zhangxi/vqa/QuesId_task_map.json') as fp:
    QuesId_task_map = json.load(fp)

with open('/data/zhangxi/vqa/ImgId_cate_map.json') as fp:
    ImgId_cate_map = json.load(fp)

print("Success to load the QuesId_task_map and QuesId_task_map")


# _6Q_idx = []
#
# _key_list = []
# for key in All_Q_v4:
#     _key_list.append(key)
#
# for key in Comp_task:
#     idx = _key_list.index(key)
#     _6Q_idx.append(idx)


import matplotlib.pyplot as plt

def Update_memory(M, task_idx, task):
    Examplar_set = {'G0': [], 'G1': [], 'G2': [], 'G3': [], 'G4': []}
    # ==================== 建立Memory, 当前是第idx个任务 =================
    each_memory = int(M / (task_idx + 1))
    # 增加第idx-1任务的memory
    data_info_path = ('/data/zhangxi/vqa/Partition_Q_v2/karpathy_train_' + f'{task}.json')
    with open(data_info_path) as f:
        data_info_dicts = json.load(f)
    # data_info_dicts = dataset.items()

    random.shuffle(data_info_dicts)  # shuffle
    each_memory_for_cate = int(each_memory / len(Category_splits))

    for cate in Category_splits:  # 保证每个Category都有memory
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

    # 减少0——idx-2任务的memory大小
    for cate in Category_splits:
        for i in range(task_idx):
            Examplar_set[cate][i] = Examplar_set[cate][i][: each_memory_for_cate]
    return Examplar_set




if __name__ == "__main__":
    # show_results_matrix([1,2])
    # plot_result([],'/home/mmc_zhangxi/result.png')
    result_matrix = {}
