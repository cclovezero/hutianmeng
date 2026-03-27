import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from sample import Sample
from utils import get_ep_indices

'''
Define a external pareto class storing all computed policies on the current pareto front.
'''
# class EP:
#     def __init__(self):
#         self.obj_batch = np.array([])
#         self.sample_batch = np.array([])
#
#     def index(self, indices, inplace=True):
#         if inplace:
#             self.obj_batch, self.sample_batch = \
#                 map(lambda batch: batch[np.array(indices, dtype=int)], [self.obj_batch, self.sample_batch])
#         else:
#             return map(lambda batch: deepcopy(batch[np.array(indices, dtype=int)]), [self.obj_batch, self.sample_batch])
#
#     def update(self, sample_batch):
#         self.sample_batch = np.append(self.sample_batch, np.array(deepcopy(sample_batch)))
#         for sample in sample_batch:
#             self.obj_batch = np.vstack([self.obj_batch, sample.objs]) if len(self.obj_batch) > 0 else np.array([sample.objs])
#
#         if len(self.obj_batch) == 0: return
#         ep_indices = get_ep_indices(self.obj_batch)
#
#         self.index(ep_indices)


# 定义一个EP类
class EP:
    # 初始化方法，创建两个空数组，用来存储目标和样本
    def __init__(self):
        self.obj_batch = np.array([])
        self.sample_batch = np.array([])

    # 索引方法，根据给定的索引值将目标和样本数组进行切片操作
    def index(self, indices, inplace=True):
        if inplace:
            # 如果 inplace 参数为True，那么就直接对原来的数组进行切片操作
            self.obj_batch, self.sample_batch = \
                map(lambda batch: batch[np.array(indices, dtype=int)], [self.obj_batch, self.sample_batch])
        else:
            # 如果 inplace 参数为False，那么就对切片后的数组进行深拷贝并返回
            return map(lambda batch: deepcopy(batch[np.array(indices, dtype=int)]), [self.obj_batch, self.sample_batch])

    # 更新方法，将新的样本添加到样本数组中，并根据新样本的目标值更新目标数组
    def update(self, sample_batch):
        # 将新的样本添加到样本数组中
        self.sample_batch = np.append(self.sample_batch, np.array(deepcopy(sample_batch)))

        # 遍历新的样本，将目标值添加到目标数组中
        for sample in sample_batch:
            self.obj_batch = np.vstack([self.obj_batch, sample.objs]) if len(self.obj_batch) > 0 else np.array(
                [sample.objs])

        # 如果目标数组为空，直接返回
        if len(self.obj_batch) == 0: return

        # 获取非支配解的索引  get_ep_indices函数负责计算非支配解。
        ep_indices = get_ep_indices(self.obj_batch)

        # 根据索引值将目标和样本数组进行切片操作
        self.index(ep_indices)

    def random_selection(self, args, scalarization_template):
        elite_batch, scalarization_batch = [], []
        weights_all = []
        weights_all_t = []
        dist = []

        for i in range(len(self.sample_batch)):
            weights_all.append(self.sample_batch[i].weight)
        weights_all_t = torch.stack(deepcopy(weights_all))  # 所有的权重
        weights_all_t = torch.nn.functional.normalize(weights_all_t, p=2, dim=1)

        for _ in range(args.num_tasks):
            elite_idx = np.random.choice(len(self.sample_batch))

            # weights = np.random.uniform(args.min_weight, args.max_weight, args.obj_num)
            weights = deepcopy(self.sample_batch[elite_idx].weight)
            weights = weights / torch.sum(weights)
            weights = torch.nn.functional.normalize(weights, dim=0)  # 归一化处理，以便计算距离。

            for ii in range(len(self.sample_batch)):
                dist.append(torch.dist(weights, weights_all_t[ii, :]))  # 当前权重与所有权重的欧式距离
            dist = torch.stack(dist)

            # 查找dist中最小的k个元素的 值和索引。
            values, indices = torch.topk(dist, k=5, largest=False)
            weights_other = deepcopy(weights_all_t[indices])

            self.sample_batch[elite_idx].neighbor_weight = weights_other
            elite_batch.append(self.sample_batch[elite_idx])

            scalarization = deepcopy(scalarization_template)
            weights = weights / torch.sum(weights)
            scalarization.update_weights(weights)
            scalarization_batch.append(scalarization)

            dist = []  # 重置dist


        return elite_batch, scalarization_batch