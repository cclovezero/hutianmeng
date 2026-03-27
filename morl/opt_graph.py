from copy import deepcopy
import numpy as np

'''
OptGraph is a data structure to store the optimization history.
The optimization history is a rooted forest, and is organized in a tree structure.
OptGraph是一个用于存储优化历史的数据结构。
优化历史是一个有根的森林，并以树状结构组织。

这段代码定义了一个OptGraph类，用于存储一个优化问题的解空间的图形表示。
类中包含了初始化方法__init__和一个用于向图中插入新解的方法insert。
其中，__init__方法初始化了类的各个属性；insert方法将一个新的解插入到优化图中，并返回该解在图中的下标。
'''
# class OptGraph:
#     def __init__(self):
#         self.weights = []
#         self.objs = []
#         self.delta_objs = []
#         self.prev = []
#         self.succ = []
#
#     def insert(self, weights, objs, prev):
#         self.weights.append(deepcopy(weights) / np.linalg.norm(weights))
#         self.objs.append(deepcopy(objs))
#         self.prev.append(prev)
#         if prev == -1:
#             self.delta_objs.append(np.zeros_like(objs))
#         else:
#             self.delta_objs.append(objs - self.objs[prev])
#         if prev != -1:
#             self.succ[prev].append(len(self.objs) - 1)
#         self.succ.append([])
#         return len(self.objs) - 1


class OptGraph:
    def __init__(self):
        # 初始化OptGraph对象的weights、objs、delta_objs、prev、succ等属性。
        self.weights = []      # 存储优化问题中每个解的权重向量
        self.objs = []         # 存储优化问题中每个解的目标函数值
        self.delta_objs = []   # 存储优化问题中每个解与其前驱解之间的目标函数差值
        self.prev = []         # 存储优化问题中每个解的前驱解的下标
        self.succ = []         # 存储优化问题中每个解的后继解的下标列表

    def insert(self, weights, objs, prev):
        # 将一个新的解插入到优化图中

        self.weights.append(deepcopy(weights) / np.linalg.norm(weights))
        # 这里存储的是归一化的权重，例如（0.2，0.8）被归一化为（0.2425,0.9701)
        '''  np.linalg.norm(weights)是numpy的求范数函数，它可以计算numpy数组的范数。
        在这里，它计算了weights数组的二范数，
        即$\sqrt{\sum_{i=1}^{n} weights_i^2}$，其中$n$是weights数组的长度，$weights_i$是数组中的第$i$个元素。
        这个值被用作权重向量的长度，以便对其进行归一化。具体来说，对权重向量进行归一化可以将其转换为单位向量，使得它的长度为1.
        这样就可以更好地比较不同向量之间的相似性或距离。 '''

        self.objs.append(deepcopy(objs))
        self.prev.append(prev)
        # 如果新的解没有前驱解，则将其delta_objs设置为全0的向量。
        if prev == -1:
            self.delta_objs.append(np.zeros_like(objs))
        # 如果新的解有前驱解，则将其delta_objs设置为与前驱解的目标函数值的差值。
        else:
            self.delta_objs.append(objs - self.objs[prev])
        # 如果前驱解存在，则将新的解加入前驱解的后继解列表中。
        if prev != -1:
            self.succ[prev].append(len(self.objs) - 1)
        # 将新的解的后继解列表初始化为空列表，并返回其下标。
        self.succ.append([])
        return len(self.objs) - 1


