import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from sample import Sample
from utils import get_ep_indices
from scipy.optimize import least_squares
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import math
import random
from torch.multiprocessing import Process, Queue, Event

def collect_nearest_data(opt_graph, optgraph_id, threshold = 0.1):
    objs_data, weights_data, delta_objs_data = [], [], []
    for i in range(len(opt_graph.objs)):
        diff = np.abs(opt_graph.objs[optgraph_id] - opt_graph.objs[i])
        if np.all(diff < np.abs(opt_graph.objs[optgraph_id]) * threshold):
            for next_index in opt_graph.succ[i]:
                objs_data.append(opt_graph.objs[i])
                weights_data.append(opt_graph.weights[next_index] / np.sum(opt_graph.weights[next_index]))
                delta_objs_data.append(opt_graph.delta_objs[next_index])
    return objs_data, weights_data, delta_objs_data

'''
train the hyperbolic prediction function for policy of a given optgraph_id.
given the predicted objectives for the test_weights.
'''
def predict_hyperbolic(args, opt_graph, optgraph_id, test_weights):
    test_weights = np.array(test_weights)

    # normalize the test_weights to be sum = 1
    for test_weight in test_weights:
        test_weight /= np.sum(test_weight)
    
    threshold = 0.1
    sigma = 0.03
    # gradually enlarging the searching range so that get enough data point to fit the model
    while True:
        objs_data, weights_data, delta_objs_data = collect_nearest_data(opt_graph, optgraph_id, threshold)
        cnt_data = 0
        for i in range(len(weights_data)):
            flag = True
            for j in range(i):
                if np.linalg.norm(weights_data[i] - weights_data[j]) < 1e-5:
                    flag = False
                    break
            if flag:
                cnt_data += 1
                if cnt_data > 3:
                    break
        if cnt_data > 3:
            break
        else:
            threshold *= 2.0
            sigma *= 2.0

    def f(x, A, a, b, c):
        return A * (np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1) + c

    def fun(params, x, y):
        # f = A * (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1) + c
        return (params[0] * (np.exp(params[1] * (x - params[2])) - 1.) / (np.exp(params[1] * (x - params[2])) + 1) + params[3] - y) * w

    def jac(params, x, y):
        A, a, b, c = params[0], params[1], params[2], params[3]

        J = np.zeros([len(params), len(x)])

        # df_dA = (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1)
        J[0] = ((np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1)) * w

        # df_da = A(x - b)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
        J[1] = (A * (x - b) * (2. * np.exp(a * (x - b))) / ((np.exp(a * (x - b)) + 1) ** 2)) * w

        # df_db = A(-a)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
        J[2] = (A * (-a) * (2. * np.exp(a * (x - b))) / ((np.exp(a * (x - b)) + 1) ** 2)) * w

        # df_dc = 1
        J[3] = w

        return np.transpose(J)

    M = args.obj_num
    delta_predictions = []
    for dim in range(M):
        train_x = []
        train_y = []
        w = []
        for i in range(len(objs_data)):
            train_x.append(weights_data[i][dim])
            train_y.append(delta_objs_data[i][dim])
            diff = np.abs(objs_data[i] - opt_graph.objs[optgraph_id])
            dist = np.linalg.norm(diff / np.abs(opt_graph.objs[optgraph_id]))
            coef = np.exp(-((dist  / sigma) ** 2) / 2.0)
            w.append(coef)
        
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        w = np.array(w)

        A_upperbound = np.clip(np.max(train_y) - np.min(train_y), 1.0, 500.0)
        params0 = np.ones(4)
        
        f_scale = 20.

        # fit the prediction function by minimizing soft_l1 loss.
        res_robust = least_squares(fun, params0, loss='soft_l1', f_scale = f_scale, args = (train_x, train_y), jac = jac, bounds = ([0, 0.1, -5., -500.], [A_upperbound, 20., 5., 500.]))
        
        delta_predictions.append(f(test_weights.T[dim], *res_robust.x))

    predictions = []
    delta_predictions = np.transpose(np.array(delta_predictions))
    original_objs = opt_graph.objs[optgraph_id]
    for i in range(len(test_weights)):
        predictions.append(original_objs + delta_predictions[i])

    results = {'sample_index': optgraph_id, 'predictions': predictions}

    return results

'''
Population class maintains the population of the policies by performance buffer strategy.
'''
class Population:
    def __init__(self, args):
        self.sample_batch = [] # all samples in population
        self.pbuffer_num = args.pbuffer_num
        self.pbuffer_size = args.pbuffer_size
        self.dtheta = np.pi / 2.0 / self.pbuffer_num
        self.z_min = np.zeros(args.obj_num) # reference point
        self.pbuffers = None
        self.pbuffer_dist = None

    '''
    insert the sample to the performance buffers (storing the index).
    '''
    def insert_pbuffer(self, index, objs):
        f = objs - self.z_min
        if np.min(f) < 1e-7:
            return False

        dist = np.linalg.norm(f)
        theta = np.arccos(np.clip(f[1] / dist, -1.0, 1.0))
        buffer_id = int(theta // self.dtheta)
        if buffer_id < 0 or buffer_id >= self.pbuffer_num:
            return False

        inserted = False
        # insert sample into the corresponding pbuffer if its distance to origin is top pbuffer_size
        # store the samples in each pbuffer in order of distance
        for i in range(len(self.pbuffers[buffer_id])):
            if self.pbuffer_dist[buffer_id][i] < dist:
                self.pbuffers[buffer_id].insert(i, index)
                self.pbuffer_dist[buffer_id].insert(i, dist)
                inserted = True
                break
        if inserted and len(self.pbuffers[buffer_id]) > self.pbuffer_size:
            self.pbuffers[buffer_id] = self.pbuffers[buffer_id][:self.pbuffer_size]
            self.pbuffer_dist[buffer_id] = self.pbuffer_dist[buffer_id][:self.pbuffer_size]
        elif (not inserted) and len(self.pbuffers[buffer_id]) < self.pbuffer_size:
            self.pbuffers[buffer_id].append(index)
            self.pbuffer_dist[buffer_id].append(dist)
            inserted = True

        return inserted

    '''
    update the population by a new offspring sample_batch.
    '''  
    def update(self, sample_batch):
        ### population = Union(population, offspring) ###
        all_sample_batch = self.sample_batch + sample_batch
        
        self.sample_batch = []
        self.pbuffers = [[] for _ in range(self.pbuffer_num)]       # store the sample indices in each pbuffer
        self.pbuffer_dist = [[] for _ in range(self.pbuffer_num)]   # store the sample distance in each pbuffer

        ### select the population by performance buffer ###       
        for i, sample in enumerate(all_sample_batch):
            self.insert_pbuffer(i, sample.objs)
        
        for pbuffer in self.pbuffers:
            for index in pbuffer:
                self.sample_batch.append(all_sample_batch[index])

    def compute_hypervolume(self, objs_batch):
        ep_objs_batch = deepcopy(np.array(objs_batch)[get_ep_indices(objs_batch)])
        ref_x, ref_y = 0.0, 0.0
        x, hv = ref_x, 0.0
        for objs in ep_objs_batch:
            hv += (max(ref_x, objs[0]) - x) * (max(ref_y, objs[1]) - ref_y)
            x = max(ref_x, objs[0])
        return hv

    def compute_sparsity(self, objs_batch):
        ep_objs_batch = deepcopy(np.array(objs_batch)[get_ep_indices(objs_batch)])
        if len(ep_objs_batch) < 2:
            return 0.0
        sparsity = 0.0
        for i in range(1, len(ep_objs_batch)):
            sparsity += np.sum(np.square(ep_objs_batch[i] - ep_objs_batch[i - 1]))
        sparsity /= (len(ep_objs_batch) - 1)
        return sparsity
    
    '''
    evaluate the hv value after virtually inserting each predicted offspring.
    '''
    def evaluate_hv(self, candidates, mask, virtual_ep_objs_batch):
        hv = [0.0 for _ in range(len(candidates))]
        for i in range(len(candidates)):
            if mask[i]:
                new_objs_batch = np.array(virtual_ep_objs_batch + [candidates[i]['prediction']])
                hv[i] = self.compute_hypervolume(new_objs_batch)
        return hv

    '''
    evaluate the sparsity value after virtually inserting each predicted offspring.
    '''
    def evaluate_sparsity(self, candidates, mask, virtual_ep_objs_batch):
        sparsity = [0.0 for _ in range(len(candidates))]
        for i in range(len(candidates)):
            if mask[i]:
                new_objs_batch = np.array(virtual_ep_objs_batch + [candidates[i]['prediction']])
                sparsity[i] = self.compute_sparsity(new_objs_batch)     
        return sparsity

    '''
    The prediction-guided task selection.
    '''
    def prediction_guided_selection(self, args, iteration, ep, opt_graph, scalarization_template):
        N = args.num_tasks # number of (sample, weight) to be selected
        num_weights = args.num_weight_candidates  # 设置每个策略的候选权重数 为7

        ### Prediction ###

        candidates = []   # 存储候选样本
        # list of candidate, each candidate is a (sample, weight) pair associated with their predicted future point
        for sample in self.sample_batch:    # 对于每个样本  sample_batch是当前种群的所有策略（样本）
            # get weights evenly distributed around the last weight direction and discard the weight outside the first quadrant
            # 下面的代码：获得均匀分布在最后一个权重方向的几个权重，并丢弃第一象限以外的权重
            weight_center = opt_graph.weights[sample.optgraph_id]  # 获取样本所属的权重（以该权重为中心）
            angle_center = np.arctan2(weight_center[1], weight_center[0])  # 计算该权重的角度
            angle_bound = [angle_center - np.pi / 4., angle_center + np.pi / 4.]  # 定义候选权重的范围 （正负45度）
            test_weights = []  # 存储候选权重
            for i in range(num_weights):  # 对于每个候选权重
                angle = angle_bound[0] + (angle_bound[1] - angle_bound[0]) / (num_weights - 1) * i  # 计算候选权重的角度 （在范围内均匀取值）
                weight = np.array([np.cos(angle), np.sin(angle)])  # 计算候选权重的向量
                if weight[0] >= -1e-7 and weight[1] >= -1e-7:  # 判断候选权重是否合法
                    duplicated = False  # 初始化为未重复
                    for succ in opt_graph.succ[sample.optgraph_id]:  # 对于样本后继的每个任务 # discard duplicate tasks
                        w = deepcopy(opt_graph.weights[succ])
                        w = w / np.linalg.norm(w)  # 对后继任务的权重进行归一化
                        if np.linalg.norm(w - weight) < 1e-3:  # 如果候选权重与后继任务的权重相近
                            duplicated = True  # 则说明候选权重重复
                            break
                    if not duplicated:  # 如果候选权重未重复
                        test_weights.append(weight)  # 则将其添加到候选权重列表中
            if len(test_weights) > 0:  # 如果候选权重列表不为空
                # 使用双曲线预测模型得到一个预测结果
                results = predict_hyperbolic(args, opt_graph, sample.optgraph_id, test_weights)
                for i in range(len(test_weights)):
                    candidates.append({'sample': sample, 'weight': test_weights[i], \
                        'prediction': results['predictions'][i]})  # 将候选策略在所有候选权重上的预测结果添加到候选列表中

        ### Optimization ###
            
        # initialize virtual ep as current ep
        virtual_ep_objs_batch = []  # 存储虚拟前沿
        for i in range(len(ep.sample_batch)):
            virtual_ep_objs_batch.append(deepcopy(ep.sample_batch[i].objs))

        mask = np.ones(len(candidates), dtype = bool)    # 初始化掩码为全一数组

        predicted_offspring_objs = []  # 存储预测的后代目标值
        elite_batch, scalarization_batch = [], []    # 存储精英样本和对应的标量化模板

        # greedy algorithm for knapsack problem
        alpha = args.sparsity  # 设置稀疏度的权重

        for _ in range(N):  # N是需要选出的任务数目。下面的循化中，每次循环都选出一个最佳的任务。
            hv = self.evaluate_hv(candidates, mask, virtual_ep_objs_batch)
            # 注意这里的hv是包含i个元素，是把candidate中的预测结果依次插入现有非支配集中，计算对应的hv
            sparsity = self.evaluate_sparsity(candidates, mask, virtual_ep_objs_batch)
            # sparsity同理

            # select the one with max dhv - alpha * dsparsity  选择具有最大 dhv - alpha * dsparsity 的候选者
            max_metrics, best_id = -np.inf, -1
            for i in range(len(candidates)):
                if mask[i]:
                    if hv[i] - alpha * sparsity[i] > max_metrics:  # 通过对HV和稀疏度加权，找到一个具有最高得分的候选解。减号是因为sparsity越小越好
                        max_metrics, best_id = hv[i] - alpha * sparsity[i], i

            if best_id == -1:
                print('Too few candidates')
                break

            # 将这个候选解加入精英集合 elite_batch 中
            elite_batch.append(candidates[best_id]['sample'])
            scalarization = deepcopy(scalarization_template)
            scalarization.update_weights(candidates[best_id]['weight'] / np.sum(candidates[best_id]['weight']))
            scalarization_batch.append(scalarization)
            mask[best_id] = False  # 将标记数组 mask 中与选定解相应的索引位置更新为 False。这个已经选出来的解在下次循环中不再考虑。

            # update virtual_ep_objs_batch
            ''' 将综合指标最大的解的预测目标函数值添加到虚拟前沿集合 virtual_ep_objs_batch 中，并更新虚拟前沿集合。
                这里的 get_ep_indices 函数用于计算当前虚拟前沿集合中的所有非支配解的索引。'''
            predicted_new_objs = [deepcopy(candidates[best_id]['prediction'])]
            new_objs_batch = np.array(virtual_ep_objs_batch + predicted_new_objs)
            virtual_ep_objs_batch = new_objs_batch[get_ep_indices(new_objs_batch)].tolist()

            # 将综合指标最大的解的预测目标函数值添加到 predicted_offspring_objs 中
            predicted_offspring_objs.extend(predicted_new_objs)

        return elite_batch, scalarization_batch, predicted_offspring_objs

    '''
    select the task by a random strategy.
    '''
    def random_selection(self, args, scalarization_template, ep, stage):
        if stage ==1:
            num_tasks = args.num_tasks
        else:
            num_tasks = int(args.num_tasks/2)

        elite_batch, scalarization_batch = [], []
        weights_all = []
        weights_all_t = []
        objs_all = []
        objs_all_t = []
        dist = []
        objs_ep = []
        weights_ep = []

        for k in range(len(ep.sample_batch)):
            objs_ep.append(torch.Tensor(ep.sample_batch[k].objs))
            weights_ep.append(ep.sample_batch[k].weight)
        objs_ep_t = torch.stack(deepcopy(objs_ep))  # ep内所有策略的评估目标值
        weights_ep_t = torch.stack(deepcopy(weights_ep))
        objs_ep_t = torch.nn.functional.normalize(objs_ep_t, p=2, dim=1)  # 归一化
        weights_ep_t = torch.nn.functional.normalize(weights_ep_t, p=2, dim=1)



        for i in range(len(self.sample_batch)):
            weights_all.append(self.sample_batch[i].weight)
            objs_all.append(torch.Tensor(self.sample_batch[i].objs))
        weights_all_t = torch.stack(deepcopy(weights_all))  # 种群内所有的权重
        objs_all_t = torch.stack(deepcopy(objs_all))  # 种群内所有的权重
        weights_all_t = torch.nn.functional.normalize(weights_all_t, p=2, dim=1)
        objs_all_t = torch.nn.functional.normalize(objs_all_t, p=2, dim=1)  # 归一化

        # weights_tan = weights_all_t[:, 0] / (weights_all_t[:, 1] + 1e-8)
        weights_tan = objs_all_t[:, 0] / (objs_all_t[:, 1] + 1e-8)
        weights_radian = torch.atan(weights_tan)
        weights_degree = torch.rad2deg(weights_radian)  # 计算这些权重的角度

        min_degree = torch.min(weights_degree)
        max_degree = torch.max(weights_degree)
        delta_degree = (max_degree-min_degree)/num_tasks

        for xx in range(num_tasks):
            dist = []  # 重置dist
            candidate_obj = []
            # 从给定的角度范围内选择一个任务
            # 定义最小值和最大值
            min_val = min_degree + xx*delta_degree
            max_val = min_degree + (xx+1)*delta_degree

            # 查找张量中值在min_val和max_val之间的元素的索引
            mask = (weights_degree >= min_val) & (weights_degree <= max_val)

            # 从特定范围的张量中随机选择一个元素的索引
            if ~(torch.logical_not(mask).all()):  # torch.logical_not把所有bool元素取反；torch.all()判断元素是否全为True
                candidate_idx = torch.where(mask)
                candidate_idx = torch.stack(candidate_idx).squeeze()  # 上一步生成的是一个元组，转为tensor
                if candidate_idx.dim() > 0:  # 至少有两个
                    if np.random.rand() < 0.5:  # 以80%的概率
                        for k in range(len(candidate_idx)):
                            ind = candidate_idx[k]
                            candidate_obj.append(torch.Tensor(self.sample_batch[ind].objs))
                        candidate_obj = torch.stack(candidate_obj)
                        candidate_norm = torch.norm(candidate_obj, dim=1)  # 计算目标向量的长度
                        half_length = math.ceil(len(candidate_idx) / 2)
                        values, ind_1 = torch.topk(candidate_norm, k=half_length, largest=True)
                        elite_idx = candidate_idx[ind_1[np.random.randint(0, half_length)]]  # np.random.randint生成的随机整数的取值区间是前闭后开区间
                    else:
                        elite_idx = candidate_idx[np.random.randint(0, len(candidate_idx))]  # np.random.randint生成的随机整数的取值区间是前闭后开区间
                else:
                    elite_idx = candidate_idx.item()
            else:
                elite_idx = np.random.choice(len(self.sample_batch))



            # weights = np.random.uniform(args.min_weight, args.max_weight, args.obj_num)
            weights = deepcopy(self.sample_batch[elite_idx].weight)
            objs = deepcopy(torch.Tensor(self.sample_batch[elite_idx].objs))
            weights = weights / torch.sum(weights)

            weights = torch.nn.functional.normalize(weights, dim=0)


            objs = torch.nn.functional.normalize(objs, dim=0)  # 归一化处理，以便计算距离。
            for ii in range(len(ep.sample_batch)):
                dist.append(torch.dist(objs, objs_ep_t[ii, :]))  # 当前目标点与ep中所有点的欧式距离
            dist = torch.stack(dist)

            # 查找dist中最小的k个元素的 值和索引。
            values, indices = torch.topk(dist, k=2, largest=False)
            weights_other = deepcopy(weights_ep_t[indices])
            weights_other = weights_other / torch.sum(weights_other, dim=1)

            self.sample_batch[elite_idx].neighbor_weight = weights_other


            elite_batch.append(self.sample_batch[elite_idx])


            scalarization = deepcopy(scalarization_template)
            weights = weights / torch.sum(weights)
            scalarization.update_weights(weights)
            scalarization_batch.append(scalarization)

            # elite_idx = np.random.choice(len(self.sample_batch))
            # elite_batch.append(self.sample_batch[elite_idx])
            # weights = np.random.uniform(args.min_weight, args.max_weight, args.obj_num)
            # weights = weights / np.sum(weights)
            # scalarization = deepcopy(scalarization_template)
            # scalarization.update_weights(weights)
            # scalarization_batch.append(scalarization)
        return elite_batch, scalarization_batch

    def greedy_selection(self, args, scalarization_template, ep, stage):
        if stage == 1:
            num_tasks = args.num_tasks
        else:
            num_tasks = int(args.num_tasks / 2) - 1

        elite_batch, scalarization_batch = [], []
        weights_all = []
        objs_all = []

        for i in range(len(self.sample_batch)):
            weights_all.append(self.sample_batch[i].weight)
            objs_all.append(torch.Tensor(self.sample_batch[i].objs))
        weights_all_t = torch.stack(deepcopy(weights_all))  # 种群内所有的权重
        objs_all_t = torch.stack(deepcopy(objs_all))  # 种群内所有的权重
        weights_all_t = torch.nn.functional.normalize(weights_all_t, p=2, dim=1)
        objs_all_t = torch.nn.functional.normalize(objs_all_t, p=2, dim=1)  # 归一化

        # weights_tan = weights_all_t[:, 0] / (weights_all_t[:, 1] + 1e-8)
        weights_tan = objs_all_t[:, 0] / (objs_all_t[:, 1] + 1e-8)
        weights_radian = torch.atan(weights_tan)
        weights_degree = torch.rad2deg(weights_radian)  # 计算这些权重的角度

        # min_degree = torch.min(weights_degree)
        # max_degree = torch.max(weights_degree)
        min_degree = 20
        max_degree = 80
        delta_degree = (max_degree - min_degree) / num_tasks
        min_val_0 = 20
        max_val_0 = 80

        for xx in range(num_tasks):
            dist = []  # 重置dist
            value_batch = []
            candidate_obj = []
            # 从给定的角度范围内选择一个任务
            # 定义最小值和最大值
            min_val = min_degree + xx * delta_degree
            max_val = min_degree + (xx + 1) * delta_degree

            # 查找张量中值在min_val和max_val之间的元素的索引
            mask = (weights_degree >= min_val) & (weights_degree <= max_val)

            # 从特定范围的张量中选择一个元素的索引
            if ~(torch.logical_not(mask).all()):  # torch.logical_not把所有bool元素取反；torch.all()判断元素是否全为True
                candidate_idx = torch.where(mask)
                candidate_idx = torch.stack(candidate_idx).squeeze()  # 上一步生成的是一个元组，转为tensor
                max_val_0 = max_val
                min_val_0 = min_val
                if candidate_idx.dim() > 0:  # 至少有两个
                    for idx in candidate_idx:
                        sample = deepcopy(self.sample_batch[idx])
                        scalarization = deepcopy(scalarization_template)
                        weights = deepcopy(sample.weight)
                        weights = weights / torch.sum(weights)
                        scalarization.update_weights(weights)
                        value = scalarization.evaluate(torch.Tensor(sample.objs))
                        value_batch.append(value)
                    value_batch = torch.stack(value_batch)
                    half_length = math.ceil(len(value_batch) / 3)
                    values, indices = torch.topk(value_batch, k=half_length,
                                                 largest=True)  # select the largest half of the elements
                    elite_idx = candidate_idx[indices[np.random.randint(0, half_length)]]  # choose one randomly
                else:
                    elite_idx = candidate_idx.item()

                elite_batch.append(self.sample_batch[elite_idx])
                scalarization = deepcopy(scalarization_template)
                weights = deepcopy(self.sample_batch[elite_idx].weight)
                weights = weights / torch.sum(weights)
                scalarization.update_weights(weights)
                scalarization_batch.append(scalarization)

            else:
                # elite_idx = np.random.choice(len(self.sample_batch))
                mask = (weights_degree >= min_val_0) & (weights_degree <= max_val_0)  # use last max_val and min_val
                candidate_idx = torch.where(mask)
                candidate_idx = torch.stack(candidate_idx).squeeze()  # 上一步生成的是一个元组，转为tensor
                if candidate_idx.dim() > 0:  # 至少有两个
                    for idx in candidate_idx:
                        sample = deepcopy(self.sample_batch[idx])
                        scalarization = deepcopy(scalarization_template)
                        weights = deepcopy(sample.weight)
                        weights = weights / torch.sum(weights)
                        scalarization.update_weights(weights)
                        value = scalarization.evaluate(torch.Tensor(sample.objs))
                        value_batch.append(value)
                    value_batch = torch.stack(value_batch)
                    half_length = math.ceil(len(value_batch) / 3)
                    values, indices = torch.topk(value_batch, k=half_length,
                                                 largest=True)  # select the largest half of the elements
                    elite_idx = candidate_idx[indices[np.random.randint(0, half_length)]]  # choose one randomly
                else:
                    elite_idx = candidate_idx.item()

                elite_batch.append(self.sample_batch[elite_idx])
                scalarization = deepcopy(scalarization_template)
                weights = deepcopy(self.sample_batch[elite_idx].weight)
                weights = weights / torch.sum(weights)
                delta_weight = 0.1
                weights[0] += delta_weight
                weights[1] -= delta_weight
                if weights[1] < 0:
                    weights = torch.Tensor([0.8, 0.2])
                scalarization.update_weights(weights)
                scalarization_batch.append(scalarization)

            # elite_batch.append(self.sample_batch[elite_idx])
            #
            # scalarization = deepcopy(scalarization_template)
            # weights = deepcopy(self.sample_batch[elite_idx].weight)
            # weights = weights / torch.sum(weights)
            # scalarization.update_weights(weights)
            # scalarization_batch.append(scalarization)
        return elite_batch, scalarization_batch

        # for sample in self.sample_batch:
        #     scalarization = deepcopy(scalarization_template)
        #     weights = deepcopy(sample.weight)
        #     weights = weights / torch.sum(weights)
        #     scalarization.update_weights(weights)
        #     value = scalarization.evaluate(torch.Tensor(sample.objs))
        #     value_batch.append(value)
        # value_batch = torch.stack(value_batch)
        # values, indices = torch.topk(value_batch, k=num_tasks, largest=True)
        # for idx in indices:
        #     elite_batch.append(self.sample_batch[idx])
        #
        #     scalarization = deepcopy(scalarization_template)
        #     weights = deepcopy(self.sample_batch[idx].weight)
        #     weights = weights / torch.sum(weights)
        #     scalarization.update_weights(weights)
        #     scalarization_batch.append(scalarization)
        #
        # return elite_batch, scalarization_batch



    def pfa_selection(self, args, scalarization_template, ep, stage):

        if stage == 1:
            num_tasks = args.num_tasks
        else:
            num_tasks = int(args.num_tasks / 2)+1

        elite_batch, scalarization_batch = [], []

        dist = []
        objs_ep = []
        weights_ep = []
        all_dist_max = []
        all_ind_max = []
        for k in range(len(ep.sample_batch)):
            obj_k = torch.Tensor(ep.sample_batch[k].objs)
            ind_1 = max(0, k-3)
            ind_2 = min(len(ep.sample_batch), k+3)
            ind_max = -1
            dist_max = 0
            for kk in range(ind_1, ind_2):
                if ~(kk==k):
                    neighbor = torch.Tensor(ep.sample_batch[kk].objs)
                    if torch.dist(obj_k, neighbor) > dist_max:
                        dist_max = torch.dist(obj_k, neighbor)
                        ind_max = kk
            all_dist_max.append(dist_max)
            all_ind_max.append(ind_max)
        all_dist_max = torch.stack(all_dist_max)

        values, elite_ind = torch.topk(all_dist_max, k=min(num_tasks-1, len(ep.sample_batch)), largest=True)
        neighbor_ind = []
        for iii in range(len(elite_ind)):
            neighbor_ind.append(all_ind_max[elite_ind[iii]])


        for i in range(len(elite_ind)):
            e_ind = elite_ind[i]
            elite_batch.append(ep.sample_batch[e_ind])

            scalarization = deepcopy(scalarization_template)
            obj_elite = torch.nn.functional.normalize(torch.Tensor(ep.sample_batch[e_ind].objs), dim=0)
            # obj_elite = torch.nn.functional.normalize(torch.Tensor(deepcopy(ep.sample_batch[e_ind].weight)), dim=0)
            obj_neighbor = torch.nn.functional.normalize(torch.Tensor(ep.sample_batch[neighbor_ind[i]].objs), dim=0)
            # obj_neighbor = torch.nn.functional.normalize(torch.Tensor(deepcopy(ep.sample_batch[neighbor_ind[i]].weight)), dim=0)
            direction = torch.nn.functional.normalize((obj_neighbor - obj_elite), dim=0)

            # weight_origin = torch.nn.functional.normalize(torch.Tensor(deepcopy(ep.sample_batch[e_ind].weight)), dim=0)
            # weight = (weight_origin + direction) / 2
            # weight = weight / torch.sum(weight)
            # if weight[0]<0:
            #     weight = torch.Tensor([0.0, 1.0])
            # if weight[1]<0:
            #     weight = torch.Tensor([0.8, 0.2])
            if direction[0] < 0:
                weight = torch.Tensor([0.0, 1.0])
            elif direction[0] >= 0:
                weight = torch.Tensor([0.8, 0.2])

            scalarization.update_weights(weight)
            scalarization_batch.append(scalarization)

        # 额外加入最后一个
        elite_last = ep.sample_batch[-1]
        elite_batch.append(elite_last)
        scalarization = deepcopy(scalarization_template)
        weight = torch.Tensor([0.8, 0.2])
        scalarization.update_weights(weight)
        scalarization_batch.append(scalarization)

        return elite_batch, scalarization_batch