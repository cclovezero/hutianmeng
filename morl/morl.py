import os, sys

import environments

# import python packages
import time
from copy import deepcopy

# import third-party packages
import numpy as np
import torch
import torch.optim as optim
from multiprocessing import Process, Queue, Event
import pickle

# import our packages
from scalarization_methods import WeightedSumScalarization
from sample import Sample
from task import Task
from ep import EP
from population_2d import Population as Population2d
from population_3d import Population as Population3d
from opt_graph import OptGraph
from utils import generate_weights_batch_dfs, print_info
from warm_up import initialize_warm_up_batch
from mopg import MOPG_worker

def run(args):

    # --------------------> Preparation <-------------------- #
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # build a scalarization template
    # 用于标量化奖励的函数
    scalarization_template = WeightedSumScalarization(num_objs = args.obj_num, weights = np.ones(args.obj_num) / args.obj_num)

    total_num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes  # //是整除运算
    # num_steps = 2048 ， num_processes = 4。 ### 要看一下这个的具体含义
    # total_num_updates是总的网络更新次数？
    '''
    在PPO算法中，num_steps和num_process是用于并行采样的两个重要参数。
    num_steps表示在一次采样中，智能体与环境交互的步数。
    num_steps越大，每次采样得到的数据越多，从而可以提高训练效率和策略的稳定性。但是，num_steps过大也会导致训练过程变得非常缓慢。
    
    num_process表示用于并行采样的进程数。在PPO算法中，通常采用多进程的方式来加速采样过程。
    通过多进程的方式，可以同时在多个环境中运行多个智能体，从而可以更快地收集训练数据。
    num_process越大，采样速度越快，但也会带来一定的系统负载和通信开销。
    '''

    start_time = time.time()

    # initialize ep and population and opt_graph
    ep = EP()  # EP是存储所有非支配解的外部种群
    if args.obj_num == 2:
        population = Population2d(args)  # 用于维护种群
    elif args.obj_num > 2:
        population = Population3d(args)
    else:
        raise NotImplementedError
    opt_graph = OptGraph()  # OptGraph是用于存储优化历史的，这部分用于预测引导方法。详见OptGraph()函数
    ep_warmup = deepcopy(population)  # for warmup

    # Construct tasks for warm up
    elite_batch, scalarization_batch = initialize_warm_up_batch(args, device)  # 创建了六个初始化的actor-critic策略。详见该函数。
    # rl_num_updates = args.warmup_iter  # warmup_iter： 80代
    rl_num_updates = args.update_iter
    # rl_num_updates = 20  # 临时
    for sample, scalarization in zip(elite_batch, scalarization_batch):  # OptGraph是用于存储优化历史的，这部分用于预测引导方法。
        sample.optgraph_id = opt_graph.insert(deepcopy(scalarization.weights), deepcopy(sample.objs), -1) # 详见OptGraph()函数

    episode = 0
    iteration = 0
    while iteration < total_num_updates:
        if episode == 0:
            print_info('\n------------------------------- Warm-up Stage -------------------------------')
        else:
            print_info('\n-------------------- Evolutionary Stage: Generation {:3} --------------------'.format(episode))

        episode += 1

        offspring_batch = np.array([])  # 后代batch

        # --------------------> RL Optimization <-------------------- #
        # compose task for each elite
        task_batch = []
        for elite, scalarization in \
                zip(elite_batch, scalarization_batch):  # elite_batch中存储了当前的（精英？）种群
            task_batch.append(Task(elite, scalarization))  # each task is a (policy, weight) pair
            # Task函数只是简单地打包了sample和scalarization
            # 所以这段就是把每个精英策略和相应的权重打包成一个任务。
        '''
        实际上这里的elite（精英）就是后面的任务选择算法选出的任务，只是在第一次循环中它是随机初始化的策略。
        '''

        if iteration < args.warmup_iter:
            rl_num_updates = 100
        elif iteration <= total_num_updates / 5 * 2:
            rl_num_updates = args.update_iter * 2  # 20
        # elif iteration < total_num_updates / 5 * 4:
        #     rl_num_updates = args.update_iter  # 20
        else:
            rl_num_updates = args.update_iter

        # run MOPG for each task in parallel
        # processes是存储每个任务执行进程的列表
        processes = []
        # results_queue是进程返回结果存储的队列
        results_queue = Queue()
        # done_event是事件对象，用于标识任务完成
        done_event = Event()

        # 创建并启动每个任务对应的进程
        # 注意！！ 这里的每个进程内部就包含了rl_num_updates次的更新。在大循环的第一次，rl_num_updates=80，为warmup阶段。后面每次都变为20。
        for task_id, task in enumerate(task_batch):
            p = Process(target = MOPG_worker, \
                args = (args, task_id, task, device, iteration, rl_num_updates, start_time, results_queue, done_event))
            p.start()
            processes.append(p)

        # collect MOPG results for offsprings and insert objs into objs buffer
        # all_offspring_batch是存储每个任务执行结果的列表
        all_offspring_batch = [[] for _ in range(len(processes))]
        cnt_done_workers = 0

        # 循环直到所有进程返回结果
        ''' 在Python的多进程编程中，queue是一个线程安全的队列，用于进程之间的数据传递。
            queue.get()函数用于从队列中获取一个元素，如果队列为空，则该函数会一直阻塞，直到有新的元素加入队列或者队列被关闭。    '''
        while cnt_done_workers < len(processes):
            rl_results = results_queue.get()  # queue.get()函数用于从队列中获取一个元素，如果队列为空，则该函数会一直阻塞
            task_id, offsprings = rl_results['task_id'], rl_results['offspring_batch']
            for sample in offsprings:
                # 将每个任务执行结果添加到all_offspring_batch对应的位置中
                all_offspring_batch[task_id].append(Sample.copy_from(sample))
            # 如果进程返回了“done”，则说明该进程已完成
            if rl_results['done']:
                cnt_done_workers += 1

        # put all intermidiate policies into all_sample_batch for EP update
        # all_sample_batch是存储每个中间策略的列表
        all_sample_batch = []
        # store the last policy for each optimization weight for RA
        # last_offspring_batch存储每个任务的最后一个子代策略
        last_offspring_batch = [None] * len(processes)

        # only the policies with iteration % update_iter = 0 are inserted into offspring_batch for population update
        # after warm-up stage, it's equivalent to the last_offspring_batch

        # offspring_batch是存储每个任务的子代策略列表
        # 仅将iteration % update_iter（20） = 0的策略插入到offspring_batch中以进行种群更新。在热身阶段之后，它等效于last_offspring_batch
        # 具体条件是：i + 1是args.update_iter的倍数
        offspring_batch = []
        for task_id in range(len(processes)):
            ''' 对于进程列表 processes 中的每个任务，将其后代存储在 all_offspring_batch 中。
                然后获取该任务的当前节点 ID prev_node_id，并从 task_batch 中获取该任务的权重，即 scalarization.weights，并通过 deepcopy() 复制它们。
                最后，将其转换为 NumPy 数组以供之后使用。 '''
            offsprings = all_offspring_batch[task_id]
            prev_node_id = task_batch[task_id].sample.optgraph_id
            opt_weights = deepcopy(task_batch[task_id].scalarization.weights).detach().numpy()
            for i, sample in enumerate(offsprings):
                ''' 对于每个后代样本，将其添加到 all_sample_batch 列表中。
                    如果已经生成了 args.update_iter 个后代，则最终的后代插入到优化图中。
                    此时，优化图的当前节点是 prev_node_id，新节点将连接到当前节点的后面。
                    插入操作会返回新节点的 ID，因此将此值赋给 sample.optgraph_id。最后，将该后代添加到 offspring_batch 中。   '''
                # 将所有中间策略添加到all_sample_batch列表中
                all_sample_batch.append(sample)
                # 判断当前子代是否满足特定条件，是则将该策略添加到offspring_batch和优化图optgraph中
                if ((i + 1) % args.update_iter) == 0 or ((i + 1) % rl_num_updates == 0):
                    prev_node_id = opt_graph.insert(opt_weights, deepcopy(sample.objs), prev_node_id)
                    sample.optgraph_id = prev_node_id
                    offspring_batch.append(sample)
                    for ii in range(len(all_sample_batch)):
                        all_sample_batch[ii].optgraph_id = prev_node_id
            # 将每个任务的最后一个子代策略添加到last_offspring_batch中
            last_offspring_batch[task_id] = offsprings[-1]

        print_info('Updated Tasks:')  # 打印出选择的任务
        for i in range(len(last_offspring_batch)):
            print_info('objs = {}, prac_weight = {}'.format(last_offspring_batch[i].objs, last_offspring_batch[i].prac_weight))

        # 标识任务已完成
        done_event.set()

        # -----------------------> Update EP <----------------------- #
        # update EP and population
        ep.update(all_sample_batch)  # 用更新过程中的所有中间策略更新外部种群EP（非支配集）。
        population.update(offspring_batch)  # 每个任务的更新过程的最后一个策略被加入offspring_batch，用这个去更新当前种群。
        # if iteration <= args.warmup_iter:
        #     ep_warmup.update(ep.sample_batch.tolist())

        # ------------------- > Task Selection <--------------------- #
        # 这是任务选择有关的算法
        if iteration >= args.warmup_iter-rl_num_updates:
        # if iteration >= 0:
            if iteration < total_num_updates/4:
            # if iteration < 0:
            #     elite_batch, scalarization_batch = population.random_selection(args, scalarization_template, ep, stage=1)
                elite_batch, scalarization_batch = population.greedy_selection(args, scalarization_template, ep, stage=1)
            else:
                # elite_batch, scalarization_batch = population.random_selection(args, scalarization_template, ep, stage=2)
                elite_batch, scalarization_batch = population.greedy_selection(args, scalarization_template, ep, stage=2)
                elite_batch_pfa, scalarization_batch_pfa = population.pfa_selection(args, scalarization_template, ep, stage=2)
                elite_batch.extend(elite_batch_pfa)
                scalarization_batch.extend(scalarization_batch_pfa)
            if iteration < (total_num_updates - rl_num_updates):
            # if iteration < total_num_updates / 5 * 4:
                for i, sample in enumerate(elite_batch):
                    sample.actor_critic.base.critic = deepcopy(offspring_batch[i].actor_critic.base.critic)  # share critic???????
        else:
            elite_batch = last_offspring_batch
            # elite_batch, scalarization_batch = ep_warmup.random_selection(args, scalarization_template, ep, stage=1)
        # elite_batch, scalarization_batch = ep.random_selection(args, scalarization_template)
        # if args.selection_method == 'moead':
        #     elite_batch, scalarization_batch = [], []
        #     weights_batch = []
        #     generate_weights_batch_dfs(0, args.obj_num, args.min_weight, args.max_weight, args.delta_weight, [], weights_batch)
        #     for weights in weights_batch:
        #         scalarization = deepcopy(scalarization_template)
        #         scalarization.update_weights(weights)
        #         scalarization_batch.append(scalarization)
        #         best_sample, best_value = None, -np.inf
        #         for sample in population.sample_batch:
        #             value = scalarization.evaluate(torch.Tensor(sample.objs))
        #             if value > best_value:
        #                 best_sample, best_value = sample, value
        #         elite_batch.append(best_sample)
        #
        # elif args.selection_method == 'prediction-guided':  # 这里是预测引导算法
        #     elite_batch, scalarization_batch, predicted_offspring_objs = population.prediction_guided_selection(args, iteration, ep, opt_graph, scalarization_template)
        #     # 使用引导算法 得到下次的任务
        # elif args.selection_method == 'random':
        #     elite_batch, scalarization_batch = population.random_selection(args, scalarization_template)
        # elif args.selection_method == 'ra':
        #     elite_batch = last_offspring_batch
        #     scalarization_batch = []
        #     weights_batch = []
        #     generate_weights_batch_dfs(0, args.obj_num, args.min_weight, args.max_weight, args.delta_weight, [], weights_batch)
        #     for weights in weights_batch:
        #         scalarization = deepcopy(scalarization_template)
        #         scalarization.update_weights(weights)
        #         scalarization_batch.append(scalarization)
        # elif args.selection_method == 'pfa':
        #     if args.obj_num > 2:
        #         raise NotImplementedError
        #     elite_batch = last_offspring_batch
        #     scalarization_batch = []
        #     delta_ratio = (iteration + rl_num_updates + args.update_iter - args.warmup_iter) / (total_num_updates - args.warmup_iter)
        #     delta_ratio = np.clip(delta_ratio, 0.0, 1.0)
        #     for i in np.arange(args.min_weight, args.max_weight + 0.5 * args.delta_weight, args.delta_weight):
        #         w = np.clip(i + delta_ratio * args.delta_weight, args.min_weight, args.max_weight)
        #         weights = np.array([abs(w), abs(1.0 - w)])
        #         scalarization = deepcopy(scalarization_template)
        #         scalarization.update_weights(weights)
        #         scalarization_batch.append(scalarization)
        # else:
        #     raise NotImplementedError


        print_info('Selected Tasks:')  # 打印出选择的任务
        for i in range(len(elite_batch)):
            print_info('objs = {}, weight = {}'.format(elite_batch[i].objs, scalarization_batch[i].weights))

        iteration = min(iteration + rl_num_updates, total_num_updates)
        # iteration保存的是迭代次数。  total_num_updates是设置的最大更新次数

        rl_num_updates = args.update_iter  # 大循环的第一次内部会执行完80代的warmup。
        # 然后rl_num_update = update_iter = 20，即每次大循环内部是20次迭代



        # ----------------------> Save Results <---------------------- #
        # save ep
        ep_dir = os.path.join(args.save_dir, str(iteration), 'ep')
        os.makedirs(ep_dir, exist_ok = True)
        with open(os.path.join(ep_dir, 'objs.txt'), 'w') as fp:
            for obj in ep.obj_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*obj))

        # save population
        population_dir = os.path.join(args.save_dir, str(iteration), 'population')
        os.makedirs(population_dir, exist_ok = True)
        with open(os.path.join(population_dir, 'objs.txt'), 'w') as fp:
            for sample in population.sample_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(sample.objs)))
        # save optgraph and node id for each sample in population
        with open(os.path.join(population_dir, 'optgraph.txt'), 'w') as fp:
            fp.write('{}\n'.format(len(opt_graph.objs)))
            for i in range(len(opt_graph.objs)):
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + ';{:5f}' + (args.obj_num - 1) * ',{:5f}' + ';{}\n').format(*(opt_graph.weights[i]), *(opt_graph.objs[i]), opt_graph.prev[i]))
            fp.write('{}\n'.format(len(population.sample_batch)))
            for sample in population.sample_batch:
                fp.write('{}\n'.format(sample.optgraph_id))

        # save elites
        elite_dir = os.path.join(args.save_dir, str(iteration), 'elites')
        os.makedirs(elite_dir, exist_ok = True)
        with open(os.path.join(elite_dir, 'elites.txt'), 'w') as fp:
            for elite in elite_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(elite.objs)))
        with open(os.path.join(elite_dir, 'weights.txt'), 'w') as fp:
            for scalarization in scalarization_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(scalarization.weights)))
        # 修改
        # if args.selection_method == 'prediction-guided':
        #     with open(os.path.join(elite_dir, 'predictions.txt'), 'w') as fp:
        #         for objs in predicted_offspring_objs:
        #             fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(objs)))
        with open(os.path.join(elite_dir, 'offsprings.txt'), 'w') as fp:
            for i in range(len(all_offspring_batch)):
                for j in range(len(all_offspring_batch[i])):
                    fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(all_offspring_batch[i][j].objs)))







    # ----------------------> Save Final Model <----------------------

    os.makedirs(os.path.join(args.save_dir, 'final'), exist_ok = True)

    # save ep policies & env_params
    for i, sample in enumerate(ep.sample_batch):
        torch.save(sample.actor_critic.state_dict(), os.path.join(args.save_dir, 'final', 'EP_policy_{}.pt'.format(i)))
        with open(os.path.join(args.save_dir, 'final', 'EP_env_params_{}.pkl'.format(i)), 'wb') as fp:
            pickle.dump(sample.env_params, fp)

    # save all ep objectives
    with open(os.path.join(args.save_dir, 'final', 'objs.txt'), 'w') as fp:
        for i, obj in enumerate(ep.obj_batch):
            fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(obj)))

    # save all ep env_params
    if args.obj_rms:
        with open(os.path.join(args.save_dir, 'final', 'env_params.txt'), 'w') as fp:
            for sample in ep.sample_batch:
                fp.write('obj_rms: mean: {} var: {}\n'.format(sample.env_params['obj_rms'].mean, sample.env_params['obj_rms'].var))
