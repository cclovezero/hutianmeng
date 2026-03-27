import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

import numpy as np
from collections import deque
from copy import deepcopy
import time
import torch

import gym
# import a2c_ppo_acktr
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs, make_env
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from sample import Sample

'''
Evaluate a policy sample.
'''
def evaluation(args, sample):
    eval_env = gym.make(args.env_name)
    objs = np.zeros(args.obj_num)
    ob_rms = sample.env_params['ob_rms']
    policy = sample.actor_critic
    with torch.no_grad():
        for eval_id in range(args.eval_num):
            eval_env.seed(args.seed + eval_id)
            ob = eval_env.reset()
            done = False
            gamma = 1.0
            while not done:
                if args.ob_rms:  # 若为True
                    ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)  # 对观测数据进行均值方差归一化
                _, action, _, _ = policy.act(torch.Tensor(ob).unsqueeze(0), None, None, deterministic=True)
                ob, _, done, info = eval_env.step(action)
                objs += gamma * info['obj']
                if not args.raw:
                    gamma *= args.gamma
    eval_env.close()
    objs /= args.eval_num
    return objs

'''
define a MOPG Worker.
Input:
    args: the arguments include necessary ppo parameters.
    task_id: the task_id in all parallel executed tasks.
    device: torch device
    iteration: starting iteration number
    num_updates: number of rl iterations to run.
    start_time: starting time
    results_queue: multi-processing queue to pass the optimized policies back to main process.
    done_event: multi-processing event for process synchronization.
'''
def MOPG_worker(args, task_id, task, device, iteration, num_updates, start_time, results_queue, done_event):
    # 获取task的scalarization、env_params、actor_critic、agent
    scalarization = task.scalarization
    neighbor_weight = task.sample.neighbor_weight  # 修改
    env_params, actor_critic, agent = task.sample.env_params, task.sample.actor_critic, task.sample.agent

    # 格式化scalarization的权重
    weights_str = (args.obj_num * '_{:.3f}').format(*task.scalarization.weights)

    # make envs  # 创建多个并行的环境  num_processes=4
    envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes, \
                        gamma=args.gamma, log_dir=None, device=device, allow_early_resets=False, \
                        obj_rms=args.obj_rms, ob_rms = args.ob_rms)
    if env_params['ob_rms'] is not None:
        envs.venv.ob_rms = deepcopy(env_params['ob_rms'])
    if env_params['ret_rms'] is not None:
        envs.venv.ret_rms = deepcopy(env_params['ret_rms'])
    if env_params['obj_rms'] is not None:
        envs.venv.obj_rms = deepcopy(env_params['obj_rms'])

    # build rollouts data structure  # 创建rollouts数据结构
    rollouts = RolloutStorage(num_steps = args.num_steps, num_processes = args.num_processes,
                              obs_shape = envs.observation_space.shape, action_space = envs.action_space,
                              recurrent_hidden_state_size = actor_critic.recurrent_hidden_state_size, obj_num=args.obj_num)
    obs = envs.reset()  # 重置环境
    rollouts.obs[0].copy_(obs)  # 复制obs到rollouts.obs[0]
    rollouts.to(device)  # 将rollouts放到指定设备上

    episode_rewards = deque(maxlen=10)  # 保存最近10个episode的奖励
    episode_lens = deque(maxlen=10)  # 保存最近10个episode的步数
    episode_objs = deque(maxlen=10)  # 保存最近10个episode中各个代价的值  # for each cost component we care
    episode_obj = np.array([None] * args.num_processes)  # 保存各个进程当前episode中各个代价的和

    total_num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    offspring_batch = []

    start_iter, final_iter = iteration, min(iteration + num_updates, total_num_updates)
    if start_iter <= args.warmup_iter:
    # if start_iter <= total_num_updates/4:
        ratio = 0.8
    else:
        ratio = 0.2
    for j in range(start_iter, final_iter):
        torch.manual_seed(j)  # 设置随机数种子
        if args.use_linear_lr_decay:  # 如果使用线性学习率衰减，按照线性函数调整当前的学习率。
            # decrease learning rate linearly
            utils.update_linear_schedule( \
                agent.optimizer, j * ratio, \
                total_num_updates, args.lr)
        
        for step in range(args.num_steps):  # 循环args.num_steps次。=2048
            # Sample actions
            # 使用actor_critic模型进行前向推断，得到当前状态下的动作，价值函数，以及相关的中间变量。
            # 获取当前状态(obs)下的动作(action)、动作的对数概率(action_log_prob)、价值(value)和L个目标的信息(obj_tensor)。
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            
            obs, _, done, infos = envs.step(action)  # 使用action进行交互，获取新的状态obs，是否结束标志done，和其他信息infos。
            # 这个step转到 externals/baselines/baselines/common/vec_env/vec_env.py 的101行，目的是告诉所有并行进行的环境进行一次step
            # 最终生效的step()应该是environments/walker2d.py 中的step()函数（或者其他环境的文件）
            # 这里的第二个返回值本来是reward，但是不再使用。两个目标的reward存储在infos中。

            obj_tensor = torch.zeros([args.num_processes, args.obj_num])
            for idx, info in enumerate(infos):  # 根据infos中的信息获取obj_tensor，更新episode_obj和episode_rewards等信息。
                obj_tensor[idx] = torch.from_numpy(info['obj'])
                episode_obj[idx] = info['obj_raw'] if episode_obj[idx] is None else episode_obj[idx] + info['obj_raw']
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_lens.append(info['episode']['l'])
                    if episode_obj[idx] is not None:
                        episode_objs.append(episode_obj[idx])
                        episode_obj[idx] = None

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])  # 构造掩码矩阵masks，如果done为True，该位置的值为0，否则为1。
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])  # 构造坏掩码矩阵bad_masks，如果infos中包含'bad_transition'，该位置的值为0，否则为1。

            ''' 每个步骤都将观察(obs)、隐状态(recurrent_hidden_states)、动作(action)、动作的对数概率(action_log_prob)、
                价值(value)、物体信息(obj_tensor)、掩码(masks)和错误掩码(bad_masks)插入到rollouts缓存中。  '''
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, obj_tensor, masks, bad_masks)


        with torch.no_grad():  # 通过执行actor_critic模型来获取下一个状态的价值(next_value)。
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        # 使用价值(next_value)计算目标回报(return)。
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        obj_rms_var = envs.obj_rms.var if envs.obj_rms is not None else None

        # 调用agent.update()函数来更新智能体(agent)的参数，得到value_loss、action_loss和dist_entropy。
        # 修改
        # if (j > total_num_updates / 4) and (j == start_iter + num_updates / 4 * 3) and (task_id >= 3):
        #     weight_p = evaluation_grad(args, actor_critic, agent, rollouts, envs, scalarization)
        #     scalarization.update_weights(weight_p)
        # new change
        if (j > total_num_updates / 4) and (j == start_iter + num_updates / 4 * 3) and (3 <= task_id <= 7):
            weight_p = evaluation_grad(args, actor_critic, agent, rollouts, envs, scalarization)
            scalarization.update_weights(weight_p)
        if (j + 1) % args.update_iter == 1 and not (
                (j > total_num_updates / 4) and (3 <= task_id <= 7)) and j > args.warmup_iter:  # 第一代，先对整个策略重新计算梯度和权重
            weight_n = evaluation_grad(args, actor_critic, agent, rollouts, envs, scalarization)  # 修改final_weight
            scalarization.update_weights(weight_n)



        if j <= args.warmup_iter:
        # if j <= 20:
            value_loss, action_loss, dist_entropy, new_weights, prac_weight = agent.update(rollouts, scalarization, obj_rms_var, warmup=True, neighbor_weight=None)
        else:
            value_loss, action_loss, dist_entropy, new_weights, prac_weight = agent.update(rollouts, scalarization, obj_rms_var, warmup=False, neighbor_weight=neighbor_weight)

        # 调用rollouts.after_update()函数来清空rollouts缓存。
        rollouts.after_update()

        env_params = {}
        env_params['ob_rms'] = deepcopy(envs.ob_rms) if envs.ob_rms is not None else None
        env_params['ret_rms'] = deepcopy(envs.ret_rms) if envs.ret_rms is not None else None
        env_params['obj_rms'] = deepcopy(envs.obj_rms) if envs.obj_rms is not None else None

        # evaluate new sample
        # 调用evaluation()函数对新的策略进行评估，并将结果存储到offspring_batch数组中。
        sample = Sample(env_params, deepcopy(actor_critic), deepcopy(agent))
        objs = evaluation(args, sample)
        sample.objs = objs

        # final_weight = evaluation_grad(args, actor_critic, agent, rollouts, envs, scalarization)  # 修改final_weight
        # sample.weight = final_weight  # 修改  将计算出的weight也存储为sample的一部分
        sample.weight = new_weights   # 除非是最后一代，否则不再重新计算策略的梯度，直接以task的权重代替
        # 新修改 在最后不计算权重，改为在每代的第一次计算，写在前面
        # if (j +c 1) % args.update_iter == 0 or j == final_iter - 1:  # 最后一代，对整个策略重新计算梯度和权重
        #     final_weight = evaluation_grad(args, actor_critic, agent, rollouts, envs, scalarization)  # 修改final_weight
        #     sample.weight = final_weight  # 修改  将计算出的weight也存储为sample的一部分

        sample.prac_weight = prac_weight  # 修改
        offspring_batch.append(sample)

        # rl-log-interval是设置的记录（打印训练进程）间隔，为10
        if args.rl_log_interval > 0 and (j + 1) % args.rl_log_interval == 0 and len(episode_rewards) > 1:
            if task_id == 0:  # 仅当task_id = 0 时才打印。因为一个MOPG worker是对应于一个任务的，有六个任务在并行进行，没有必要重复打印6次。
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                end = time.time()
                print(
                    "[RL] Updates {}, num timesteps {}, FPS {}, time {:.2f} seconds"
                    .format(j + 1, total_num_steps,
                            int(total_num_steps / (end - start_time)),
                            end - start_time))

        # put results back every update_iter iterations, to avoid the multi-processing crash
        # 将offspring_batch数组存储到results_queue中。
        # update_iter = 20 。在热身阶段，一共要进行80次迭代；而在进化阶段，只进行20次迭代。
        # 另外，这里j是从start_iter开始增加的，直到final_iter。并不是从0开始。
        if (j + 1) % args.update_iter == 0 or j == final_iter - 1:
            offspring_batch = np.array(offspring_batch)
            results = {}
            results['task_id'] = task_id
            results['offspring_batch'] = offspring_batch
            if j == final_iter - 1:
                results['done'] = True
            else:
                results['done'] = False
            results_queue.put(results)
            offspring_batch = []

    envs.close()   
    
    done_event.wait()




def evaluation_grad(args, actor_critic, agent, rollouts, envs, scalarization):
    for step in range(args.num_steps):  # 循环args.num_steps次。=2048
        # Sample actions
        # 使用actor_critic模型进行前向推断，得到当前状态下的动作，价值函数，以及相关的中间变量。
        # 获取当前状态(obs)下的动作(action)、动作的对数概率(action_log_prob)、价值(value)和L个目标的信息(obj_tensor)。
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                rollouts.masks[step])

        obs, _, done, infos = envs.step(action)  # 使用action进行交互，获取新的状态obs，是否结束标志done，和其他信息infos。
        # 这个step转到 externals/baselines/baselines/common/vec_env/vec_env.py 的101行，目的是告诉所有并行进行的环境进行一次step
        # 最终生效的step()应该是environments/walker2d.py 中的step()函数（或者其他环境的文件）
        # 这里的第二个返回值本来是reward，但是不再使用。两个目标的reward存储在infos中。

        obj_tensor = torch.zeros([args.num_processes, args.obj_num])
        # for idx, info in enumerate(infos):  # 根据infos中的信息获取obj_tensor，更新episode_obj和episode_rewards等信息。
        #     obj_tensor[idx] = torch.from_numpy(info['obj'])
        #     episode_obj[idx] = info['obj_raw'] if episode_obj[idx] is None else episode_obj[idx] + info['obj_raw']
        #     if 'episode' in info.keys():
        #         episode_rewards.append(info['episode']['r'])
        #         episode_lens.append(info['episode']['l'])
        #         if episode_obj[idx] is not None:
        #             episode_objs.append(episode_obj[idx])
        #             episode_obj[idx] = None

        # If done then clean the history of observations.
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])  # 构造掩码矩阵masks，如果done为True，该位置的值为0，否则为1。
        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0]
             for info in infos])  # 构造坏掩码矩阵bad_masks，如果infos中包含'bad_transition'，该位置的值为0，否则为1。

        ''' 每个步骤都将观察(obs)、隐状态(recurrent_hidden_states)、动作(action)、动作的对数概率(action_log_prob)、
            价值(value)、物体信息(obj_tensor)、掩码(masks)和错误掩码(bad_masks)插入到rollouts缓存中。  '''
        rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, obj_tensor, masks, bad_masks)

    with torch.no_grad():  # 通过执行actor_critic模型来获取下一个状态的价值(next_value)。
        next_value = actor_critic.get_value(
            rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()

    # 使用价值(next_value)计算目标回报(return)。
    rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                             args.gae_lambda, args.use_proper_time_limits)

    obj_rms_var = envs.obj_rms.var if envs.obj_rms is not None else None

    final_weight = agent.evaluate_grad_1(rollouts, scalarization, obj_rms_var, warmup=False)

    # 调用rollouts.after_update()函数来清空rollouts缓存。
    rollouts.after_update()

    return final_weight
