import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

from copy import deepcopy

import gym
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.envs import make_vec_envs, make_env
from a2c_ppo_acktr.model import Policy

from sample import Sample
from utils import generate_weights_batch_dfs
from scalarization_methods import WeightedSumScalarization
from mopg import evaluation

'''
initialize_warm_up_batch: method to generate tasks in the warm-up stage.
Each task is a pair of an initial random policy and an evenly distributed optimization weight.
The optimization weight is represented by a weighted-sum scalarization function.
'''
def initialize_warm_up_batch(args, device):
    # using evenly distributed weights for warm-up stage
    weights_batch = []
    generate_weights_batch_dfs(0, args.obj_num, args.min_weight, args.max_weight, args.delta_weight, [], weights_batch)
    # generate_weights_batch_dfs(0, args.obj_num, 0.5, args.max_weight, 0.1, [], weights_batch)
    # delta-weight = 0.2， 以0.2为间隔，共采样6组权重，放入weights_batch。形成6个任务
    sample_batch = []
    scalarization_batch = []

    temp_env = gym.make(args.env_name) # temp_env is only used for initialization

    for weights in weights_batch:  # 为每个权重（任务）创建actor、critic网络
        actor_critic = Policy(    # 创建actor、critic网络
            temp_env.observation_space.shape,
            temp_env.action_space,
            base_kwargs={'layernorm' : args.layernorm},
            obj_num=args.obj_num)

        actor_critic.to(device).double()

        if args.algo == 'ppo':
            agent = algo.PPO(
                actor_critic,  # 用刚创建的actor、critic网络，建立PPO代理
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=1e-5,
                max_grad_norm=args.max_grad_norm)
        else:
            # NOTE: other algorithms are not supported yet
            raise NotImplementedError
    
        envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes, \
                            gamma=args.gamma, log_dir=None, device=device, allow_early_resets=False, \
                            obj_rms = args.obj_rms, ob_rms = args.ob_rms)  # 创建环境
        env_params = {}
        env_params['ob_rms'] = deepcopy(envs.ob_rms) if envs.ob_rms is not None else None
        env_params['ret_rms'] = deepcopy(envs.ret_rms) if envs.ret_rms is not None else None
        env_params['obj_rms'] = deepcopy(envs.obj_rms) if envs.obj_rms is not None else None
        envs.close()
        '''
        ob_rms 如果为True，将会对观测数据进行均值方差归一化。
        同理，ret_rms是对return进行类似的处理。在这里，ret_rms对应的reward似乎没有使用，环境返回的reward是0。
        多个目标的reward是通过info里的obj返回的，因此，这里的obj_rms其实扮演了ret_rms的角色。
        
        在PPO算法中，running mean和running std主要用于数据归一化（data normalization）。
        数据归一化是深度强化学习中一个常用的技术，它可以使数据的分布更加稳定，从而更容易训练出更好的模型。
        具体来说，running mean和running std会对每个时间步的观测值（observation）进行归一化，使得它们的均值为0，标准差为1。
        这样可以将不同时间步的观测值之间的差异降到最小，从而更好地适应训练数据的分布。
        PPO算法中的running mean和running std是在训练过程中动态计算的，它们会随着训练数据的变化而更新。
        这样可以保证模型始终在适应最新的数据分布，从而更容易训练出更好的模型。
        '''

        scalarization = WeightedSumScalarization(num_objs = args.obj_num, weights = weights)
        # scalarization 后面应该会用于对多目标加权求和。在这里就存好特定的权重，后续可计算相应的加权标量值

        sample = Sample(env_params, actor_critic, agent, optgraph_id = -1)
        # 这里使用随机初始化的actor、critic网络，组成一个策略（样本）
        '''
        每个Sample都是一个策略，它包含了actor_critic, agent status和running mean std信息。
        算法可以选择任何一个样本来恢复其训练过程，或者通过这些信息用另一个优化方向进行训练。
        每个Sample都由一个唯一的optgraph_id索引。
        '''
        objs = evaluation(args, sample)
        # 对一个策略进行评估，返回在两个目标上取得的累计折扣奖励
        sample.objs = objs
        # 把相应评估结果也存储进sample里

        sample_batch.append(sample)
        scalarization_batch.append(scalarization)
    
    temp_env.close()

    return sample_batch, scalarization_batch  # 返回6个任务的sample、scalarization。