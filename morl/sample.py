from copy import deepcopy
import torch.optim as optim

'''
Each Sample is a policy which contains the actor_critic, agent status and running mean std info.
The algorithm can pick any sample to resume its training process or train with another optimization direction
through those information.
Each Sample is indexed by a unique optgraph_id
每个样本都是一个策略，它包含了actor_critic, agent status和running mean std信息。
算法可以选择任何一个样本来恢复其训练过程，或者通过这些信息用另一个优化方向进行训练。
每个Sample都由一个唯一的optgraph_id索引。
'''
class Sample:
    def __init__(self, env_params, actor_critic, agent, objs = None, optgraph_id = None, weight = None, neighbor_weight = None, prac_weight = None):
        self.env_params = env_params
        self.actor_critic = actor_critic
        self.agent = agent
        self.link_policy_agent()
        self.objs = objs
        self.optgraph_id = optgraph_id
        self.weight = weight  # 修改
        self.prac_weight = prac_weight
        self.neighbor_weight = neighbor_weight

    @classmethod
    def copy_from(cls, sample):
        env_params = deepcopy(sample.env_params)
        actor_critic = deepcopy(sample.actor_critic)
        agent = deepcopy(sample.agent)
        objs = deepcopy(sample.objs)
        optgraph_id = sample.optgraph_id
        weight = deepcopy(sample.weight)  # 修改
        neighbor_weight = deepcopy(sample.neighbor_weight)  # 修改
        prac_weight = deepcopy(sample.prac_weight)  # 修改
        return cls(env_params, actor_critic, agent, objs, optgraph_id, weight, neighbor_weight, prac_weight)

    def link_policy_agent(self):
        self.agent.actor_critic = self.actor_critic
        optim_state_dict = deepcopy(self.agent.optimizer.state_dict())
        self.agent.optimizer = optim.Adam(self.actor_critic.parameters(), lr = 3e-4, eps = 1e-5)
        self.agent.optimizer.load_state_dict(optim_state_dict)