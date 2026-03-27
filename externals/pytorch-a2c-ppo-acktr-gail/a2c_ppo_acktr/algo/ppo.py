import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 obj_weights=None,
                 scalarization_func=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.obj_weights = None if obj_weights is None else torch.Tensor(obj_weights)

        self.scalarization_func = scalarization_func

        self.reward_dim = 0  # 修改

    def update(self, rollouts, scalarization = None, obj_var = None, warmup = False, neighbor_weight = None):
        op_axis = tuple(range(len(rollouts.returns.shape) - 1))
        new_weights = []
        prac_weight = []

        # # 计算优势值  GAE方法? 这是未加权的优势值。如果下一个if判断为真，说明给了标量化方法，这个优势值似乎会被覆盖掉。
        # advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        # # 未加权的优势值是（2048，4，2），一次交互长度为2048，并行运行了4个进程，2个目标。
        # if self.scalarization_func is not None or scalarization is not None:
        # # scalarization_func是建立PPO agent时为它赋予的标量化方法；而scalarization是调用update函数时另外赋予的标量化方法
        #     # recover the raw returns
        #     # 恢复原始的return（输入的return是经过方差归一化处理了）
        #     returns = rollouts.returns * torch.Tensor(np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.returns
        #     value_preds = rollouts.value_preds * torch.Tensor(np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.value_preds
        #
        #     # 这里的意思是，如果调用update函数时给定了标量化方法（权重），就用这个权重进行标量化。否则就用建立PPO agent时的那个权重
        #     if scalarization is not None:
        #         # 用scalarization.evaluate函数把优势值加权。注意是先把critic预测的value加权后，再用return减去value，再对整体加权。
        #         advantages = scalarization.evaluate(returns[:-1]) - scalarization.evaluate(value_preds[:-1])
        #     else:
        #         advantages = self.scalarization_func.evaluate(returns[:-1]) - self.scalarization_func.evaluate(value_preds[:-1])

        # 恢复原始的return（输入的return是经过方差归一化处理了）  修改
        # returns = rollouts.returns * torch.Tensor(np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.returns
        # value_preds = rollouts.value_preds * torch.Tensor(np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.value_preds
        # 如果不恢复原始的return会怎样？
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        # advantages = returns[:-1] - value_preds[:-1]

        # 归一化优势值
        advantages = (advantages - advantages.mean(axis=op_axis)) / (
            advantages.std(axis=op_axis) + 1e-5)

        # 初始化损失函数和熵的值为0
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        # 重复多次更新模型参数
        for e in range(self.ppo_epoch):  # ppo-epoch = 10
            # 如果模型是recurrent，则使用recurrent_generator，否则使用feed_forward_generator
            if self.actor_critic.is_recurrent:  # 这里没有使用rnn，为false
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)  # 注意这个函数内部有修改，优势值在这里还没有加权，输出对应于2个目标的

            # 遍历mini-batch数据
            for sample in data_generator:
                # 从mini-batch数据中获取需要的数据。其中adv_targ是把输入的优势值按batch size分割了，维度是（256，1）
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                # 计算策略的动作值，动作的对数概率，策略的熵，以及可忽略的值
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                # 计算重要性采样的比值
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ

                # 计算动作损失  修改
                # action_loss = -torch.min(surr1, surr2).mean()
                obj_num = len(obj_var)  # 目标数目
                self.reward_dim = obj_num
                action_losses = []
                for ii in range(obj_num):
                    action_losses.append(-torch.min(surr1[:, ii], surr2[:, ii]).mean())

                # if warmup:
                #     new_weights = scalarization.weights
                # else:
                #     new_weights, pareto_loss = self.find_preferences(action_losses)

                # prac_weight = new_weights
                new_weights = scalarization.weights
                prac_weight = scalarization.weights
                # if neighbor_weight is not None:
                #     for kk in range(len(neighbor_weight)):
                #         prac_weight = prac_weight + neighbor_weight[kk, :]
                #     prac_weight = prac_weight / (len(neighbor_weight)+1)
                #     # prac_weight, _ = self.find_preferences_prac(neighbor_weight)
                # else:
                #     prac_weight = new_weights


                action_loss = 0
                for i in range(self.reward_dim):
                    action_loss += prac_weight[i] * action_losses[i]

                # penalty = 0
                # if neighbor_weight is not None:
                #     for iii in range(len(neighbor_weight)):
                #         penalty += torch.dist(new_weights, neighbor_weight[iii, :])
                #     penalty = penalty / (len(neighbor_weight) - 1)
                #     action_loss = 0.8 * action_loss + 0.2 * penalty
                #
                # # if neighbor_weight is not None:
                # #     for iii in range(len(neighbor_weight)):
                # #         action_loss += torch.dist(new_weights, neighbor_weight[iii, :])


                # 计算值函数损失
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # 清除优化器的梯度
                self.optimizer.zero_grad()
                # 计算总损失并反向传播
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                # 对梯度进行剪切，以避免梯度爆炸
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                # 更新模型参数
                self.optimizer.step()

                ''' 在 PyTorch 中，每个张量都有一个 requires_grad 属性，用于指示是否需要计算其梯度。
                    如果需要计算梯度，则可以通过调用张量的 backward() 方法来计算梯度。
                    该方法将使用反向自动微分算法计算梯度，以便通过链式规则将梯度传播到所有相关的张量。
                    如果该张量不是标量，需要指定梯度张量的形状，这通常通过将参数 grad_outputs 指定为与输出形状相同的张量来实现。
                    在上述代码中，backward() 函数被用于计算损失函数对模型参数的梯度，用于模型参数的更新。
                    在这里，通过调用 backward() 函数计算了 value_loss、action_loss 和 dist_entropy 的梯度。
                    这里的梯度计算采用了链式法则，将输出梯度传播到所有相关的张量。然后，使用优化器（如 Adam）根据计算得到的梯度更新模型参数。'''

                '''
                    Value loss和Actor loss的加法可以理解为将它们加权后相加。
                    在PPO算法中，我们要最大化的是一个带约束条件的目标函数，这个函数由Actor的策略损失和Value的值函数损失组成，同时还要控制策略分布的熵。
                    其中，$L_{policy}$是Actor的策略损失，$L_{value}$是Value的值函数损失，$L_{entropy}$是策略分布的熵.
                    由此可见，Value loss和Actor loss之所以可以直接相加，是因为它们都被用作目标函数中的一部分，被赋予了一定的权重。
                    在这个目标函数的优化过程中，通过梯度下降的方式，同时更新Actor和Value的参数，以最小化目标函数的值。
                '''

                # 记录本轮epoch的损失和熵
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        # 计算总更新次数
        num_updates = self.ppo_epoch * self.num_mini_batch

        # 计算平均损失和熵
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, new_weights.detach(), prac_weight.detach()

    ## Optimize Pareto objective for 2d reward singal, can be done analytically
    def find_preferences_2d(self, *losses):
        assert len(losses) == 2

        grads = []
        for loss in losses:
            self.optimizer.zero_grad()

            grad = \
            torch.autograd.grad(loss, self.actor_critic.base.actor.parameters(), retain_graph=True, create_graph=True)[0]
            torch.nn.utils.clip_grad_norm_(self.actor_critic.base.actor.parameters(), self.max_grad_norm)
            grad = torch.flatten(grad)
            grad = torch.squeeze(grad)
            grads.append(grad)

        total_grad = grads[1] - grads[0]
        nom = torch.dot(total_grad, grads[1])  # 修改
        den = torch.norm(total_grad) ** 2
        eps = nom / (den + 1e-8)  # 1e-8是一个参数
        eps = torch.clamp(eps, 0, 1)
        pareto_loss = eps * grads[0] + (1 - eps) * grads[1]
        pareto_loss = torch.norm(pareto_loss) ** 2

        if eps >0:
            pass

        return torch.Tensor([eps, 1-eps]).detach(), pareto_loss  # 修改


    def find_preferences(self, losses):
        if len(losses) == 2:
            return self.find_preferences_2d(*losses)

        # Bellow does not work for adaptive setting, really slow cuz of the projection
        grads = []
        for loss in losses:
            grad = torch.autograd.grad(loss, self.actor_critic.base.actor.parameters(), retain_graph=True, create_graph=True)[0]
            grad = torch.flatten(grad)
            grad = torch.squeeze(grad)
            grads.append(grad)

        epsilon = torch.ones(self.reward_dim, requires_grad=True)
        opt = torch.optim.SGD([epsilon], lr=0.1)

        done = False
        while not done:
            # Crete loss for Pareto stationary problem
            pareto_loss = torch.zeros_like(grads[0])
            for i, grad in enumerate(grads):
                pareto_loss += epsilon[i] * grad
            pareto_loss = torch.linalg.norm(pareto_loss) ** 2

            opt.zero_grad()
            pareto_loss.backward()
            opt.step()
            epsilon.data = self.simplex_proj(epsilon)

            if pareto_loss < 1e-1:
                done = True

        return epsilon.detach(), pareto_loss


    def simplex_proj(self, eps):
        x = eps.detach().numpy()
        y = -np.sort(-x)
        sum = 0
        ind = []
        for j in range(len(x)):
            sum = sum + y[j]
            if y[j] + (1 - sum) / (j + 1) > 0:
                ind.append(j)
            else:
                ind.append(0)
        rho = np.argmax(ind)
        delta = (1 - (y[:rho+1]).sum())/(rho+1)
        proj = np.clip(x + delta, 0, 1)
        return torch.tensor(proj)



    def evaluate_grad_1(self, rollouts, scalarization = None, obj_var = None, warmup = False, neighbor_weight = None):
        op_axis = tuple(range(len(rollouts.returns.shape) - 1))
        new_weights = []
        prac_weight = []

        # 恢复原始的return（输入的return是经过方差归一化处理了）  修改
        returns = rollouts.returns * torch.Tensor(np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.returns
        value_preds = rollouts.value_preds * torch.Tensor(np.sqrt(obj_var + 1e-8)) if obj_var is not None else rollouts.value_preds
        advantages = returns[:-1] - value_preds[:-1]

        # 归一化优势值
        advantages = (advantages - advantages.mean(axis=op_axis)) / (
            advantages.std(axis=op_axis) + 1e-5)

        # 初始化损失函数和熵的值为0
        value_loss_epoch = []
        action_loss_epoch = []
        dist_entropy_epoch = []

        # 重复多次更新模型参数

        # 如果模型是recurrent，则使用recurrent_generator，否则使用feed_forward_generator
        if self.actor_critic.is_recurrent:  # 这里没有使用rnn，为false
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch)
        else:
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)  # 注意这个函数内部有修改，优势值在这里还没有加权，输出对应于2个目标的

        # 遍历mini-batch数据
        for sample in data_generator:
            # 从mini-batch数据中获取需要的数据。其中adv_targ是把输入的优势值按batch size分割了，维度是（256，1）
            obs_batch, recurrent_hidden_states_batch, actions_batch, \
               value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample

            # Reshape to do in a single forward pass for all steps
            # 计算策略的动作值，动作的对数概率，策略的熵，以及可忽略的值
            values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                obs_batch, recurrent_hidden_states_batch, masks_batch,
                actions_batch)
            # 计算重要性采样的比值
            ratio = torch.exp(action_log_probs -
                              old_action_log_probs_batch)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                1.0 + self.clip_param) * adv_targ

            # 计算动作损失  修改
            obj_num = len(obj_var)  # 目标数目
            self.reward_dim = obj_num
            action_losses = []
            for ii in range(obj_num):
                action_losses.append(-torch.min(surr1[:, ii], surr2[:, ii]).mean())

            # 记录本轮epoch的损失
            action_loss_epoch.append(torch.stack(action_losses).squeeze())

        # 计算总更新次数
        num_updates = self.num_mini_batch

        # 计算平均损失和熵
        action_loss_epoch = torch.stack(action_loss_epoch)
        action_loss_epoch_sum = torch.sum(action_loss_epoch, dim=0)
        action_loss_epoch_ave =  action_loss_epoch_sum/num_updates

        final_weights, pareto_loss = self.find_preferences(action_loss_epoch_ave)

        return final_weights.detach()


    # def find_preferences_prac(self, grads):
    #     assert len(grads) == 2
    #
    #     total_grad = grads[1] - grads[0]
    #     nom = torch.dot(total_grad, grads[1])  # 修改
    #     den = torch.norm(total_grad) ** 2
    #     eps = nom / (den + 1e-8)  # 1e-8是一个参数
    #     eps = torch.clamp(eps, 0, 1)
    #     pareto_loss = eps * grads[0] + (1 - eps) * grads[1]
    #     pareto_loss = torch.norm(pareto_loss) ** 2
    #
    #     if eps >0:
    #         pass
    #
    #     return torch.Tensor([eps, 1 - eps]).detach(), pareto_loss  # 修改