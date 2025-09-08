import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

# external packages
import numpy as np
import time
import torch
import copy
import json
from tqdm import tqdm

# self writing files
import grid_networks as networks
import grid_model


class MARL(object):
    def __init__(self,
                num_grid=4,
                num_agents=None,
                num_obstacles=0,
                env_max_steps=200,
                env_seed=None,
                fully_connected_adj=False,
                max_steps=6,
                max_episode=500,
                update_batch=16,
                batch_size=64,
                buffer_capacity=100000,
                update_interval=10,
                save_interval=200,
                eval_interval=50,
                eval_episodes=10,
  # —— 优化/稳定性 ——
                lr=3e-4,
                lr_decay=False,
                grad_clip=True,
                max_grad_norm=5,
                soft_replace_rate=0.02,
                gamma=0.0,
                explore_noise=0.2,
                explore_noise_decay=True,
                explore_noise_decay_rate=0.2):
        super().__init__()

        print('Initializing...')

        # generate config
        config_data=locals()
        del config_data['self']
        del config_data['__class__']
        time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        config_data['time']=time_data

        # environment
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(9)
        if self.device=='cuda':
            torch.cuda.manual_seed(9)
        np.random.seed(9)

        self.num_grid=num_grid

        # simulator
        self.simulator=grid_model.Model(grid_size_default=self.num_grid,
                                        num_agents_default=3,
                                        num_obstacles_default=8,
                                        fully_connected_adj=bool(fully_connected_adj),
                                        device=self.device,
                                        max_steps_default=int(env_max_steps),
                                        seed_default=env_seed)
        self.simulator.init_exogenous_variables(num_grid=self.num_grid)
        self.N = self.simulator.Gmat.shape[0]

        # adjency matrix
        self.Gmat=self.simulator.Gmat
        self.Gmat=torch.FloatTensor(self.Gmat).to(self.device)

        # learning parameters
        self.max_steps=max_steps
        self.max_episode=max_episode
        self.update_batch=update_batch
        self.batch_size=batch_size
        self.update_interval=update_interval
        self.save_interval=save_interval
        self.eval_interval=eval_interval      # 新增
        self.eval_episodes=eval_episodes      # 新增
        self.lr=lr
        self.lr_decay=lr_decay
        self.grad_clip=grad_clip
        self.max_grad_norm=max_grad_norm
        self.soft_replace_rate=soft_replace_rate
        self.gamma=gamma
        self.explore_noise=explore_noise
        self.explore_noise_decay=explore_noise_decay
        self.explore_noise_decay_rate=explore_noise_decay_rate

        # networks and optimizers
        self.actor=networks.Actor().to(self.device)
        self.actor_target=copy.deepcopy(self.actor).eval()
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=self.lr)

        self.critic=networks.Critic().to(self.device)
        self.critic_target=copy.deepcopy(self.critic).eval()
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=self.lr)

        self.actor_attention=networks.Attention().to(self.device)
        self.actor_attention_target=copy.deepcopy(self.actor_attention).eval()
        self.actor_attention_optimizer=torch.optim.Adam(self.actor_attention.parameters(),lr=self.lr)

        self.critic_attention=networks.Attention().to(self.device)
        self.critic_attention_target=copy.deepcopy(self.critic_attention).eval()
        self.critic_attention_optimizer=torch.optim.Adam(self.critic_attention.parameters(),lr=self.lr)

        if self.lr_decay:
            self.actor_optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=lambda epoch: 0.99**epoch)
            self.critic_optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=lambda epoch: 0.99**epoch)
            self.actor_attention_optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_attention_optimizer, lr_lambda=lambda epoch: 0.99**epoch)
            self.critic_attention_optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_attention_optimizer, lr_lambda=lambda epoch: 0.99**epoch)

        # buffer
        self.buffer_capacity=buffer_capacity
        self.buffer_pointer=0
        self.buffer_size=0
        self.buffer_s = np.empty((self.buffer_capacity, self.N, 5), np.float32)
        self.buffer_a = np.empty((self.buffer_capacity, self.N, 5), np.float32)
        self.buffer_s1 = np.empty((self.buffer_capacity, self.N, 5), np.float32)
        self.buffer_r = np.empty((self.buffer_capacity, self.N, 1), np.float32)
        self.buffer_end = np.ones((self.buffer_capacity, self.N, 1), np.float32)

        # training trackors
        self.episode_return_trackor=list()
        self.critic_loss_trackor=list()
        self.actor_loss_trackor=list()
        
        # 新增评估指标记录
        self.safety_ratio_trackor=list()
        self.arrival_rate_trackor=list()

        # making output directory
        self.output_dir=os.path.join('..','model',f'{self.num_grid}grid_{time_data}')
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir,'config.json'),'w') as f:
            json.dump(config_data,f)

        print(f'Training platform with {self.num_grid*self.num_grid} grids initialized')

    def get_action(self,action):
        action_vector=torch.softmax(action,dim=-1)
        return action_vector

    def get_entropy(self,action):
        weight=torch.softmax(action,dim=1)
        action_entropy=-torch.sum(weight*torch.log2(weight))
        return action_entropy

    def is_safe(self, graph):
        """
        检查智能体是否安全（不与其他智能体或障碍物同格或相邻格）

        Args:
            graph: 当前图状态

        Returns:
            safe_flags: (N,) 布尔张量，True表示安全
        """
        agent_pos = graph.env_states.agent  # (N, 2)
        obs_pos = graph.env_states.obstacle.positions  # (n_obs, 2)
        N = agent_pos.shape[0]

        # 初始化安全标志（默认所有智能体都安全）
        safe_flags = torch.ones(N, dtype=torch.bool, device=agent_pos.device)

        # 检查每个智能体
        for i in range(N):
            current_pos = agent_pos[i]  # (2,)
            is_safe_i = True

            # 1. 检查与其他智能体的冲突（同格或相邻格）
            for j in range(N):
                if i == j:
                    continue
                other_pos = agent_pos[j]  # (2,)

                # 计算曼哈顿距离
                manhattan_dist = torch.abs(current_pos - other_pos).sum().item()

                # 如果距离 <= 1（同格或相邻格），则不安全
                if manhattan_dist <= 1:
                    is_safe_i = False
                    break

            # 2. 如果与智能体没有冲突，再检查与障碍物的冲突
            if is_safe_i and obs_pos.numel() > 0:
                for k in range(obs_pos.shape[0]):
                    obs_k = obs_pos[k]  # (2,)

                    # 计算与障碍物的曼哈顿距离
                    manhattan_dist = torch.abs(current_pos - obs_k).sum().item()

                    # 如果距离 <= 1（同格或相邻格），则不安全
                    if manhattan_dist <= 1:
                        is_safe_i = False
                        break

            safe_flags[i] = is_safe_i

        return safe_flags

    def has_reached_goal(self, graph):
        """检查智能体是否到达目标"""
        agent_pos = graph.env_states.agent  # (N, 2)
        goal_pos = graph.env_states.goal    # (N, 2)
        
        # 检查每个智能体是否到达其目标（允许小误差）
        reached = torch.all(torch.abs(agent_pos - goal_pos) < 0.5, dim=1)
        return reached

    def evaluate_policy(self):
        """评估当前策略的安全性和到达率"""
        total_safe_steps = 0
        total_steps = 0
        total_agents_reached = 0
        total_agents = 0
        
        print("Evaluating policy...")
        
        for eval_ep in range(self.eval_episodes):
            # 重置环境
            self.simulator.init_endogenous_variables()
            current_state = self.simulator.output_record()
            
            episode_safe_steps = 0
            agents_reached = torch.zeros(self.N, dtype=torch.bool)
            
            for step in range(self.max_steps):
                with torch.no_grad():
                    state = torch.FloatTensor(current_state).to(self.device).unsqueeze(0)
                    
                    # 获取动作（无噪声，确定性策略）
                    Actor_attention = self.actor_attention(state, self.Gmat)
                    Actor_state_bar = torch.bmm(Actor_attention, state)
                    Actor_state_all = torch.concat([state, Actor_state_bar], dim=-1)
                    action = self.actor(Actor_state_all)
                    action_vector = self.get_action(action).squeeze(0).cpu().numpy()

                # 执行动作
                self.simulator.move_miner(action_vector)
                current_state = self.simulator.output_record()
                
                # 检查安全性
                current_graph = self.simulator.graph
                is_safe = self.is_safe(current_graph)
                episode_safe_steps += is_safe.sum().item()
                
                # 检查到达情况
                has_reached = self.has_reached_goal(current_graph)
                agents_reached = agents_reached | has_reached.cpu()
                
                total_steps += self.N
            
            total_safe_steps += episode_safe_steps
            total_agents_reached += agents_reached.sum().item()
            total_agents += self.N
        
        # 计算指标
        safety_ratio = total_safe_steps / total_steps if total_steps > 0 else 0.0
        arrival_rate = total_agents_reached / total_agents if total_agents > 0 else 0.0
        
        return safety_ratio, arrival_rate

    def update(self):
        actor_loss_sum=0
        critic_loss_sum=0

        for batch_count in range(self.update_batch):
            batch_index=np.random.randint(self.buffer_size,size=self.batch_size)
            s_batch=torch.FloatTensor(self.buffer_s[batch_index]).to(self.device)
            a_batch=torch.FloatTensor(self.buffer_a[batch_index]).to(self.device)
            s1_batch=torch.FloatTensor(self.buffer_s1[batch_index]).to(self.device)
            r_batch=torch.FloatTensor(self.buffer_r[batch_index]).to(self.device)
            end_batch=torch.FloatTensor(self.buffer_end[batch_index]).to(self.device)

            with torch.no_grad():
                update_Actor_attention1=self.actor_attention_target(s1_batch,self.Gmat)
                update_Actor_state1_bar=torch.bmm(update_Actor_attention1,s1_batch)
                update_Actor_state1_all=torch.concat([s1_batch,update_Actor_state1_bar],dim=-1)
                update_action1=self.actor_target(update_Actor_state1_all)

                update_Critic_attention1=self.critic_attention_target(s1_batch,self.Gmat)
                update_Critic_state1_bar=torch.bmm(update_Critic_attention1,s1_batch)
                update_Critic_state1_all=torch.concat([s1_batch,update_Critic_state1_bar],dim=-1)
                update_action1_bar=torch.bmm(update_Critic_attention1,update_action1)
                update_action1_all=torch.concat([update_action1,update_action1_bar],dim=-1)

                y=r_batch+self.gamma*self.critic_target(update_Critic_state1_all,update_action1_all)*end_batch

            update_Critic_attention=self.critic_attention(s_batch,self.Gmat)
            update_Critic_state_bar=torch.bmm(update_Critic_attention,s_batch)
            update_Critic_state_all=torch.concat([s_batch,update_Critic_state_bar],dim=-1)
            update_action_bar=torch.bmm(update_Critic_attention,a_batch)
            update_action_all=torch.concat([a_batch,update_action_bar],dim=-1)

            critic_loss=torch.sum(torch.square(y-self.critic(update_Critic_state_all,update_action_all)))/self.batch_size
            critic_loss_sum+=critic_loss.cpu().item()

            self.critic_optimizer.zero_grad()
            self.critic_attention_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic_attention.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            self.critic_attention_optimizer.step()

            update_Actor_attention=self.actor_attention(s_batch,self.Gmat)
            update_Actor_state_bar=torch.bmm(update_Actor_attention,s_batch)
            update_Actor_state_all=torch.concat([s_batch,update_Actor_state_bar],dim=-1)
            update_action=self.actor(update_Actor_state_all)

            with torch.no_grad():
                update_Critic_attention_new=self.critic_attention(s_batch,self.Gmat)
                update_Critic_state_bar_new=torch.bmm(update_Critic_attention_new,s_batch)
                update_Critic_state_all_new=torch.concat([s_batch,update_Critic_state_bar_new],dim=-1)

            update_action_bar_new=torch.bmm(update_Critic_attention_new,update_action)
            update_action_all_new=torch.concat([update_action,update_action_bar_new],dim=-1)

            actor_loss=-torch.sum(self.critic(update_Critic_state_all_new,update_action_all_new))/self.batch_size
            actor_loss_sum+=actor_loss.cpu().item()

            self.actor_optimizer.zero_grad()
            self.actor_attention_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.actor_attention.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.actor_attention_optimizer.step()

            for x in self.actor.state_dict().keys():
                eval('self.actor_target.'+x+'.data.mul_(1-self.soft_replace_rate)')
                eval('self.actor_target.'+x+'.data.add_(self.soft_replace_rate*self.actor.'+x+'.data)')
            for x in self.actor_attention.state_dict().keys():
                eval('self.actor_attention_target.'+x+'.data.mul_(1-self.soft_replace_rate)')
                eval('self.actor_attention_target.'+x+'.data.add_(self.soft_replace_rate*self.actor_attention.'+x+'.data)')
            for x in self.critic.state_dict().keys():
                eval('self.critic_target.'+x+'.data.mul_(1-self.soft_replace_rate)')
                eval('self.critic_target.'+x+'.data.add_(self.soft_replace_rate*self.critic.'+x+'.data)')
            for x in self.critic_attention.state_dict().keys():
                eval('self.critic_attention_target.'+x+'.data.mul_(1-self.soft_replace_rate)')
                eval('self.critic_attention_target.'+x+'.data.add_(self.soft_replace_rate*self.critic_attention.'+x+'.data)')

        if self.lr_decay:
            self.actor_optimizer_scheduler.step()
            self.actor_attention_optimizer_scheduler.step()
            self.critic_optimizer_scheduler.step()
            self.critic_attention_optimizer_scheduler.step()

        critic_loss_mean=critic_loss_sum/self.update_batch
        self.critic_loss_trackor.append(critic_loss_mean)
        actor_loss_mean=actor_loss_sum/self.update_batch
        self.actor_loss_trackor.append(actor_loss_mean)

        tqdm.write(f'Update: Critic Loss {critic_loss_mean} | Actor Loss {actor_loss_mean}')

    def train(self):
        for episode in tqdm(range(self.max_episode)):
            self.simulator.init_endogenous_variables()

            if self.explore_noise_decay:
                self.explore_noise=self.explore_noise/((episode+1)**self.explore_noise_decay_rate)

            current_state=self.simulator.output_record()
            for step in range(self.max_steps):
                self.buffer_s[self.buffer_pointer]=current_state

                with torch.no_grad():
                    state=torch.FloatTensor(current_state).to(self.device).unsqueeze(0)

                    Actor_attention=self.actor_attention(state,self.Gmat)
                    Actor_state_bar=torch.bmm(Actor_attention,state)
                    Actor_state_all=torch.concat([state,Actor_state_bar],dim=-1)
                    action=self.actor(Actor_state_all)

                    action=action+torch.randn_like(action)*torch.mean(torch.abs(action))*self.explore_noise
                    action_vector=self.get_action(action).squeeze(0).cpu().numpy()

                reward_old=self.simulator.get_reward()
                self.simulator.move_miner(action_vector)
                reward_new=self.simulator.get_reward()
                reward=reward_new-reward_old

                current_state=self.simulator.output_record()

                self.buffer_s1[self.buffer_pointer]=current_state
                self.buffer_a[self.buffer_pointer]=action.cpu().numpy()
                self.buffer_r[self.buffer_pointer]=reward
                if step==self.max_steps-1:
                    self.buffer_end[self.buffer_pointer]=0

                self.buffer_size=max(self.buffer_size,self.buffer_pointer+1)
                self.buffer_pointer=(self.buffer_pointer+1)%self.buffer_capacity

            episode_return=self.simulator.get_return()
            self.episode_return_trackor.append(episode_return)

            tqdm.write(f'Episode {episode}: Return {episode_return}')

            # 定期评估策略
            if (episode+1) % self.eval_interval == 0:
                safety_ratio, arrival_rate = self.evaluate_policy()
                self.safety_ratio_trackor.append(safety_ratio)
                self.arrival_rate_trackor.append(arrival_rate)
                tqdm.write(f'Evaluation - Safety Ratio: {safety_ratio:.4f}, Arrival Rate: {arrival_rate:.4f}')

            if (episode+1)%self.update_interval==0:
                self.update()

            if (episode+1)%self.save_interval==0:
                self.save_models(episode+1)

    def save_models(self,episode):
        torch.save(self.actor.state_dict(),os.path.join(self.output_dir,f'actor_{episode}.pth'))
        torch.save(self.actor_target.state_dict(),os.path.join(self.output_dir,f'actor_target_{episode}.pth'))

        torch.save(self.actor_attention.state_dict(),os.path.join(self.output_dir,f'actor_attention_{episode}.pth'))
        torch.save(self.actor_attention_target.state_dict(),os.path.join(self.output_dir,f'actor_attention_target_{episode}.pth'))

        torch.save(self.critic.state_dict(),os.path.join(self.output_dir,f'critic_{episode}.pth'))
        torch.save(self.critic_target.state_dict(),os.path.join(self.output_dir,f'critic_target_{episode}.pth'))

        torch.save(self.critic_attention.state_dict(),os.path.join(self.output_dir,f'critic_attention_{episode}.pth'))
        torch.save(self.critic_attention_target.state_dict(),os.path.join(self.output_dir,f'critic_attention_target_{episode}.pth'))

        with open(os.path.join(self.output_dir,'episode_return.json'),'w') as f:
            json.dump(str(self.episode_return_trackor),f)
        with open(os.path.join(self.output_dir,'critic_loss.json'),'w') as f:
            json.dump(str(self.critic_loss_trackor),f)
        with open(os.path.join(self.output_dir,'actor_loss.json'),'w') as f:
            json.dump(str(self.actor_loss_trackor),f)
        # 保存新增的评估指标
        with open(os.path.join(self.output_dir,'safety_ratio.json'),'w') as f:
            json.dump(str(self.safety_ratio_trackor),f)
        with open(os.path.join(self.output_dir,'arrival_rate.json'),'w') as f:
            json.dump(str(self.arrival_rate_trackor),f)


if __name__ == '__main__':
    trainer = MARL(
        # —— 环境参数 ——
        num_grid=6,
        num_agents=3,  # 不填则默认为 6
        num_obstacles=2,
        env_max_steps=150,
        env_seed=123,
        fully_connected_adj=False,
        # —— 训练器采样/优化 ——
        max_steps=10,
        max_episode=300,
        update_batch=16,
        batch_size=64,
        buffer_capacity=100_000,
        update_interval=10,
        save_interval=100,
        eval_interval=50,
        eval_episodes=10,
        lr=3e-4,
        grad_clip=True,
        max_grad_norm=5,
        soft_replace_rate=0.02,
        gamma=0.0,
        explore_noise=0.2,
        explore_noise_decay=True,
        explore_noise_decay_rate=0.2,
    )
    trainer.train()
