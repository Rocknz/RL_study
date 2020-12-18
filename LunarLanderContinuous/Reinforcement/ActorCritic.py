import torch
import numpy as np
from torch import nn
import random

class Network(nn.Module):
    def __init__(self, input_space, output_space):
        super(Network, self).__init__()
        self.L1 = nn.Linear(input_space, 1000)
        self.Seq = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 10),
            nn.LeakyReLU(),
            nn.Linear(10, output_space)
        )

    def forward(self, x):
        x = self.L1(x)
        x = self.Seq(x)
        return x

class ActorCritic():
    def __init__(self, name, state_space, action_space, cuda_num = 0):
        self.Critic = Network(state_space + action_space, 1).cuda(0)
        self.Actor = Network(state_space, action_space).cuda(0)
        self.name = name
        self.sample_size = 100
        self.learning_rate = 0.001
        self.gamma = 0.999
        self.cuda_num = cuda_num
        self.optimizer_critic = torch.optim.Adam(
            params=self.Critic.parameters(), lr=self.learning_rate
        )
        self.optimizer_actor = torch.optim.Adam(
            params=self.Actor.parameters(), lr=self.learning_rate
        )
        self.criterion = nn.MSELoss(reduction='sum')
    
    def action(self, state):
        state = torch.from_numpy(state).float().cuda(0)
        state = state.unsqueeze(0)
        res = self.Actor(state)
        res = res.squeeze(0).detach().cpu().numpy()
        return res

    def sampling(self, history):
        size = self.sample_size
        if size > len(history):
            size = len(history)

        history_sample = random.sample(history, size)
        return history_sample

    def learn_actor(self, history):
        sample = self.sampling(history)
        s = np.array([i[0] for i in sample])
        s = torch.from_numpy(s).float().cuda(self.cuda_num)
        actions = self.Actor(s)

        input = torch.cat((s, actions), 1)
        now = self.Critic(input).squeeze(1)
        zero = torch.zeros(now.shape).float().cuda()

        self.optimizer_actor.zero_grad()
        loss = -1 * self.criterion(now, zero)
        loss.backward()
        self.optimizer_actor.step()


    def learn_critic(self, history):
        sample = self.sampling(history)
        s = np.array([i[0] for i in sample])
        action = np.array([i[1] for i in sample])
        s_next = np.array([i[2] for i in sample])
        action_next = np.array([i[3] for i in sample])
        reward = np.array([i[4] for i in sample])
        done = np.array([i[5] for i in sample])
        
        s = torch.from_numpy(s).float().cuda(self.cuda_num)
        action = torch.from_numpy(action).float().cuda(self.cuda_num)
        s = torch.cat((s, action), 1)

        s_next = torch.from_numpy(s_next).float().cuda(self.cuda_num)
        action_next = torch.from_numpy(action_next).float().cuda(self.cuda_num)        
        s_next = torch.cat((s_next, action_next), 1)

        reward = torch.from_numpy(reward).float().cuda(self.cuda_num)
        done = torch.from_numpy(done).bool().cuda(self.cuda_num)

        n_now = self.Critic(s).squeeze(1)
        n_next = self.Critic(s_next).squeeze(1)

        ans = reward + self.gamma * n_next
        # if its done. => Q-value = 0
        ans[done] = 0

        self.optimizer_critic.zero_grad()
        loss = self.criterion(n_now, ans)
        loss.backward()
        self.optimizer_critic.step()
        return loss.item() / len(sample)

    def learn(self, history, num = 100):
        # critic update
        for _ in range(num):
            self.learn_critic(history)

        # action update
        for _ in range(num):
            self.learn_actor(history)

    def save(self):
        torch.save(self.Actor.state_dict(), self.name + "_actor")
        torch.save(self.Critic.state_dict(), self.name + "_critic")

    def load(self):
        try:
            self.Actor.load_state_dict(torch.load(self.name + "_actor"))
            self.Critic.load_state_dict(torch.load(self.name + "_critic"))
            print("load_end!")
        except:
            print("load_error!")

