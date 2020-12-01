import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.autograd import Variable


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        # self.d1 = nn.Linear(5, 1024, bias=False)
        # self.d2 = nn.Linear(1029, 256, bias=False)
        # self.d3 = nn.Linear(261, 128, bias=False)
        # self.d4 = nn.Linear(133, 32, bias=False)
        # self.d5 = nn.Linear(37, 1, bias=False)

        self.d1 = nn.Linear(4, 100, bias=False)
        self.d2 = nn.Linear(100, 100, bias=False)
        self.d3 = nn.Linear(100, 10, bias=False)
        self.d4 = nn.Linear(10, 10, bias=False)
        self.d5 = nn.Linear(10, 2, bias=False)
        self.relu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # ox = x
        x = self.relu(self.d1(x))
        # x = torch.cat((x, ox), 1)
        x = self.relu(self.d2(x))
        # x = torch.cat((x, ox), 1)
        x = self.relu(self.d3(x))
        # x = torch.cat((x, ox), 1)
        x = self.relu(self.d4(x))
        # x = torch.cat((x, ox), 1)
        x = self.d5(x)
        return x


class Q_Learning:
    def __init__(self):
        self.Q = Q()
        self.Q.cuda(0)
        self.gamma = 0.95
        # self.optimizer = optim.SGD(self.Q.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.explorate_rate_max = 1.0
        self.explorate_rate_min = 0.01
        self.explorate_decay = 0.9999

        self.explorate_rate = self.explorate_rate_max

    def next(self, state):

        explore = np.random.rand()
        if explore < self.explorate_rate:
            return random.randrange(2)
        else:
            next_action = 0
            now = state
            # print(now)
            A = torch.tensor(now).float().cuda(0)
            A = self.Q(A)
            # print(now_A)
            if A[0] > A[1]:
                now = 0
            else:
                now = 1

            next_action = now
            return next_action

    def learn(self, save):
        self.explorate_rate *= self.explorate_decay
        if self.explorate_rate < self.explorate_rate_min:
            self.explorate_rate = self.explorate_rate_min

        self.optimizer.zero_grad()

        input = np.array([i[0] for i in save])
        input = torch.from_numpy(input).float().cuda(0)

        action = np.array([i[1] for i in save])
        action = torch.from_numpy(action).long().cuda(0)

        output = self.Q(input)
        output = output[:, action]

        left = np.array([i[2] for i in save])
        left = torch.from_numpy(left).float().cuda(0)

        res1 = self.Q(left)
        res, index = torch.max(res1, 1)
        # res = Variable(res, requires_grad=False)

        reward = np.array([i[3] for i in save])
        reward = torch.from_numpy(reward).float().cuda(0)
        done = np.array([i[4] for i in save])
        done = torch.from_numpy(done).bool().cuda(0)

        done = done
        reward = reward
        res[done] = 0
        # print(res)
        res = self.gamma * res + reward
        # loss = self.criterion(output, res)
        loss = ((output - res) * (output - res)).sum()
        # print(loss.item())
        loss.backward()

        # for now in save:
        #     val = torch.from_numpy(np.append(now[0], now[1])).float().cuda(0)
        #     val = self.Q(val)
        #     result = torch.tensor([0]).float().cuda(0)
        #     if now[4] != True:
        #         res1 = torch.from_numpy(np.append(now[2], 0)).float().cuda(0)
        #         res2 = torch.from_numpy(np.append(now[2], 1)).float().cuda(0)

        #         res1 = self.Q(res1).item()
        #         res2 = self.Q(res2).item()

        #         if res1 < res2:
        #             res1 = res2

        #         result = torch.tensor(self.gamma * (res1) + now[3]).float().cuda(0)

        #     loss = self.criterion(result, val)
        #     loss.backward()

        self.optimizer.step()

    def save(self):
        torch.save(self.Q.state_dict(), "cartpole_v1")

    def load(self):
        try:
            self.Q.load_state_dict(torch.load("cartpole_v1"))
            print("load_end!")
        except:
            print("load_error!")


def main():
    torch.device("cuda:0")

    env = gym.make("CartPole-v1")
    q = Q_Learning()
    q.load()
    print("action_space")
    print(env.action_space)
    print("env_space")
    print(env.observation_space)
    # for _ in range(1000):
    dictionary = []
    while True:
        obs = env.reset()
        next_action = 0
        save = []
        done_cnt = 0
        res = 0
        for _ in range(1000):
            # env.render()
            # print(obs)
            next_action = q.next(obs)
            before_obs = obs
            before_action = next_action
            obs, reward, done, info = env.step(next_action)
            # print("{}, {}".format(done, reward))
            # if obs[0] >= 2.0 or obs[0] <= -2.0:
            #     reward = 0
            #     done = True

            res += reward
            save.append([before_obs, before_action, obs, reward, done])

            # dictionary.extend(save[:])
            dictionary.extend([[before_obs, before_action, obs, reward, done]])
            dictionary_size = 100000
            sampling_size = 20
            while len(dictionary) > dictionary_size:
                n = random.sample(range(len(dictionary)), 1)
                dictionary.pop(n[0])

            if len(dictionary) > sampling_size:
                dictionary_sample = random.sample(dictionary, min(sampling_size, len(dictionary)))
                q.learn(dictionary_sample)

            # add position threshold.
            if done:
                done_cnt += 1
                # env.render()
                if done_cnt >= 1:
                    break

        print(res, q.explorate_rate)

        cnt = 0
        if cnt % 1000 == 0:
            q.save()
        cnt += 1

    env.close()


if __name__ == "__main__":
    main()
