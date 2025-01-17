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

        self.d1 = nn.Linear(5, 1000)
        self.d2 = nn.Linear(1000, 1000)
        self.d3 = nn.Linear(1000, 100)
        self.d4 = nn.Linear(100, 100)
        self.d5 = nn.Linear(100, 1)
        self.relu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        ox = x
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
        self.criterion = nn.MSELoss(reduction="sum")
        self.explorate_rate_max = 1.0
        self.explorate_rate_min = 0.1
        self.explorate_decay = 0.9999
        # self.explorate_decay = 0.995

        self.explorate_rate = self.explorate_rate_max

    def next(self, state):

        explore = np.random.rand()
        if explore < self.explorate_rate:
            return random.randrange(2)
        else:
            next_action = 0
            now = state
            # print(now)
            now_A = torch.from_numpy(np.array([[np.append(now, 0)]])).float().cuda(0)
            now_B = torch.from_numpy(np.array([[np.append(now, 1)]])).float().cuda(0)
            # print(now_A)
            A = self.Q(now_A)
            B = self.Q(now_B)
            if A[0] > B[0]:
                now = 0
            else:
                now = 1

            next_action = now
            return next_action

    def decay(self):
        self.explorate_rate *= self.explorate_decay
        if self.explorate_rate < self.explorate_rate_min:
            self.explorate_rate = self.explorate_rate_min

    def learn(self, save):
        self.optimizer.zero_grad()
        input = np.array([np.append(i[0], i[1]) for i in save])
        input = torch.from_numpy(input).float().cuda(0)
        output = self.Q(input).squeeze(1)

        left = np.array([np.append(i[2], 0) for i in save])
        left = torch.from_numpy(left).float().cuda(0)
        right = np.array([np.append(i[2], 1) for i in save])
        right = torch.from_numpy(right).float().cuda(0)

        res1 = self.Q(left)
        res2 = self.Q(right)
        res = torch.stack([res1, res2], 1).squeeze(2)
        res, index = torch.max(res, 1)
        # res = Variable(res, requires_grad=False)

        reward = np.array([i[3] for i in save])
        reward = torch.from_numpy(reward).float().cuda(0)
        done = np.array([i[4] for i in save])
        done = torch.from_numpy(done).bool().cuda(0)

        res[done] = 0
        # print(res)
        res = self.gamma * res + reward
        loss = self.criterion(output, res)
        # loss = ((output - res) * (output - res)).sum()
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
        cnt = 0
        for _ in range(1000):
            env.render()
            # print(obs)
            next_action = q.next(obs)
            before_obs = obs
            before_action = next_action
            obs, reward, done, info = env.step(next_action)
            # print("{}, {}".format(done, reward))
            if obs[0] <= 1.0 and obs[0] >= -1.0:
                reward += 0.2

            if obs[2] <= 0.1 and obs[2] >= -0.1:
                reward += 1

            if obs[2] <= 0.01 and obs[2] >= -0.01:
                reward += 10

            res += reward
            cnt += 1

            save.append([before_obs, before_action, obs, reward, done])
            # add position threshold.
            dictionary.extend([[before_obs, before_action, obs, reward, done]])
            # dictionary.extend(save[:])
            dictionary_size = 1000000
            sampling_size = 200
            while len(dictionary) > dictionary_size:
                n = random.sample(range(len(dictionary)), 1)
                dictionary.pop(n[0])

            if len(dictionary) >= sampling_size:
                dictionary_sample = random.sample(dictionary, sampling_size)
                q.learn(dictionary_sample)

            q.decay()
            if done:
                done_cnt += 1
                # env.render()
                if done_cnt >= 1:
                    break

        print(res, cnt, q.explorate_rate)

        cnt = 0
        if cnt % 1000 == 0:
            q.save()
        cnt += 1

    env.close()


if __name__ == "__main__":
    main()
