import torch
from torch import nn
import gym
import numpy as np
import random
from threading import Thread, Lock
# independent.

# Critic
# Q^PI (s, a), PI(s) = [values] ==> #a

# PI(s) = vector_a(a == argmax_a Q^PI (s, aa)) +
# PI has soft-max


class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_space, 1000)
        self.l = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, action_space),
            nn.LeakyReLU(),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l(x)
        return x


class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_space, 1000)
        self.l = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, action_space)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l(x)
        return x


class Learning:
    def __init__(self, state_space, action_space, cuda_num=0):
        self.actor = Actor(state_space, action_space).cuda(cuda_num)
        self.critic = Critic(state_space, action_space).cuda(cuda_num)
        self.cuda_num = cuda_num
        self.sample_size = 1000
        self.criterion = nn.MSELoss(reduction='sum')
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.optimizer_actor = torch.optim.Adam(
            params=self.actor.parameters(), lr=self.learning_rate)

        self.optimizer_critic = torch.optim.Adam(
            params=self.critic.parameters(), lr=self.learning_rate)

    def save(self):
        torch.save(self.actor.state_dict(), "cartpole_actor")
        torch.save(self.critic.state_dict(), "cartpole_critic")

    def load(self):
        try:
            self.actor.load_state_dict(torch.load("cartpole_actor"))
            self.critic.load_state_dict(torch.load("cartpole_critic"))
            print("load_end!")
        except:
            print("load_error!")

    def action(self, state):
        torch_state = torch.from_numpy(
            state).float().unsqueeze(0).cuda(self.cuda_num)
        res = self.actor(torch_state).squeeze(0).cpu().detach().numpy()
        val = np.random.choice(2, 1, p=res)[0]
        return val

    def sampling(self, history):
        size = self.sample_size
        if size > len(history):
            size = len(history)

        history_sample = random.sample(history, size)
        return history_sample

    def learn_actor(self, history):
        sample = self.sampling(history)
        s = np.array([i[0] for i in sample])
        # action = np.array([i[1] for i in sample])
        s = torch.from_numpy(s).float().cuda(self.cuda_num)
        now = self.actor(s)
        with torch.no_grad():
            n_now = self.critic(s)

        softmax = nn.Softmax(1)
        n_now = softmax(n_now)

        self.optimizer_actor.zero_grad()
        loss = self.criterion(now, n_now)
        loss.backward()
        self.optimizer_actor.step()

    def learn_critic(self, history):
        sample = self.sampling(history)
        s = np.array([i[0] for i in sample])
        action = np.array([i[1] for i in sample])
        s_next = np.array([i[2] for i in sample])
        action_next = np.array([i[3] for i in sample])
        # action = np.array([i[1] for i in sample])
        reward = np.array([i[4] for i in sample])
        done = np.array([i[5] for i in sample])

        s = torch.from_numpy(s).float().cuda(self.cuda_num)
        action = torch.from_numpy(action).long().cuda(self.cuda_num)
        s_next = torch.from_numpy(s_next).float().cuda(self.cuda_num)
        action_next = torch.from_numpy(action_next).long().cuda(self.cuda_num)

        # action = torch.from_numpy(action).float().cuda(self.cuda_num)
        reward = torch.from_numpy(reward).float().cuda(self.cuda_num)
        done = torch.from_numpy(done).bool().cuda(self.cuda_num)

        n_now = self.critic(s)
        n_next = self.critic(s_next)

        iters = torch.Tensor(range(n_now.shape[0])
                             ).long().cuda(self.cuda_num)

        n_now = n_now[iters[:], action[:]]
        n_next = n_next[iters[:], action_next[:]]

        ans = reward + self.gamma * n_next
        # if its done. => Q-value = 0
        ans[done] = 0

        self.optimizer_critic.zero_grad()
        loss = self.criterion(n_now, ans)
        loss.backward()
        self.optimizer_critic.step()
        return loss.item() / len(sample)


def simulation(index, mutex, history, steps, learning):
    env = gym.make("CartPole-v1")
    state = env.reset()
    step = 0
    next_action = learning.action(state)
    while True:
        # env.render()
        action = next_action

        before_state = state
        before_action = action
        state, reward, done, info = env.step(action)
        next_action = learning.action(state)
        if state[0] < 0.5 and state[0] > -0.5:
            reward += (0.5 - abs(state[0])) * (0.5 - abs(state[0])) * 4

        if state[1] < 0.5 and state[1] > -0.5:
            reward += (0.5 - abs(state[1])) * (0.5 - abs(state[1])) * 4
        
        reward *= 20
        
        mutex.acquire()
        history.append(
            [before_state, before_action, state, next_action, reward, done])
        mutex.release()

        step += 1

        if done:
            break

    steps[index] = step


def main():
    torch.device("cuda:0")

    env = gym.make("CartPole-v1")

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    learning = Learning(state_space=state_space, action_space=action_space)
    learning.load()
    while True:
        # iteration actor-critic
        # critic
        history = []

        avg_step = 0
        N = 0

        # run simulation with multithread
        mutex = Lock()
        threads = []
        steps = []
        simulation_num = 500
        print("sim_start")
        for _ in range(simulation_num):
            steps.append(0)

        for index in range(simulation_num):
            step = 0
            thread = Thread(target=simulation, args=(
                index, mutex, history, steps, learning))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        for step in steps:
            avg_step += step
            N += 1

        avg_step /= N
        print("avg_step {}".format(avg_step))
        print("learn_critic")
        min = -1
        # while True:
        for _ in range(1000):
            loss = learning.learn_critic(history)
            # if loss < min or min == -1:
            #     min = loss
            #     print(min)

            # if loss < 0.1:
            #     break

        print("learn_actor")
        for _ in range(1000):
            learning.learn_actor(history)

        learning.save()


if __name__ == "__main__":
    main()
