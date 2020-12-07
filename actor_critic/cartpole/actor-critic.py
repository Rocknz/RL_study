import torch
from torch import nn
import gym
import numpy as np

# independent.

# Critic
# Q^PI (s, a), PI(s) = [values] ==> #a

# PI(s) = vector_a(a == argmax_a Q^PI (s, a)) +
# PI has soft-max


class Actor(nn.Module):
    def Actor(self, state_space, action_space):
        self.l1 = nn.Linear(state_space, 1000)
        self.l = nn.Sequential(
            [
                nn.LeakyReLU(),
                nn.Linear(1000, 1000),
                nn.LeakyReLU(),
                nn.Linear(1000, 100),
                nn.LeakyReLU(),
                nn.Linear(100, 100),
                nn.LeakyReLU(),
                nn.Linear(100, action_space),
                nn.Softmax(),
            ]
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l(x)
        return x


class Critic(nn.Module):
    def Critic(self, state_space, action_space):
        self.l1 = nn.Linear(state_space + action_space, 100)
        self.l = nn.Sequential(
            [
                nn.LeakyReLU(),
                nn.Linear(1000, 1000),
                nn.LeakyReLU(),
                nn.Linear(1000, 100),
                nn.LeakyReLU(),
                nn.Linear(100, 100),
                nn.LeakyReLU(),
                nn.Linear(100, 1),
            ]
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l(x)
        return x


class Learning:
    def Learning(self, state_space, action_space):
        self.actor = Actor(state_space, action_space)
        self.critic = Critic(state_space, action_space)

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
        res = self.actor(state).numpy()
        val = np.random.choice(5, 3, p=res)
        return val

    def learn_critic(self, history):
        pass

    def learn_actor(self, history):
        pass


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
        for _ in range(100):
            state = env.reset()
            step = 0
            while True:
                action = learning.action(state)
                before_state = state
                before_action = action
                state, reward, done, info = env.step(action)
                history.append([before_state, before_action, state, reward, done])
                learning.learn_actor(history)
                step += 1

                if done:
                    break
            avg_step += step
            N += 1

        avg_step /= N
        print("avg_step {}".format(avg_step))
        for _ in range(1000):
            learning.learn_critic(history)


if __name__ == "__main__":
    main()
