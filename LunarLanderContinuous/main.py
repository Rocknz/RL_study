from Reinforcement import DQN
import gym
import torch
import numpy


def main():
    torch.device("cuda:0")
    RL = DQN()
    # install box2d-py

    env = gym.make("LunarLanderContinuous-v2")
    # Position X
    # Position Y
    # Velocity X
    # Velocity Y
    # Angle
    # Angular Velocity
    # Is left leg touching the ground
    # Is right leg touching the ground

    print("state_space", env.observation_space)
    # main engine, left - mid - right engine
    print("action_space", env.action_space)

    while True:
        state = env.reset()
        dictionary = numpy([])
        while True:
            print(state)
            env.render()

            action = [0, 0]
            state, reward, done, info = env.step(action)
            dictionary.append([])
            if done:
                break


if __name__ == "__main__":
    main()