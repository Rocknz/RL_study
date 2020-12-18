# from Reinforcement import DQN
from Reinforcement.ActorCritic import ActorCritic
import numpy
import gym
import torch


# def simulation(index, mutex, history, steps, learning):
#     env = gym.make("CartPole-v1")
#     state = env.reset()
#     step = 0
#     next_action = learning.action(state)
#     while True:
#         # env.render()
#         action = next_action

#         before_state = state
#         before_action = action
#         state, reward, done, info = env.step(action)
#         next_action = learning.action(state)

#         mutex.acquire()
#         history.append([before_state, before_action, state, next_action, reward, done])
#         mutex.release()

#         step += 1

#         if done:
#             break

#     steps[index] = step

def main():
    torch.device("cuda:0")
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

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    print("state_space", env.observation_space)
    # main engine, left - mid - right engine
    print("action_space", env.action_space)
    RL = ActorCritic(name = "Lunar", state_space = state_space, action_space = action_space)

    while True:
        state = env.reset()
        history = []
        action = RL.action(state)
        while True:
            # print(state)
            env.render()

            before_state = state
            before_action = action

            state, reward, done, info = env.step(action)
            action = RL.action(state)
            history.append([before_state, before_action, state, action, reward, done])
            
            if done:
                break
        
        RL.learn(history)


if __name__ == "__main__":
    main()