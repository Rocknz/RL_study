# from Reinforcement import DQN
from Reinforcement.ActorCritic import ActorCritic
import numpy
import gym
import torch
from threading import Thread, Lock

def simulation(index, mutex, history, steps, RL):
    env = gym.make("LunarLanderContinuous-v2")
    state = env.reset()
    history = []
    action = RL.action(state)
    step = 0
    while True:
        # print(state)
        # env.render()

        before_state = state
        before_action = action

        state, reward, done, info = env.step(action)
        action = RL.action(state)
        mutex.acquire()
        history.append([before_state, before_action, state, action, reward, done])
        mutex.release()

        step += 1
        if done:
            break

    steps[index] = step

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
    mutex = Lock()
    while True:
        threads = []
        steps = []
        simulation_num = 500
        history = []
        print("sim_start")
        for _ in range(simulation_num):
            steps.append(0)

        for index in range(simulation_num):
            step = 0
            thread = Thread(target=simulation, args=(index, mutex, history, steps, RL))
            thread.start()
            threads.append(thread)


        for thread in threads:
            thread.join()

        N = 0
        avg_step = 0
        for step in steps:
            avg_step += step
            N += 1

        avg_step /= N
        print("avg_step {}".format(avg_step))
        RL.learn(history, num = 500)

if __name__ == "__main__":
    main()