import traci
import numpy as np
from envs.sumo_env import SumoEnv
from agents.dqn import DQN

def demo():
    env = SumoEnv("config/config.sumocfg")
    s = env.reset()
    state_dim = s.shape[0]
    action_dim = env.action_space.n

    agent = DQN(state_dim, action_dim)
    agent.q.load_weights("dqn_weights.h5")

    done = False
    while not done:
        a = agent.select_action(s)
        s, r, done, _ = env.step(a)

    env.close()

if __name__ == "__main__":
    demo()
