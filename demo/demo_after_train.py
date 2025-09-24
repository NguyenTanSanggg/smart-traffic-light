import numpy as np
from envs.sumo_env import SumoEnv
from agents.dqn import DQN

def demo():
    env = SumoEnv("../config/config.sumocfg", use_gui=True)

    s = env.reset()
    state_dim = s.shape[0]
    action_dim = env.action_space.n

    agent = DQN(state_dim, action_dim)

    dummy = np.zeros((1, state_dim), dtype=np.float32)
    agent.q(dummy)
    agent.q.load_weights("../dqn_weights.weights.h5")

    done = False
    while not done:
        a = agent.select_action_greedy(s)
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

    env.close()

if __name__ == "__main__":
    demo()
