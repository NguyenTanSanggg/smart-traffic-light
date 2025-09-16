from envs.sumo_env import SumoEnv
from agents.dqn import DQN

def train():
    env = SumoEnv("config/config.sumocfg")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(state_dim, action_dim)

    episodes = 100
    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_reward = 0
        while not done:
            a = agent.select_action(s)
            ns, r, done, _ = env.step(a)
            agent.replay.push(s, a, r, ns, done)
            agent.optimize()
            s = ns
            ep_reward += r
        agent.update_target()
        print(f"Episode {ep}, Reward: {ep_reward}")

    env.close()

if __name__ == "__main__":
    train()
