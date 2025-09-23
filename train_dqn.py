from envs.sumo_env import SumoEnv
from agents.dqn import DQN

def train():
    env = SumoEnv("config/config.sumocfg", verbose=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(state_dim, action_dim)

    try:
        agent.q.load_weights("dqn_weights.weights.h5")
    except:
        pass

    episodes = 10
    for ep in range(episodes):
        s, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            a = agent.select_action(s)
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            agent.replay.push(s, a, r, ns, done)
            agent.optimize()
            s = ns
            ep_reward += r
        agent.update_target()
        print(f"Episode {ep}, Reward: {ep_reward}")

        agent.q.save_weights("dqn_weights.weights.h5")

    env.close()

if __name__ == "__main__":
    train()
