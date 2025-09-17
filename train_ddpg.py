from envs.sumo_env import SumoEnv
from agents.ddpg import DDPGAgent
import numpy as np

def train():
    env = SumoEnv("config/config.sumocfg")
    state_dim = env.observation_space.shape[0]
    action_dim = 1

    agent = DDPGAgent(state_dim, action_dim)

    episodes = 100
    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_reward = 0
        while not done:
            cont_a = agent.select_action(s)
            a = int(np.clip(np.round((cont_a[0] + 1) / 2 * (env.action_space.n - 1)), 0, env.action_space.n - 1))

            ns, r, done, _ = env.step(a)

            agent.replay.push(s, cont_a, r, ns, done)

            agent.optimize()
            s = ns
            ep_reward += r

        print(f"Episode {ep}, Reward: {ep_reward}")

    env.close()
    # Lưu model
    agent.actor.save_weights("ddpg_actor.h5")
    agent.critic.save_weights("ddpg_critic.h5")

if __name__ == "__main__":
    train()
