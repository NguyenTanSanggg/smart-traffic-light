import os
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from envs.sumo_env import SumoEnv
from agents.dqn import DQN


def train():
    env = SumoEnv("config/config.sumocfg", verbose=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(state_dim, action_dim)
    #dummy = np.zeros((1, state_dim), dtype=np.float32)
    #agent.q(dummy)
    agent.q.build((None, state_dim))

    weights_path = "weights/dqn_weights.weights.h5"
    log_dir = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(log_dir)

    start_episode = 0
    global_step = 0
    if os.path.exists(weights_path):
        agent.q.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    else:
        print("No previous weights found")

    try:
        agent.load_checkpoint()
    except Exception as e:
        print("No checkpoint found:")

    episodes = 100
    for ep in range(start_episode, episodes):
        s = env.reset()
        done = False
        ep_reward = 0
        ep_loss = []

        while not done:
            a = agent.select_action(s)
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            agent.replay.push(s, a, r, ns, done)
            loss = agent.optimize()

            if loss is not None:
                ep_loss.append(loss)
                writer.add_scalar("Loss/step", loss, global_step, new_style=True)

            s = ns
            ep_reward += r
            global_step += 1
            agent.update_target(tau=0.005)

        mean_loss = np.mean(ep_loss) if ep_loss else 0
        writer.add_scalar("Loss/episode", mean_loss, ep, new_style=True)
        writer.add_scalar("Reward/episode", ep_reward, ep, new_style=True)
        if hasattr(env, "avg_queue"):
            writer.add_scalar("Traffic/AverageQueue", env.avg_queue, ep)
        if hasattr(env, "avg_wait"):
            writer.add_scalar("Traffic/AverageWaitingTime", env.avg_wait, ep)
        if hasattr(env, "throughput"):
            writer.add_scalar("Traffic/Throughput", env.episode_throughput, ep)

        print(f"Episode {ep + 1}/{episodes} - Reward: {ep_reward:.2f}, Loss: {mean_loss:.6f}")

        if (ep + 1) % 5 == 0:
            agent.save_checkpoint()
            agent.q.save_weights(weights_path)
            print("Saved checkpoint.")

    env.close()
    writer.close()
    print("Training completed successfully!")


if __name__ == "__main__":
    train()
