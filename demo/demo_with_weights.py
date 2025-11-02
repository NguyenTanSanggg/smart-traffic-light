import numpy as np
import csv
from envs.sumo_env import SumoEnv
from agents.dqn import DQN

def demo():
    env = SumoEnv("../config/config.sumocfg", use_gui=True, verbose=False)

    s = env.reset()
    state_dim = s.shape[0]
    action_dim = env.action_space.n

    agent = DQN(state_dim, action_dim)
    agent.q(np.zeros((1, state_dim), dtype=np.float32))

    try:
        agent.q.load_weights("../weights/dqn_weights.weights.h5")
        print("Loaded trained weights")
    except:
        print("No weights found, running with random policy")

    done = False
    while not done:
        a = agent.select_action_greedy(s)
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

    results = {
        "throughput_total": env.throughput,
        "avg_waiting_time": env.avg_wait,
        "avg_queue_length": env.avg_queue
    }

    print("=== DQN Controller Results ===")
    print(results)

    with open("dqn_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["throughput_total", "avg_waiting_time", "avg_queue_length"])
        writer.writerow([results["throughput_total"], results["avg_waiting_time"], results["avg_queue_length"]])

    env.close()

if __name__ == "__main__":
    demo()
