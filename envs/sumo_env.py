import os, sys
import numpy as np
import gymnasium as gym
import traci

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError("Please set SUMO_HOME")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

class SumoEnv(gym.Env):
    def __init__(self, sumo_cfg, max_steps=3600, control_interval=5,
                 use_gui=False, verbose=True, min_green=15):
        super().__init__()
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.control_interval = control_interval
        self.steps = 0
        self.use_gui = use_gui
        self.verbose = verbose
        self.min_green = min_green
        self.time_since_change = 0
        self.last_action = None
        self.total_throughput = []
        self.waiting_times = []
        self.queue_lengths = []

        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(9,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        if self.use_gui:
            sumo_binary = "sumo-gui"
        else:
            sumo_binary = "sumo"

        sumo_cfg_path = os.path.abspath(self.sumo_cfg)
        traci.start([sumo_binary, "-c", sumo_cfg_path, "--no-step-log", "true"])
        self.steps = 0
        self.episode_throughput = 0
        self.step_throughput = []
        self.steps = 0

        self.episode_throughput = 0
        self.step_throughput = []

        self.queue_lengths = []
        self.waiting_times = []

        self.last_total_wait = 0
        self.action_count = [0] * self.action_space.n
        return self._get_state()

    def step(self, action):
        self._apply_action(action)
        for _ in range(self.control_interval):
            traci.simulationStep()
            self.steps += 1

            self.time_since_change += 1

            self._log_metrics()

        state = self._get_state()
        reward = self._get_reward()
        terminated = self.steps >= self.max_steps
        truncated = False
        if terminated:
            self._finalize_metrics()
            traci.close()

        if self.verbose:
            print(f"[Step {self.steps}] Action={action}, Reward={reward:.2f}, Done={terminated}")

        return state, reward, terminated, truncated, {}

    def _log_metrics(self):
        tls = traci.trafficlight.getIDList()[0]
        lanes = traci.trafficlight.getControlledLanes(tls)

        q = [traci.lane.getLastStepVehicleNumber(l) for l in lanes]
        self.queue_lengths.append(np.mean(q))

        veh_ids = traci.vehicle.getIDList()
        if veh_ids:
            w = [traci.vehicle.getAccumulatedWaitingTime(v) for v in veh_ids]
            self.waiting_times.append(np.mean(w))

        arrived_now = traci.simulation.getArrivedNumber()
        self.episode_throughput += arrived_now
        self.step_throughput.append(arrived_now)

    def _finalize_metrics(self):
        self.throughput = self.episode_throughput
        self.avg_queue = float(np.mean(self.queue_lengths)) if self.queue_lengths else 0
        self.avg_wait = float(np.mean(self.waiting_times)) if self.waiting_times else 0

    def _get_state(self):
        tls = traci.trafficlight.getIDList()[0]
        lanes = traci.trafficlight.getControlledLanes(tls)
        q = [traci.lane.getLastStepVehicleNumber(l) for l in lanes]
        w = [traci.lane.getWaitingTime(l) for l in lanes]

        tsc = [self.time_since_change]

        q = np.array(q, dtype=np.float32) / 50.0
        w = np.array(w, dtype=np.float32) / 50.0

        return np.concatenate([q, w, tsc], axis=0)

    def _apply_action(self, action):
        tls = traci.trafficlight.getIDList()[0]
        current_phase = traci.trafficlight.getPhase(tls)

        if action == current_phase:
            self.time_since_change += self.control_interval
            return

        if self.time_since_change < self.min_green:
            self.time_since_change += self.control_interval
            return

        traci.trafficlight.setPhase(tls, int(action))
        self.time_since_change = 0
        self.last_action = action
    # ================================================================
    # def _apply_action(self, action):
    #     tls = traci.trafficlight.getIDList()[0]
    #     current_phase = traci.trafficlight.getPhase(tls)
    #
    #     if action == current_phase:
    #         return
    #
    #     if self.time_since_change < self.min_green:
    #         return
    #
    #     traci.trafficlight.setPhase(tls, int(action))
    #     self.time_since_change = 0
    #     self.last_action = action
#================================================================
    def _get_reward(self):
        tls = traci.trafficlight.getIDList()[0]
        lanes = traci.trafficlight.getControlledLanes(tls)

        current_wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
        throughput = traci.simulation.getArrivedNumber()

        if self.last_total_wait is None:
            delta_wait = 0
        else:
            delta_wait = self.last_total_wait - current_wait

        self.last_total_wait = current_wait

        reward = 0.01 * delta_wait + 0.1 * throughput

        if self.last_action is not None:
            reward += 0.05 / (1 + self.action_count[self.last_action])
            self.action_count[self.last_action] += 1

        reward -= 0.001 * self.time_since_change

        reward = float(np.clip(reward, -10.0, 10.0))

        return reward

    def close(self):
        try:
            traci.close()
        except:
            pass

