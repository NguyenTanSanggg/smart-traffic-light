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

        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(8,), dtype=np.float32
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

        return self._get_state()

    def step(self, action):
        self._apply_action(action)
        for _ in range(self.control_interval):
            traci.simulationStep()
            self.steps += 1

        state = self._get_state()
        reward = self._get_reward()
        terminated = self.steps >= self.max_steps
        truncated = False
        if terminated:
            traci.close()

        if self.verbose:
            print(f"[Step {self.steps}] Action={action}, Reward={reward:.2f}, Done={terminated}")

        return state, reward, terminated, truncated, {}

    def _get_state(self):
        tls = traci.trafficlight.getIDList()[0]
        lanes = traci.trafficlight.getControlledLanes(tls)
        q = [traci.lane.getLastStepVehicleNumber(l) for l in lanes]
        w = [traci.lane.getWaitingTime(l) for l in lanes]
        return np.array(q + w, dtype=np.float32)

    def _apply_action(self, action):
        tls = traci.trafficlight.getIDList()[0]
        if action != traci.trafficlight.getPhase(tls):
            traci.trafficlight.setPhase(tls, int(action))
            self.time_since_change = 0
            self.last_action = action

    def _get_reward(self):
        tls = traci.trafficlight.getIDList()[0]
        lanes = traci.trafficlight.getControlledLanes(tls)
        total_wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
        return -total_wait

    def close(self):
        try:
            traci.close()
        except:
            pass

