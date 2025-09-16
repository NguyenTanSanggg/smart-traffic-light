import traci

sumo_cfg = "configs/floww.sumocfg"
traci.start(["sumo-gui", "-c", sumo_cfg])

for step in range(3600):
    traci.simulationStep()
traci.close()
