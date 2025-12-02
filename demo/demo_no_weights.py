# import traci
#
# sumo_cfg = "../config/config.sumocfg"
# traci.start(["sumo-gui", "-c", sumo_cfg])
#
# for step in range(3600):
#     traci.simulationStep()
# traci.close()
import traci
import csv
import numpy as np

sumo_cfg = "../config/config.sumocfg"
output_file = "fixed_results.csv"

traci.start(["sumo-gui", "-c", sumo_cfg, "--no-step-log", "true"])

lanes = traci.trafficlight.getControlledLanes(traci.trafficlight.getIDList()[0])
vehicles_seen = set()
waiting_times = []
queue_lengths = []
throughput_log = []

for step in range(3600):
    traci.simulationStep()

    throughput = traci.simulation.getArrivedNumber()
    throughput_log.append(throughput)

    q = [traci.lane.getLastStepVehicleNumber(l) for l in lanes]
    queue_lengths.append(np.mean(q))

    veh_ids = traci.vehicle.getIDList()
    if len(veh_ids) > 0:
        w = [traci.vehicle.getAccumulatedWaitingTime(v) for v in veh_ids]
        waiting_times.append(np.mean(w))

traci.close()

result = {
    "throughput_total": np.sum(throughput_log),
    "avg_waiting_time": np.mean(waiting_times),
    "avg_queue_length": np.mean(queue_lengths),
}

print("=== Fixed-time Controller Results ===")
print(result)

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["throughput_total", "avg_waiting_time", "avg_queue_length"])
    writer.writerow([result["throughput_total"], result["avg_waiting_time"], result["avg_queue_length"]])
