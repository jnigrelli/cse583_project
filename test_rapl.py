# test_rapl.py
import time

RAPL_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"

def read_rapl_uj():
    with open(RAPL_PATH) as f:
        return int(f.read().strip())

# Idle baseline
print("Reading idle energy for 2 seconds...")
start = read_rapl_uj()
time.sleep(2)
end = read_rapl_uj()
idle_power_w = (end - start) / 1e6 / 2  # joules / seconds = watts
print(f"Idle power: {idle_power_w:.2f} W")

# Load baseline
print("Reading energy under load for 2 seconds...")
start = read_rapl_uj()
x = sum(i * i for i in range(10_000_000))  # busy work
end = read_rapl_uj()
load_power_w = (end - start) / 1e6 / 2
print(f"Load power: {load_power_w:.2f} W")

print(f"\nDelta (load - idle): {load_power_w - idle_power_w:.2f} W")
print("Difference detected - RAPL is working" if load_power_w > idle_power_w else "✗ No difference detected — check your RAPL path")