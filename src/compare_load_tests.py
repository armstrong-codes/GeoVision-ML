
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project results directory
RESULTS_DIR = Path("results/load_testing")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Results from the three Locust experiments
data = {
    "Users": [5, 10, 20],
    "Total Requests": [892, 699, 2177],
    "Failed Requests": [0, 0, 0],
    "Average Response Time (ms)": [39.96, 39.22, 53.75],
    "P95 Response Time (ms)": [110, 120, 150],
    "Requests per Second": [2.40, 4.95, 9.67]
}

df = pd.DataFrame(data)

# Save comparison table
csv_path = RESULTS_DIR / "load_test_comparison.csv"
df.to_csv(csv_path, index=False)

print("\nLOAD TEST COMPARISON")
print(df.to_string(index=False))

# Graph 1: Average response time
plt.figure(figsize=(8, 5))
plt.plot(
    df["Users"],
    df["Average Response Time (ms)"],
    marker="o"
)
plt.title("Average Response Time vs Number of Users")
plt.xlabel("Number of Concurrent Users")
plt.ylabel("Average Response Time (ms)")
plt.grid(True)
plt.savefig(RESULTS_DIR / "average_response_time.png")
plt.close()

# Graph 2: Requests per second
plt.figure(figsize=(8, 5))
plt.plot(
    df["Users"],
    df["Requests per Second"],
    marker="o"
)
plt.title("Throughput vs Number of Users")
plt.xlabel("Number of Concurrent Users")
plt.ylabel("Requests per Second")
plt.grid(True)
plt.savefig(RESULTS_DIR / "requests_per_second.png")
plt.close()

print("\nComparison files created successfully!")
print(f"Saved in: {RESULTS_DIR.resolve()}")
