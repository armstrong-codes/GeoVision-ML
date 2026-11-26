
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Results collected from the three Locust experiments
data = {
    "API Containers": [1, 2, 3],
    "Total Requests": [572, 576, 574],
    "Failed Requests": [0, 0, 0],
    "Average Response Time (ms)": [59, 41, 84],
    "P95 Response Time (ms)": [170, 110, 230],
    "Throughput (req/s)": [4.78, 4.82, 4.81]
}

df = pd.DataFrame(data)

# Create output folder
output_dir = Path("results/load_testing")
output_dir.mkdir(parents=True, exist_ok=True)

# Save comparison as CSV
df.to_csv(output_dir / "replica_comparison.csv", index=False)

# Graph 1: Average response time
plt.figure(figsize=(8, 5))
plt.bar(df["API Containers"], df["Average Response Time (ms)"])
plt.xlabel("Number of API Containers")
plt.ylabel("Average Response Time (ms)")
plt.title("Average Response Time by Container Count")
plt.xticks([1, 2, 3])
plt.tight_layout()
plt.savefig(output_dir / "average_response_time.png")
plt.close()

# Graph 2: P95 response time
plt.figure(figsize=(8, 5))
plt.bar(df["API Containers"], df["P95 Response Time (ms)"])
plt.xlabel("Number of API Containers")
plt.ylabel("P95 Response Time (ms)")
plt.title("P95 Response Time by Container Count")
plt.xticks([1, 2, 3])
plt.tight_layout()
plt.savefig(output_dir / "p95_response_time.png")
plt.close()

# Graph 3: Throughput
plt.figure(figsize=(8, 5))
plt.bar(df["API Containers"], df["Throughput (req/s)"])
plt.xlabel("Number of API Containers")
plt.ylabel("Requests per Second")
plt.title("Throughput by Container Count")
plt.xticks([1, 2, 3])
plt.tight_layout()
plt.savefig(output_dir / "throughput.png")
plt.close()

print("Comparison completed!")
print(df)
print(f"\nFiles saved in: {output_dir}")
