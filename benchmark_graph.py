import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('benchmark_results.csv')
averaged_data = df.groupby(['instance_count', 'mesh_variant'])['avg_ms_per_frame'].mean().reset_index()

instance_counts = sorted(averaged_data["instance_count"].unique())
x_positions = np.arange(len(instance_counts))

plt.figure(figsize=(12, 6))

A_values = averaged_data[averaged_data["mesh_variant"] == "A"].set_index("instance_count")["avg_ms_per_frame"].reindex(instance_counts).values
B_values = averaged_data[averaged_data["mesh_variant"] == "B"].set_index("instance_count")["avg_ms_per_frame"].reindex(instance_counts).values

width = 0.4

# Calculate triangle counts
tris_A = np.array(instance_counts) * 1174
tris_B = np.array(instance_counts) * 886

# Function to format numbers with k and M suffixes
def format_number(x):
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x/1_000:.0f}k'
    return str(x)

# Create bars
bars_A = plt.bar(x_positions - width/2, A_values, width=width, label="Mesh A (1.2k tris)", color='#FF6B6B', align='center')
bars_B = plt.bar(x_positions + width/2, B_values, width=width, label="Mesh B (0.9k tris)", color='#4ECDC4', align='center')

# Add triangle count labels on top of each bar
def add_value_labels(bars, tris):
    for bar, tri_count in zip(bars, tris):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                f'{format_number(tri_count)}\ntris',
                ha='center', va='bottom', fontsize=6)

add_value_labels(bars_A, tris_A)
add_value_labels(bars_B, tris_B)

plt.ylabel("Average ms per frame")
plt.xlabel("Instance Count")
plt.title("Performance Comparison: Mesh Variant A vs B\nAveraged across all test runs")
plt.legend()

# Format x-axis labels with k suffix for thousands
plt.xticks(x_positions, [format_number(x) for x in instance_counts], rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add some padding to the top of the graph to fit the labels
plt.margins(y=0.2)

plt.tight_layout()
plt.show() 