import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('images', exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial']

models = ['Sequential CPU\n(Single-Threaded)', 'MPI Cluster\n(4 VM Nodes)', 'OpenMP\n(8 CPU Threads)', 'CUDA Acceleration\n(NVIDIA GPU)']
# Reversing the order so CUDA is at the top
models.reverse()
times = [244.12, 92.98, 30.83, 0.165]
times.reverse()
speedups = [1.0, 2.63, 7.92, 1479.48]
speedups.reverse()

# Data visualization best practice: Highlight the winner, mute the rest.
colors = ['#10B981', '#94A3B8', '#94A3B8', '#94A3B8'] # CUDA is green, rest are grey

# 1. Combined Performance Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor='#FFFFFF')

bars1 = ax1.barh(models, times, color=colors, height=0.55, edgecolor='#0F172A', linewidth=1)

ax1.set_xscale('log')
ax1.set_title('Benchmark Execution Time', fontsize=14, fontweight='bold', color='#0F172A', pad=15)
ax1.set_xlabel('Execution Time (Seconds) - Log Scale', fontsize=11, fontweight='bold', color='#475569')
ax1.grid(True, axis='x', linestyle='--', color='#E2E8F0', alpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.tick_params(axis='y', length=0)

for bar, time in zip(bars1, times):
    width = bar.get_width()
    label = f'{time:.2f} s' if time >= 1 else f'{time:.3f} s'
    ax1.text(width * 1.15, bar.get_y() + bar.get_height()/2.0, label, 
             ha='left', va='center', fontsize=11, fontweight='bold', color='#0F172A')

bars2 = ax2.barh(models, speedups, color=colors, height=0.55, edgecolor='#0F172A', linewidth=1)

ax2.set_xscale('log')
ax2.set_title('Parallel Speedup Factor', fontsize=14, fontweight='bold', color='#0F172A', pad=15)
ax2.set_xlabel('Speedup vs. Sequential - Log Scale', fontsize=11, fontweight='bold', color='#475569')
ax2.grid(True, axis='x', linestyle='--', color='#E2E8F0', alpha=0.9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(axis='y', length=0)

for bar, speedup in zip(bars2, speedups):
    width = bar.get_width()
    label = f'{speedup:.2f}x'
    ax2.text(width * 1.15, bar.get_y() + bar.get_height()/2.0, label, 
             ha='left', va='center', fontsize=11, fontweight='bold', color='#0F172A')

plt.suptitle('Performance Analysis: Matrix Multiplication (4000x4000)', fontsize=16, fontweight='bold', color='#0F172A', y=1.05)
plt.tight_layout()
plt.savefig('images/performance_comparison_charts.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
plt.close()

# 2. Standalone Execution Time Chart
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#FFFFFF')
bars = ax.barh(models, times, color=colors, height=0.55, edgecolor='#0F172A', linewidth=1)

ax.set_xscale('log')
ax.set_title('Matrix Multiplication (4000x4000) Execution Time Comparison', fontsize=14, fontweight='bold', color='#0F172A', pad=15)
ax.set_xlabel('Execution Time (Seconds) - Log Scale', fontsize=11, fontweight='bold', color='#475569')
ax.grid(True, axis='x', linestyle='--', color='#E2E8F0', alpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)

for bar, time in zip(bars, times):
    width = bar.get_width()
    label = f'{time:.2f} s' if time >= 1 else f'{time:.3f} s'
    ax.text(width * 1.15, bar.get_y() + bar.get_height()/2.0, label, 
            ha='left', va='center', fontsize=11, fontweight='bold', color='#0F172A')

plt.tight_layout()
plt.savefig('images/execution_time_chart.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
plt.close()

# 3. Standalone Speedup Chart
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#FFFFFF')
bars = ax.barh(models, speedups, color=colors, height=0.55, edgecolor='#0F172A', linewidth=1)

ax.set_xscale('log')
ax.set_title('Parallel Speedup Factor relative to Sequential Baseline', fontsize=14, fontweight='bold', color='#0F172A', pad=15)
ax.set_xlabel('Speedup Factor (x) - Log Scale', fontsize=11, fontweight='bold', color='#475569')
ax.grid(True, axis='x', linestyle='--', color='#E2E8F0', alpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)

for bar, speedup in zip(bars, speedups):
    width = bar.get_width()
    label = f'{speedup:.2f}x'
    ax.text(width * 1.15, bar.get_y() + bar.get_height()/2.0, label, 
            ha='left', va='center', fontsize=11, fontweight='bold', color='#0F172A')

plt.tight_layout()
plt.savefig('images/speedup_chart.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
plt.close()
