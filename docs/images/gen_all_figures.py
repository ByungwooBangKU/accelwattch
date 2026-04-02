#!/usr/bin/env python3
"""07_Research_Roadmap용 모든 Figure 생성"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# ============================================================
# Figure 1: Gantt Chart — 8-month Timeline with Priority
# ============================================================
def fig1_gantt():
    fig, ax = plt.subplots(figsize=(14, 7))

    phases = [
        # (name, start_month, duration, priority, color)
        ("1.1 Env Setup & HW Access",          1, 0.5, "P0-Critical", "#d32f2f"),
        ("1.2 DVFS Sweep → P_const",           1.5, 0.5, "P0-Critical", "#d32f2f"),
        ("1.3 Static Power Measurement",        2, 0.5, "P0-Critical", "#d32f2f"),
        ("1.4 A100 XML Creation",               2, 0.5, "P0-Critical", "#d32f2f"),
        ("1.5 µBenchmark + QP → Baseline",      2.5, 0.5, "P0-Critical", "#d32f2f"),

        ("2.1 INT/FP Concurrent Model",         3, 1, "P1-High", "#f57c00"),
        ("2.2 TF32/BF16 Component Split",       3.5, 1, "P1-High", "#f57c00"),
        ("2.3 Python QP Solver (cvxpy)",        4, 0.5, "P1-High", "#f57c00"),
        ("2.4 QP Iteration → MAPE < 12%",       4.5, 0.5, "P1-High", "#f57c00"),

        ("3.1 Virtual Accelerator Design (×3)", 5, 1, "P2-Medium", "#1976d2"),
        ("3.2 Simulation & Power Prediction",   5.5, 1, "P2-Medium", "#1976d2"),
        ("3.3 Sensitivity & Pareto Analysis",   6, 1, "P2-Medium", "#1976d2"),

        ("4.1 Figures & Tables",                7, 0.5, "P3-Normal", "#388e3c"),
        ("4.2 Paper Draft",                     7, 1, "P3-Normal", "#388e3c"),
        ("4.3 Internal Review & Revision",      7.5, 1, "P3-Normal", "#388e3c"),
        ("4.4 Submission",                      8, 0.5, "P3-Normal", "#388e3c"),
    ]

    yticks = []
    ylabels = []
    for i, (name, start, dur, prio, color) in enumerate(reversed(phases)):
        y = i
        ax.barh(y, dur, left=start, height=0.6, color=color, edgecolor='white', linewidth=0.5)
        yticks.append(y)
        ylabels.append(name)

    # Milestones
    milestones = [
        (3, "MS1\nMAPE<15%", "#d32f2f"),
        (5, "MS2\nMAPE<12%", "#f57c00"),
        (6, "MS3\nDSE Report", "#1976d2"),
        (7.5, "MS4\nDraft", "#388e3c"),
        (8.5, "MS5\nSubmit", "#388e3c"),
    ]
    for mx, label, color in milestones:
        ax.plot(mx, len(phases)//2, marker='v', markersize=0, color=color)
        ax.axvline(x=mx, color=color, linestyle='--', alpha=0.4, linewidth=1)
        ax.text(mx, len(phases)-0.5, label, ha='center', va='bottom', fontsize=8,
                fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.9))

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_xlim(0.5, 9)
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels([f"M{i}" for i in range(1, 9)])
    ax.set_title("8-Month Research Roadmap — Gantt Chart with Priority", fontsize=13, fontweight='bold')

    # Legend
    legend_patches = [
        mpatches.Patch(color="#d32f2f", label="P0-Critical (Must have)"),
        mpatches.Patch(color="#f57c00", label="P1-High (Major improvement)"),
        mpatches.Patch(color="#1976d2", label="P2-Medium (Core contribution)"),
        mpatches.Patch(color="#388e3c", label="P3-Normal (Publication)"),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
    fig.savefig(os.path.join(OUT, 'fig1_gantt_timeline.png'))
    plt.close(fig)
    print("  fig1_gantt_timeline.png")

# ============================================================
# Figure 2: MAPE Improvement Trajectory
# ============================================================
def fig2_mape():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    stages = ["GPUWattch\n(Volta)", "AccelWattch\nV100 Baseline", "Phase1\nA100 Baseline",
              "Phase2\n+Components", "Phase2\n+ML Residual", "Phase3\nVirtual Accel"]
    mape = [219, 9.2, 14, 11, 8, 10]
    colors = ["#bdbdbd", "#66bb6a", "#ef5350", "#ffa726", "#42a5f5", "#ab47bc"]

    bars = ax.bar(range(len(stages)), mape, color=colors, edgecolor='white', width=0.65)
    for bar, val in zip(bars, mape):
        ypos = bar.get_height() + 1.5
        if val > 50:
            ypos = 30
            ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{val}%", ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{val}%", ha='center', va='bottom',
                    fontweight='bold', fontsize=11)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("MAPE (%)", fontsize=12)
    ax.set_ylim(0, 35)
    ax.set_title("Expected MAPE Improvement Trajectory", fontsize=13, fontweight='bold')

    # Target line
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(5.4, 10.5, "Target: <10%", color='red', fontsize=9, fontstyle='italic')

    # Break y-axis indicator for GPUWattch
    ax.annotate("219%", xy=(0, 30), fontsize=9, color="#757575", ha='center')
    bars[0].set_height(30)

    ax.grid(axis='y', alpha=0.3)
    fig.savefig(os.path.join(OUT, 'fig2_mape_trajectory.png'))
    plt.close(fig)
    print("  fig2_mape_trajectory.png")

# ============================================================
# Figure 3: Virtual Accelerator Power Breakdown Comparison
# ============================================================
def fig3_power_breakdown():
    fig, ax = plt.subplots(figsize=(11, 6))

    configs = ["A100\n(Baseline)", "Accel-A\n(AI Train)", "Accel-B\n(AI Infer)", "Accel-C\n(HPC)"]

    # Component breakdown (estimated W)
    components = {
        "Tensor Core":   [35, 70, 10, 20],
        "FP32/INT ALU":  [30, 15, 15, 25],
        "FP64":          [5,  3,  2,  45],
        "Register File": [20, 22, 25, 20],
        "Caches (L1+L2)": [25, 35, 20, 30],
        "DRAM+MC":       [15, 18, 12, 25],
        "NoC+Sched+Pipe":[15, 17, 20, 15],
        "Static Power":  [40, 35, 25, 45],
        "Constant Power": [55, 55, 30, 55],
    }

    colors = ["#e53935", "#fb8c00", "#fdd835", "#43a047",
              "#1e88e5", "#8e24aa", "#6d4c41", "#78909c", "#455a64"]

    x = np.arange(len(configs))
    width = 0.55
    bottom = np.zeros(len(configs))

    for (comp, vals), color in zip(components.items(), colors):
        ax.bar(x, vals, width, bottom=bottom, label=comp, color=color, edgecolor='white', linewidth=0.3)
        bottom += vals

    # Total power labels
    totals = [sum(v[i] for v in components.values()) for i in range(len(configs))]
    for i, t in enumerate(totals):
        ax.text(i, t + 3, f"{t}W", ha='center', va='bottom', fontweight='bold', fontsize=11)

    # TDP lines
    tdps = [400, 400, 200, 400]
    for i, tdp in enumerate(tdps):
        ax.plot([i-0.3, i+0.3], [tdp, tdp], 'r--', linewidth=1.5, alpha=0.7)
    ax.text(0.35, 405, "TDP", color='red', fontsize=8, fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel("Power (W)", fontsize=12)
    ax.set_title("Power Breakdown Comparison: A100 vs Virtual Accelerators", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
    ax.set_ylim(0, 440)
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(os.path.join(OUT, 'fig3_power_breakdown.png'))
    plt.close(fig)
    print("  fig3_power_breakdown.png")

# ============================================================
# Figure 4: Priority Matrix (Impact vs Effort)
# ============================================================
def fig4_priority():
    fig, ax = plt.subplots(figsize=(9, 7))

    tasks = [
        # (name, effort(1-10), impact(1-10), priority, color, size)
        ("P_const\nDVFS Sweep", 2, 9, "P0", "#d32f2f", 400),
        ("Static Power\nMeasurement", 3, 8, "P0", "#d32f2f", 350),
        ("A100 XML\nCreation", 3, 9, "P0", "#d32f2f", 400),
        ("Baseline QP\n(MATLAB)", 2, 8, "P0", "#d32f2f", 350),

        ("INT/FP\nConcurrent", 7, 10, "P1", "#f57c00", 500),
        ("TF32/BF16\nSplit", 5, 7, "P1", "#f57c00", 350),
        ("Python QP\n(cvxpy)", 4, 6, "P1", "#f57c00", 300),
        ("ML Residual\nCorrection", 5, 5, "P1", "#f57c00", 250),

        ("Virtual Accel\nDesign (×3)", 6, 9, "P2", "#1976d2", 450),
        ("Sensitivity\nAnalysis", 4, 7, "P2", "#1976d2", 350),
        ("Pareto\nFrontier", 3, 6, "P2", "#1976d2", 300),

        ("Paper\nWriting", 6, 8, "P3", "#388e3c", 400),
        ("2:4 Sparsity\nComponent", 6, 4, "P3", "#388e3c", 200),
        ("Half-warp\nImplement", 7, 5, "P3", "#388e3c", 250),
    ]

    for name, effort, impact, prio, color, size in tasks:
        ax.scatter(effort, impact, s=size, c=color, alpha=0.7, edgecolors='white', linewidth=1.5, zorder=3)
        ax.annotate(name, (effort, impact), textcoords="offset points",
                    xytext=(0, -22 if impact > 5 else 15), ha='center', fontsize=7.5, fontweight='bold')

    # Quadrant labels
    ax.text(1.5, 9.5, "Quick Wins ★", fontsize=11, fontstyle='italic', color='#2e7d32', fontweight='bold')
    ax.text(7, 9.5, "Strategic\nInvestments", fontsize=10, fontstyle='italic', color='#1565c0', fontweight='bold', ha='center')
    ax.text(1.5, 1.5, "Low Priority", fontsize=10, fontstyle='italic', color='#9e9e9e')
    ax.text(7, 1.5, "Reconsider", fontsize=10, fontstyle='italic', color='#e65100')

    ax.axhline(y=5.5, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(x=5, color='gray', linestyle=':', alpha=0.4)

    ax.set_xlabel("Effort (1=Easy → 10=Hard)", fontsize=12)
    ax.set_ylabel("Impact on MAPE / Contribution (1=Low → 10=High)", fontsize=11)
    ax.set_title("Priority Matrix: Impact vs Effort", fontsize=13, fontweight='bold')
    ax.set_xlim(0.5, 10)
    ax.set_ylim(0.5, 10.5)

    legend_patches = [
        mpatches.Patch(color="#d32f2f", label="P0-Critical"),
        mpatches.Patch(color="#f57c00", label="P1-High"),
        mpatches.Patch(color="#1976d2", label="P2-Medium"),
        mpatches.Patch(color="#388e3c", label="P3-Normal"),
    ]
    ax.legend(handles=legend_patches, loc='lower left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.savefig(os.path.join(OUT, 'fig4_priority_matrix.png'))
    plt.close(fig)
    print("  fig4_priority_matrix.png")

# ============================================================
# Figure 5: Perf/Watt Radar Chart
# ============================================================
def fig5_radar():
    categories = ['FP32\nTFLOPS/W', 'FP64\nTFLOPS/W', 'Tensor\nTFLOPS/W',
                   'Memory\nBW/W', 'Inference\nTOPS/W', 'Power\nEfficiency']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    configs = {
        'A100 (Baseline)': [0.6, 0.3, 0.7, 0.5, 0.5, 0.5],
        'Accel-A (AI Train)': [0.3, 0.15, 0.95, 0.55, 0.4, 0.65],
        'Accel-B (AI Infer)': [0.4, 0.1, 0.3, 0.4, 0.95, 0.85],
        'Accel-C (HPC)':     [0.5, 0.9, 0.4, 0.6, 0.3, 0.45],
    }
    colors = ['#43a047', '#e53935', '#1e88e5', '#ff9800']

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for (name, vals), color in zip(configs.items(), colors):
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, vals_plot, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color='gray')
    ax.set_title("Performance/Watt Profile: A100 vs Virtual Accelerators", fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=9, framealpha=0.9)
    fig.savefig(os.path.join(OUT, 'fig5_radar_perfwatt.png'))
    plt.close(fig)
    print("  fig5_radar_perfwatt.png")


if __name__ == '__main__':
    print("Generating figures...")
    fig1_gantt()
    fig2_mape()
    fig3_power_breakdown()
    fig4_priority()
    fig5_radar()
    print("Done. All figures saved to:", OUT)
