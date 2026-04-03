#!/usr/bin/env python3
"""Figure 6: Virtual Accelerator Power Modeling Procedure (English)"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

def box(ax, x, y, w, h, text, fc='#E3F2FD', ec='#1565C0', fs=8, fw='normal', tc='black'):
    p = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.02",
                       facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs,
            fontweight=fw, color=tc, linespacing=1.3)

def arr(ax, x1, y1, x2, y2, c='#333', lw=1.2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw))

fig, ax = plt.subplots(figsize=(18, 13))
ax.set_xlim(0, 18); ax.set_ylim(0, 13); ax.axis('off'); ax.set_aspect('equal')

# Title
ax.text(9, 12.7, 'Virtual Accelerator Power Modeling Procedure',
        fontsize=16, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333', lw=2))

# ===== LEFT: Constant + Static + Idle (Phase 1) =====
ax.text(2.2, 12.1, 'Phase 1: Constant Power', fontsize=10, fontweight='bold',
        color='#B71C1C', ha='center')

box(ax, 0.3, 11.2, 3.8, 0.7, 'DVFS Sweep (A100 reference)\nSM clock: 210~1410 MHz (8 steps)\nMemory clock: 1593 MHz (FIXED)',
    fc='#FFCDD2', ec='#C62828', fs=7.5, fw='bold')
arr(ax, 2.2, 11.2, 2.2, 10.95)
box(ax, 0.3, 10.2, 3.8, 0.7, "Cubic Fitting: P = B'C'f_sm^3 + T'f_sm + P_const'\nPearson r > 0.99 expected",
    fc='#FFCDD2', ec='#C62828', fs=7.5)
arr(ax, 2.2, 10.2, 2.2, 9.95)
box(ax, 0.5, 9.5, 3.4, 0.6, "P_const' = y-intercept at f=0\n(includes HBM2e fixed power)",
    fc='#EF9A9A', ec='#C62828', fs=7.5, fw='bold')

ax.text(2.2, 9.0, 'Phase 1: Static + Idle Power', fontsize=10, fontweight='bold',
        color='#E65100', ha='center')

box(ax, 0.3, 8.1, 3.8, 0.7, 'uBench: vary active threads (y=1~32)\nMeasure firstLane & addLane per category\nInference: 4 categories (not 9)',
    fc='#FFE0B2', ec='#E65100', fs=7.5)
arr(ax, 2.2, 8.1, 2.2, 7.85)
box(ax, 0.3, 7.2, 3.8, 0.7, 'Linear Static Model (Eq.4)\nP_static = firstLane + addLane*(y-1)\n* per_active_core ratio',
    fc='#FFE0B2', ec='#E65100', fs=7.5)
arr(ax, 2.2, 7.2, 2.2, 6.95)
box(ax, 0.3, 6.3, 3.8, 0.7, 'uBench: vary active SMs (k=1~N_sm)\nIdle SM Power: P_perIdleSM = geomean\nInference Accel-B: N_sm = 216',
    fc='#FFE0B2', ec='#E65100', fs=7.5)

# Arrows to center
arr(ax, 4.1, 9.5, 4.7, 9.0, c='#C62828', lw=1.5)
arr(ax, 4.1, 7.2, 4.7, 7.0, c='#E65100', lw=1.5)
arr(ax, 4.1, 6.3, 4.7, 5.8, c='#E65100', lw=1.5)

# ===== CENTER: uBench + PTX Sim + Mapping (Phase 2) =====
ax.text(7.0, 12.1, 'Phase 2: uBench + Performance Modeling', fontsize=10,
        fontweight='bold', color='#1B5E20', ha='center')

box(ax, 4.8, 11.0, 4.4, 0.8,
    'uBenchmarks (60~100 kernels)\nEach isolates one HW component\nInference: remove FP64/SFU benchmarks',
    fc='#C8E6C9', ec='#2E7D32', fs=7.5)
arr(ax, 7.0, 11.0, 7.0, 10.55)

box(ax, 4.8, 9.7, 4.4, 0.8,
    'Performance Modeling: PTX Mode\nnvcc -ptx -> GPGPU-Sim emulation\ngpgpusim.config = virtual accel specs\n(No real GPU required!)',
    fc='#C8E6C9', ec='#2E7D32', fs=7.5, fw='bold')
arr(ax, 7.0, 9.7, 7.0, 9.25)

# Two sub-boxes
box(ax, 5.0, 8.6, 2.0, 0.6, 'PTX SIM\nAccel-Sim\n(No GPU!)',
    fc='#A5D6A7', ec='#1B5E20', fs=7.5, fw='bold')
box(ax, 7.2, 8.6, 2.0, 0.6, 'HW Profiling\nnvidia-smi/NVML\n(A100 ref data)',
    fc='#A5D6A7', ec='#1B5E20', fs=7.5)

ax.text(7.0, 8.25, 'Activity factors, #SMs, #lanes, instr mix, cycles, V_dd, f',
        fontsize=7, color='#1B5E20', fontstyle='italic', ha='center')
arr(ax, 7.0, 8.25, 7.0, 7.85)

box(ax, 4.8, 7.1, 4.4, 0.7,
    'PTX -> Power Component Map\n"fma.f32" -> FP_MUL_OP | "mad.s32" -> INT_MUL_OP\nInference: DP___OP, FP_SIN_OP unused (=0)',
    fc='#DCEDC8', ec='#558B2F', fs=7.5)
arr(ax, 7.0, 7.1, 7.0, 6.65)

box(ax, 4.8, 5.8, 4.4, 0.8,
    'AccelWattch Engine\nP_total = P_dyn + P_static*k\n        + P_idle*(N_sm - k) + P_const\'',
    fc='#B2DFDB', ec='#00695C', fs=8.5, fw='bold')

# ===== RIGHT: QP + Validation (Phase 3-4) =====
ax.text(13.5, 12.1, 'Phase 3: QP Optimization', fontsize=10,
        fontweight='bold', color='#1A237E', ha='center')

box(ax, 10.8, 11.0, 4.4, 0.8,
    'QP Solver (Python/cvxpy)\nmin ||Ax - b||^2 + lambda*||x||^2\nInference: 20-dim (general: 31-dim)',
    fc='#C5CAE9', ec='#283593', fs=7.5, fw='bold')
arr(ax, 13.0, 11.0, 13.0, 10.55)

box(ax, 10.8, 9.8, 4.4, 0.6,
    'Constraints\n0.1 <= x_i <= 1000 | x_idle = x_const = 1\nInference: 8 constraints (remove FP64)',
    fc='#C5CAE9', ec='#283593', fs=7.5)
arr(ax, 13.0, 9.8, 13.0, 9.5)

box(ax, 10.8, 8.8, 4.4, 0.6,
    'New Scaling Factors -> XML Update\nIterate 3~5 times until convergence',
    fc='#9FA8DA', ec='#283593', fs=8, fw='bold')

# Iteration arrow
ax.annotate('', xy=(10.3, 11.4), xytext=(10.3, 8.8),
            arrowprops=dict(arrowstyle='->', color='#283593', lw=1.5,
                           connectionstyle='arc3,rad=0.4'))
ax.text(9.8, 10.1, 'Iterate', fontsize=8, color='#283593', ha='center', fontstyle='italic')

# Diamond
from matplotlib.patches import Polygon
diamond = Polygon([[13.0, 8.45], [13.7, 8.1], [13.0, 7.75], [12.3, 8.1]],
                  facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=1.5)
ax.add_patch(diamond)
ax.text(13.0, 8.1, 'Better?', fontsize=8, ha='center', va='center', fontweight='bold')
arr(ax, 13.0, 8.8, 13.0, 8.5, c='#283593')
ax.text(12.0, 7.95, 'Yes', fontsize=8, color='#2E7D32', fontweight='bold')
ax.text(14.0, 7.95, 'No', fontsize=8, color='#C62828', fontweight='bold')
arr(ax, 12.3, 8.1, 10.3, 9.0, c='#2E7D32', lw=1.5)

# Validation
ax.text(13.5, 7.2, 'Phase 4: Validation', fontsize=10,
        fontweight='bold', color='#4A148C', ha='center')

box(ax, 10.8, 6.2, 4.4, 0.7,
    'Validation Kernels (independent test set)\nMAPE = mean(|P_sim - P_hw| / P_hw) * 100%\nTarget: < 15% for DSE',
    fc='#E1BEE7', ec='#6A1B9A', fs=7.5)
arr(ax, 13.7, 8.1, 15.2, 8.1, c='#C62828')
arr(ax, 15.2, 8.1, 15.2, 6.95, c='#C62828')
arr(ax, 15.2, 6.95, 15.2, 6.9, c='#C62828')

box(ax, 10.8, 5.1, 4.4, 0.7,
    'Technology Scaling\ncacti/technology.cc parameter addition\n7nm: QP absorbs | 2nm: MUST add GAA params',
    fc='#F3E5F5', ec='#6A1B9A', fs=7.5, fw='bold')
arr(ax, 13.0, 6.2, 13.0, 5.85)

# ===== BOTTOM: Final Output =====
box(ax, 5.0, 3.8, 8.0, 0.9,
    'Virtual Accelerator Power Model (XML + config)\nConstant + Static + Dynamic (15~17 components for inference)\nPer-component breakdown | Cycle-level power trace',
    fc='#FFF3E0', ec='#E65100', fs=9, fw='bold')
arr(ax, 7.0, 5.8, 7.0, 4.75, c='#00695C', lw=2)
arr(ax, 13.0, 5.1, 13.0, 4.75, c='#6A1B9A', lw=2)

box(ax, 4.5, 2.5, 9.0, 0.9,
    'VALIDATED Virtual Accelerator Power Model\nInference-only Accel-B: ~80W (200W TDP) | General Accel-A: ~270W (400W TDP)\nExpected MAPE: 15~25% (DSE) -> 10~15% (with QP + ML correction)',
    fc='#1B5E20', ec='#1B5E20', fs=9.5, fw='bold', tc='white')
arr(ax, 9.0, 3.8, 9.0, 3.45, c='#E65100', lw=2)

# ===== Side boxes =====
box(ax, 0.2, 2.3, 4.0, 2.3,
    'Key Modifications\nfor Virtual Accelerator\n\n'
    '* gpgpusim.config:\n  SM count, core config, cache\n'
    '* accelwattch.xml:\n  scaling factors, static params\n'
    '* technology.cc:\n  2nm GAA params (IRDS 2024)\n'
    '* PTX mode (no GPU needed)',
    fc='#FFFDE7', ec='#F57F17', fs=7.5, fw='bold')

box(ax, 13.8, 2.3, 4.0, 2.3,
    'Inference-Only Reduction\n\n'
    '* FP64 units: REMOVED\n'
    '* SFU (sin,log,sqrt): REMOVED\n'
    '* Components: 22 -> 15\n'
    '* QP dims: 31 -> 20\n'
    '* Static categories: 9 -> 4\n'
    '* Registers: 64K -> 32K\n'
    '* Power: 148W -> ~80W (-45%)',
    fc='#FFF8E1', ec='#FF8F00', fs=7.5)

# Legend
lp = [mpatches.Patch(facecolor='#FFCDD2', edgecolor='#C62828', label='Constant Power (DVFS)'),
      mpatches.Patch(facecolor='#FFE0B2', edgecolor='#E65100', label='Static + Idle Power'),
      mpatches.Patch(facecolor='#C8E6C9', edgecolor='#2E7D32', label='uBench + PTX Sim + Mapping'),
      mpatches.Patch(facecolor='#C5CAE9', edgecolor='#283593', label='QP Optimization'),
      mpatches.Patch(facecolor='#E1BEE7', edgecolor='#6A1B9A', label='Validation + Tech Scaling'),
      mpatches.Patch(facecolor='#1B5E20', label='Final Output')]
ax.legend(handles=lp, loc='upper right', fontsize=8, framealpha=0.95,
          title='Phase', title_fontsize=9, bbox_to_anchor=(1.0, 0.98))

fig.savefig(os.path.join(OUT, 'fig6_virtual_accel_procedure.png'), dpi=150)
plt.close(fig)
print("fig6_virtual_accel_procedure.png saved")
