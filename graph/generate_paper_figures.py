"""
R-AML-UAN — graphs

"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ── Paths — portable, works on any machine ────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = BASE + os.sep

# ── Load CSV data ─────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, 'hybrid_drl_results.csv'))
dq = pd.read_csv(os.path.join(BASE, 'hybrid_qvalues.csv'))
dd = pd.read_csv(os.path.join(BASE, 'hybrid_decoy_log.csv'))

df['CumReward'] = df['Reward'].cumsum()
df['MovingAvg'] = df['Reward'].rolling(50, min_periods=1).mean()

# ── Simulation constants (from CSV output) ────────────────────────
EXPLORE_STEPS = 422
EXPLOIT_STEPS = 2566
TOTAL_REWARD  = 26321.1
AVG_REWARD    = 8.81

# ── Color palette (IEEE-friendly, readable in grayscale print) ────
C = {
    'optical':   '#2166AC',
    'acoustic':  '#B2182B',
    'decoy':     '#4DAC26',
    'green':     '#1B7837',
    'red':       '#A50026',
    'gray':      '#636363',
    'lightgray': '#BDBDBD',
    'gold':      '#8C6D31',
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'grid.color':       '#CCCCCC',
    'grid.linewidth':   0.6,
    'axes.labelsize':   11,
    'axes.titlesize':   12,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  9,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})

# ══════════════════════════════════════════════════════════════════
# FIG 1 — Packet Mode Distribution
# ══════════════════════════════════════════════════════════════════
print("Generating Fig01...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle('Fig 1: R-AML-UAN Packet Mode Distribution\n'
             'Total: 2988 Packets  |  100% PDR  |  0 Drops',
             fontsize=12, fontweight='bold')

counts = df['Mode'].value_counts()
opt = counts.get('OPTICAL', 0)
acu = counts.get('ACOUSTIC', 0)
dec = counts.get('DECOY', 0)

sizes = [opt, acu, dec]
clrs  = [C['optical'], C['acoustic'], C['decoy']]
lbls  = [f'OPTICAL\n{opt} ({opt/2988*100:.1f}%)',
         f'ACOUSTIC\n{acu} ({acu/2988*100:.1f}%)',
         f'DECOY\n{dec} ({dec/2988*100:.1f}%)']

wedges, texts, autotexts = ax1.pie(
    sizes, labels=lbls, colors=clrs,
    autopct='%1.1f%%', startangle=140, pctdistance=0.78,
    wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
    textprops={'fontsize': 9.5})
for at in autotexts:
    at.set_fontsize(9); at.set_fontweight('bold'); at.set_color('white')
ax1.set_title('Mode Distribution (Pie)', fontsize=11, pad=8)

modes_sorted = ['OPTICAL', 'ACOUSTIC', 'DECOY']
vals_sorted  = [opt, acu, dec]
bars = ax2.bar(modes_sorted, vals_sorted, color=clrs,
               width=0.45, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, vals_sorted):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 12,
             f'{v}\n({v/2988*100:.1f}%)',
             ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax2.set_ylabel('Packet Count')
ax2.set_title('Mode Distribution (Bar Chart)', fontsize=11)
ax2.set_ylim(0, 1500)
ax2.yaxis.grid(True, alpha=0.3); ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT + 'Fig01_PacketDistribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: Fig01_PacketDistribution.png")

# ══════════════════════════════════════════════════════════════════
# FIG 2 — DRL Learning Curve
# ══════════════════════════════════════════════════════════════════
print("Generating Fig02...")
fig, ax = plt.subplots(figsize=(11, 4.5))
ax2b = ax.twinx()
idx  = np.arange(1, len(df) + 1)

ax.fill_between(idx, df['CumReward'], alpha=0.12, color=C['optical'])
ax.plot(idx, df['CumReward'], color=C['optical'], linewidth=2,
        label='Cumulative Reward')
ax2b.plot(idx, df['MovingAvg'], color=C['acoustic'], linewidth=1.5,
          label='Moving Avg (50-pkt)')

eps_boundary = int((dq['Epsilon'] > 0.1).sum())

ax.axvspan(0, eps_boundary, alpha=0.05, color='red')
ax.axvspan(eps_boundary, len(df), alpha=0.05, color='green')
ax.axvline(eps_boundary, color='gray', linestyle='--', linewidth=1, alpha=0.6)

ax.text(eps_boundary * 0.35, df['CumReward'].max() * 0.15,
        'Explore\nPhase', ha='center', color='#B2182B', fontsize=9, alpha=0.8)
ax.text(eps_boundary + (len(df) - eps_boundary) * 0.4, df['CumReward'].max() * 0.15,
        'Exploit Phase', ha='center', color='#1B7837', fontsize=9, alpha=0.8)
ax.annotate(f'Total: {int(df["CumReward"].iloc[-1]):,}',
            xy=(len(df) * 0.78, df['CumReward'].iloc[-1] * 0.86),
            fontsize=10, color=C['optical'], fontweight='bold')

ax.set_xlabel('Packet Index (Steps)')
ax.set_ylabel('Cumulative Reward', color=C['optical'])
ax2b.set_ylabel('Moving Average Reward', color=C['acoustic'])
ax.set_title(
    f'Fig 2: DRL Learning Curve — Cumulative & Moving-Average Reward\n'
    f'α=0.15, γ=0.90  |  Total Reward = {TOTAL_REWARD:,.1f}  |  AvgRew = {AVG_REWARD:.2f}',
    fontsize=12, fontweight='bold')

lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2b.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, loc='upper left', fontsize=9, framealpha=0.8)

plt.tight_layout()
plt.savefig(OUT + 'Fig02_LearningCurve.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: Fig02_LearningCurve.png")

# ══════════════════════════════════════════════════════════════════
# FIG 3 — Epsilon Decay
# ══════════════════════════════════════════════════════════════════
print("Generating Fig03...")
fig, ax = plt.subplots(figsize=(10, 4.5))
idx = np.arange(1, len(dq) + 1)
eps_vis_boundary = int((dq['Epsilon'] > 0.1).sum())

ax.fill_between(idx[:eps_vis_boundary],
                dq['Epsilon'].iloc[:eps_vis_boundary],
                alpha=0.12, color='red')
ax.fill_between(idx[eps_vis_boundary:],
                dq['Epsilon'].iloc[eps_vis_boundary:],
                alpha=0.08, color='green')
ax.plot(idx, dq['Epsilon'], color='#4D4D4D', linewidth=2.2)
ax.axhline(y=0.08, color='black', linestyle='--', linewidth=1.5, label='ε_min = 0.08')
ax.axhline(y=0.5,  color=C['lightgray'], linestyle=':', linewidth=1)

ax.text(eps_vis_boundary * 0.38, 0.62, 'High Explore',
        ha='center', color='#B2182B', fontsize=10)
ax.text(eps_vis_boundary + (len(dq) - eps_vis_boundary) * 0.45, 0.62,
        'Exploit Dominant', ha='center', color='#1B7837', fontsize=10)

ax.annotate(
    f'Start: ε = {dq["Epsilon"].max():.2f}\n'
    f'Decay: ×0.985 per 5 steps\n'
    f'Final: ε = {dq["Epsilon"].min():.2f}',
    xy=(60, 0.80), fontsize=9.5,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
              edgecolor='#AAAAAA', alpha=0.9))

ax.set_xlabel('Packet Index (Steps)')
ax.set_ylabel('Exploration Rate ε')
ax.set_title(
    f'Fig 3: DRL ε-Greedy Exploration Rate Decay\n'
    f'Explore steps: {EXPLORE_STEPS}  |  Exploit steps: {EXPLOIT_STEPS}  |  Final ε = 0.08',
    fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(OUT + 'Fig03_EpsilonDecay.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: Fig03_EpsilonDecay.png")

# ══════════════════════════════════════════════════════════════════
# FIG 4 — Q-Value Convergence (4 representative states)
# ══════════════════════════════════════════════════════════════════
print("Generating Fig04...")
pairs = [
    (0, 'CLEAR',    '75 m',    'DRL learns → OPTICAL'),
    (1, 'PRISTINE', '201.6 m', 'DRL learns → OPTICAL'),
    (2, 'MURKY',    '90 m',    'DRL learns → ACOUSTIC'),
    (4, 'CLEAR',    '85 m',    'DRL learns → OPTICAL'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    'Fig 4: Q-Value Convergence — DRL Learns Correct Policy per Node & Water State\n'
    'Bellman Equation: Q(s,a) += α[r + γ · maxQ(s′) − Q(s,a)]',
    fontsize=12, fontweight='bold')

for ax, (node, water, dist, verdict) in zip(axes.flat, pairs):
    sub = dq[(dq['Src'] == node) & (dq['Water'] == water)].reset_index(drop=True)
    if len(sub) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes)
        continue
    idx2 = np.arange(len(sub))
    ax.plot(idx2, sub['Qopt'],   color=C['optical'],  linewidth=2,   label='Q_opt',   zorder=3)
    ax.plot(idx2, sub['Qacu'],   color=C['acoustic'], linewidth=2,   label='Q_acu',   zorder=3)
    ax.plot(idx2, sub['Qdecoy'], color=C['decoy'],    linewidth=1.5, label='Q_decoy',
            linestyle='--', zorder=3)

    fopt = sub['Qopt'].iloc[-1]
    facu = sub['Qacu'].iloc[-1]
    fdec = sub['Qdecoy'].iloc[-1]
    winner = ['ACOUSTIC', 'OPTICAL', 'DECOY'][np.argmax([facu, fopt, fdec])]
    winner_col = {'OPTICAL': C['optical'], 'ACOUSTIC': C['acoustic'], 'DECOY': C['decoy']}[winner]

    ax.annotate(f'→ {winner}',
                xy=(len(sub) * 0.7, max(fopt, facu, fdec) * 0.88),
                fontsize=10, color=winner_col, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=winner_col, alpha=0.8, linewidth=0.8))
    ax.set_title(f'Node{node}→Node{node+1}  |  {water}  |  dist={dist}\n{verdict}', fontsize=10.5)
    ax.set_xlabel('Packet Index')
    ax.set_ylabel('Q-Value')
    ax.legend(loc='upper left', fontsize=8.5)

plt.tight_layout()
plt.savefig(OUT + 'Fig04_QValueConvergence.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: Fig04_QValueConvergence.png")

# ══════════════════════════════════════════════════════════════════
# FIG 5 — UDDS Results
# ══════════════════════════════════════════════════════════════════
print("Generating Fig08...")
e0_real=129; e0_dec=8;  e0_miss=762
e1_real=42;  e1_dec=7;  e1_miss=1081

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Fig 8: UDDS — Underwater Decoy-Based Deception System Results\n'
             'Stealth Score = 8.1%  |  Decoy Activation Rate = 19.2%',
             fontsize=12, fontweight='bold')

cats  = ['Real Detected', 'Decoy Hits', 'Enemy Missed']
e0v   = [e0_real, e0_dec, e0_miss]
e1v   = [e1_real, e1_dec, e1_miss]
bcols = [C['red'], C['gold'], C['green']]
x2    = np.arange(3); w2 = 0.35

for i, col in enumerate(bcols):
    b0 = ax1.bar(i - w2/2, e0v[i], w2, color=col, alpha=0.95, edgecolor='white')
    b1 = ax1.bar(i + w2/2, e1v[i], w2, color=col, alpha=0.55, edgecolor='white',
                 hatch='//', linewidth=0.5)
    for bar, v in [(b0, e0v[i]), (b1, e1v[i])]:
        ax1.text(bar[0].get_x() + bar[0].get_width()/2,
                 bar[0].get_height() + 6,
                 str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xticks(x2)
ax1.set_xticklabels(cats, fontsize=10)
ax1.set_ylabel('Event Count')
ax1.set_title('Enemy Detection Events (per submarine)')
p1 = mpatches.Patch(color='#636363', alpha=0.95, label='Enemy-0 (solid)')
p2 = mpatches.Patch(color='#636363', alpha=0.55, hatch='//', label='Enemy-1 (hatch)')
ax1.legend(handles=[p1, p2], fontsize=9)
ax1.set_axisbelow(True)

sizes2  = [e0_real+e1_real, e0_dec+e1_dec, e0_miss+e1_miss]
labels2 = [f'Real Detected\n{e0_real+e1_real}',
           f'Decoy Hits\n{e0_dec+e1_dec}',
           f'Enemy Missed\n{e0_miss+e1_miss}']
ax2.pie(sizes2, labels=labels2, colors=[C['red'], C['gold'], C['green']],
        autopct='%1.1f%%', startangle=90, pctdistance=0.75,
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
        textprops={'fontsize': 9.5})
ax2.set_title('UDDS Outcome Distribution (Combined)')

plt.tight_layout()
plt.savefig(OUT + 'Fig08_UDDSResults.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: Fig08_UDDSResults.png")

# ══════════════════════════════════════════════════════════════════
# FIG 6 — Decoy Suspicion Analysis
# ══════════════════════════════════════════════════════════════════
print("Generating Fig11...")
fig = plt.figure(figsize=(13, 9))
fig.suptitle(
    'Fig 11: Enemy Decoy Suspicion Analysis — R-AML-UAN UDDS\n'
    'How Enemy Submarines Learn to Distrust Decoy Signals Over Time',
    fontsize=12, fontweight='bold')
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

dd['TimeWindow'] = (dd['Time'] // 30) * 30
s_e0 = dd.groupby('TimeWindow')['Suspicion_E0'].mean()
s_e1 = dd.groupby('TimeWindow')['Suspicion_E1'].mean()
x_t  = s_e0.index

ax_a = fig.add_subplot(gs[0, :])
ax_a.bar(x_t - 4, s_e0.values, width=14, color=C['acoustic'],
         alpha=0.85, label='Enemy-0 avg suspicion')
ax_a.bar(x_t + 4, s_e1.values, width=14, color=C['optical'],
         alpha=0.85, label='Enemy-1 avg suspicion')
ax_a.axhline(5.0, color=C['red'], linewidth=2, linestyle='--',
             label='Threshold = 5.0 (enemy ignores decoy)')
max_susp = dd[['Suspicion_E0', 'Suspicion_E1']].values.max()
ax_a.axhline(max_susp, color=C['gray'], linewidth=1, linestyle=':',
             label=f'Max observed ≈ {max_susp:.1f}')
ax_a.set_xlabel('Simulation Time (s)')
ax_a.set_ylabel('Avg Suspicion Level\n(per 30-s window)')
ax_a.set_title('A — Average Enemy Suspicion Level in Each 30-Second Window',
               fontsize=11, fontweight='bold')
ax_a.legend(fontsize=8.5, ncol=2)
ax_a.set_ylim(0, 7.5)
ax_a.set_axisbelow(True)

bins = np.arange(0, 7.0, 0.8)

ax_b = fig.add_subplot(gs[1, 0])
ax_b.hist(dd[dd['Enemy0Deceived'] == 1]['Suspicion_E0'], bins=bins,
          color=C['green'], alpha=0.8, label='Decoy Succeeded', edgecolor='white')
ax_b.hist(dd[dd['Enemy0Deceived'] == 0]['Suspicion_E0'], bins=bins,
          color=C['gray'], alpha=0.7, label='Decoy Ignored', edgecolor='white')
ax_b.axvline(5.0, color=C['red'], linewidth=2, linestyle='--', label='Threshold = 5.0')
e0s = (dd['Enemy0Deceived'] == 1).sum()
e0i = (dd['Enemy0Deceived'] == 0).sum()
ax_b.text(0.60, 0.84, f'Succeeded: {e0s}\nIgnored:    {e0i}',
          transform=ax_b.transAxes, fontsize=9,
          bbox=dict(boxstyle='round', facecolor='white', edgecolor='#AAAAAA', alpha=0.9))
ax_b.set_title('B — Enemy-0: Suspicion vs Decoy Outcome', fontsize=10.5, fontweight='bold')
ax_b.set_xlabel('Enemy-0 Suspicion Level at Decoy Event')
ax_b.set_ylabel('Count of Decoy Events')
ax_b.legend(fontsize=8.5)
ax_b.set_axisbelow(True)

ax_c = fig.add_subplot(gs[1, 1])
ax_c.hist(dd[dd['Enemy1Deceived'] == 1]['Suspicion_E1'], bins=bins,
          color=C['green'], alpha=0.8, label='Decoy Succeeded', edgecolor='white')
ax_c.hist(dd[dd['Enemy1Deceived'] == 0]['Suspicion_E1'], bins=bins,
          color=C['gray'], alpha=0.7, label='Decoy Ignored', edgecolor='white')
ax_c.axvline(5.0, color=C['red'], linewidth=2, linestyle='--', label='Threshold = 5.0')
e1s = (dd['Enemy1Deceived'] == 1).sum()
e1i = (dd['Enemy1Deceived'] == 0).sum()
ax_c.text(0.60, 0.84, f'Succeeded: {e1s}\nIgnored:    {e1i}',
          transform=ax_c.transAxes, fontsize=9,
          bbox=dict(boxstyle='round', facecolor='white', edgecolor='#AAAAAA', alpha=0.9))
ax_c.set_title('C — Enemy-1: Suspicion vs Decoy Outcome', fontsize=10.5, fontweight='bold')
ax_c.set_xlabel('Enemy-1 Suspicion Level at Decoy Event')
ax_c.set_ylabel('Count of Decoy Events')
ax_c.legend(fontsize=8.5)
ax_c.set_axisbelow(True)

plt.savefig(OUT + 'Fig11_DecoySuspicion.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: Fig11_DecoySuspicion.png")

# ══════════════════════════════════════════════════════════════════
# FIG 7— Beer-Lambert Link Quality Curves (theoretical)
# ══════════════════════════════════════════════════════════════════
print("Generating Fig12...")
dist_range = np.linspace(0, 450, 600)
water_cfg = {
    'PRISTINE': {'opt': 0.5, 'acu': 1.0,  'col': '#1A9641', 'ls': '-'},
    'CLEAR':    {'opt': 1.0, 'acu': 1.0,  'col': '#2166AC', 'ls': '-'},
    'HAZY':     {'opt': 1.2, 'acu': 1.05, 'col': '#74ADD1', 'ls': '--'},
    'COASTAL':  {'opt': 1.8, 'acu': 1.1,  'col': '#F46D43', 'ls': '--'},
    'MURKY':    {'opt': 3.0, 'acu': 1.3,  'col': '#D73027', 'ls': '-.'},
    'THERMAL':  {'opt': 2.0, 'acu': 2.5,  'col': '#762A83', 'ls': '-.'},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Fig 12: Beer-Lambert Link Quality vs Distance\n'
             'Optical: LQ = exp(−α × d / 100)  |  Acoustic: LQ = exp(−α × d / 450)',
             fontsize=12, fontweight='bold')

for wname, wc in water_cfg.items():
    ax1.plot(dist_range, np.exp(-wc['opt'] * dist_range / 100),
             color=wc['col'], linewidth=1.8, linestyle=wc['ls'],
             label=wname, alpha=0.9)

ax1.axvline(100, color='black', linestyle=':', linewidth=1.5, label='Optical Range = 100 m')
ax1.axhline(0.30, color=C['gray'], linestyle=':', linewidth=1.2, label='LQ Threshold = 0.30')
ax1.set_xlim(0, 200)
ax1.set_title('Optical LQ vs Distance')
ax1.set_xlabel('Distance (m)')
ax1.set_ylabel('Link Quality (LQ)')
ax1.legend(fontsize=8.5)
ax1.set_ylim(0, 1.05)
ax1.set_axisbelow(True)

for wname, wc in water_cfg.items():
    ax2.plot(dist_range, np.exp(-wc['acu'] * dist_range / 450),
             color=wc['col'], linewidth=1.8, linestyle=wc['ls'],
             label=wname, alpha=0.9)

ax2.axvline(450, color='black', linestyle=':', linewidth=1.5, label='Acoustic Range = 450 m')
ax2.axhline(0.20, color=C['gray'], linestyle=':', linewidth=1.2, label='LQ Threshold = 0.20')
ax2.set_title('Acoustic LQ vs Distance')
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('Link Quality (LQ)')
ax2.legend(fontsize=8.5)
ax2.set_ylim(0, 1.05)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT + 'Fig12_LinkQualityCurves.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: Fig12_LinkQualityCurves.png")

# ── Done ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("All 7 figures generated successfully.")
print("=" * 50)
print("""
Files saved:
  Fig01_PacketDistribution.png
  Fig02_LearningCurve.png
  Fig03_EpsilonDecay.png
  Fig04_QValueConvergence.png
  Fig05_UDDSResults.png
  Fig06_DecoySuspicion.png
  Fig07_LinkQualityCurves.png
""")
