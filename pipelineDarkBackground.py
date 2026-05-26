"""
DeepAir-Dhaka: Full Pipeline & Methodology Visualization
Run this in Kaggle or any Python environment.
Saves two images:
  1. pipeline_diagram.png       — the step-by-step data flow
  2. methodology_diagram.png    — what happens INSIDE each step (the how)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = '.'
os.makedirs(OUT, exist_ok=True)

BG      = '#0a0a1a'
CARD    = '#12122a'
BORDER  = '#1e1e4a'
WHITE   = '#f0f0ff'
DIM     = '#8888aa'
TEAL    = '#00d4aa'
ORANGE  = '#ff8c42'
PURPLE  = '#9b59ff'
PINK    = '#ff4488'
GOLD    = '#ffd700'
BLUE    = '#4488ff'
GREEN   = '#44ff88'

def rounded_box(ax, x, y, w, h, color, alpha=0.15, border=None, lw=2, zorder=3):
    border = border or color
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02",
                          facecolor=color, alpha=alpha,
                          edgecolor=border, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

def arrow(ax, x1, y1, x2, y2, color=TEAL, lw=2.5, zorder=4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'),
                zorder=zorder)

def label(ax, x, y, text, size=11, color=WHITE, weight='bold', ha='center', va='center', zorder=5):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder,
            path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: PIPELINE DIAGRAM
# Shows the 5 main steps in order, left to right
# ═══════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(22, 10))
fig1.patch.set_facecolor(BG)
ax1.set_facecolor(BG)
ax1.set_xlim(0, 22)
ax1.set_ylim(0, 10)
ax1.axis('off')

# Title
ax1.text(11, 9.4, 'DeepAir-Dhaka: Complete Pipeline',
         fontsize=22, color=WHITE, fontweight='bold', ha='center', va='center',
         path_effects=[pe.withStroke(linewidth=5, foreground=BG)])
ax1.text(11, 8.9, 'From 9 Raw Sensors → Neighborhood-Level PM2.5 Forecast Maps',
         fontsize=12, color=TEAL, ha='center', va='center')

# ── STEP BOXES ─────────────────────────────────────────────────────
steps = [
    (2.0,  5.0, TEAL,   '① DATA\nCOLLECTION',
     ['9 CAMS Stations', 'Open-Meteo API', '2022 → 2026', '38,304 hours',
      '7 pollutants', '8 weather vars']),
    (6.0,  5.0, ORANGE, '② PREPROCESSING\n& GRID BUILDING',
     ['IDW Interpolation', '9 pts → 32×32 grid', '1,024 virtual sensors',
      'LUR multiplier', 'OSM spatial priors', '3.7 GB tensor']),
    (10.5, 5.0, PURPLE, '③ FEATURE\nENGINEERING',
     ['24 channels/cell', 'Cyclical time encode', 'Wind U/V vectors',
      'Industrial prior', 'Water prior', 'Road density prior']),
    (15.0, 5.0, PINK,   '④ ConvLSTM\nTRAINING',
     ['Input: 24h history', 'ConvLSTM-64 encoder', 'Attention gate',
      'ConvLSTM-128 decoder', 'Weighted Huber loss', '50 epochs, P100 GPU']),
    (19.5, 5.0, GOLD,   '⑤ FORECAST\n& EVALUATION',
     ['Output: PM2.5 map', '24h autoregressive', '7-day forecast',
      'R=0.9729', 'MAE=3.39 µg/m³', '96.9% within ±10']),
]

for bx, by, col, title, details in steps:
    # Main card
    rounded_box(ax1, bx, by, 3.4, 7.0, col, alpha=0.12, border=col, lw=2)
    # Header bar
    rounded_box(ax1, bx, by + 2.8, 3.4, 1.2, col, alpha=0.5, border=col, lw=0)
    label(ax1, bx, by + 2.8, title, size=11, color=WHITE, weight='bold')
    # Detail lines
    for i, d in enumerate(details):
        yy = by + 1.7 - i * 0.65
        ax1.plot([bx - 1.4, bx - 1.1], [yy, yy], color=col, lw=1.5, alpha=0.6, zorder=4)
        label(ax1, bx + 0.1, yy, d, size=9, color=WHITE, weight='normal')

# Arrows between steps
for x1, x2, col in [(3.72, 4.28, TEAL), (7.72, 8.78, ORANGE),
                     (12.22, 13.28, PURPLE), (16.72, 17.78, PINK)]:
    ax1.annotate('', xy=(x2, 5.0), xytext=(x1, 5.0),
                arrowprops=dict(arrowstyle='->', color=col, lw=3,
                                mutation_scale=20),
                zorder=6)

# Bottom: what we promised vs what we delivered
ax1.text(11, 1.1, 'PROPOSAL PROMISED:', fontsize=10, color=DIM, ha='center')
promises = [
    (3.5,  '✓ IDW virtual sensor grid', TEAL),
    (8.5,  '✓ ConvLSTM spatiotemporal model', PURPLE),
    (13.5, '✓ 24h PM2.5 heatmap forecast', PINK),
    (18.5, '✓ Exceeded all metrics', GOLD),
]
for px, pt, pc in promises:
    rounded_box(ax1, px, 0.55, 4.8, 0.7, pc, alpha=0.15, border=pc, lw=1.5)
    label(ax1, px, 0.55, pt, size=9.5, color=pc, weight='bold')

plt.tight_layout(pad=0.5)
p1 = os.path.join(OUT, 'pipeline_diagram.png')
plt.savefig(p1, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'✓ Saved: {p1}')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: METHODOLOGY DIAGRAM
# Shows HOW each step works internally — the technical details
# ═══════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 3, figsize=(24, 14))
fig2.patch.set_facecolor(BG)
fig2.suptitle('DeepAir-Dhaka: Methodology Deep-Dive\n'
              'How each component works internally',
              fontsize=20, color=WHITE, fontweight='bold', y=0.98)

panel_configs = [
    # (row, col, title, accent_color, content_function)
    (0, 0, 'HOW: IDW Interpolation',   TEAL),
    (0, 1, 'HOW: 24-Channel Tensor',   ORANGE),
    (0, 2, 'HOW: ConvLSTM Architecture', PURPLE),
    (1, 0, 'HOW: Custom Loss Function', PINK),
    (1, 1, 'HOW: Normalization',        BLUE),
    (1, 2, 'HOW: Results vs Proposal',  GOLD),
]

# ── Panel (0,0): IDW Interpolation ──────────────────────────────────
ax = axes[0][0]
ax.set_facecolor(CARD)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('HOW: IDW Interpolation\n"9 stations → 1,024 grid cells"',
             color=TEAL, fontsize=12, fontweight='bold', pad=8)

# Draw grid
for i in range(6):
    for j in range(6):
        c = ax.add_patch(FancyBboxPatch((1 + j*1.4, 1 + i*1.3), 1.2, 1.1,
                          boxstyle='round,pad=0.05',
                          facecolor=TEAL, alpha=0.06 + 0.04*np.random.random(),
                          edgecolor=TEAL, linewidth=0.5))
# 3 sensor dots
sensors = [(2.5, 7.5, 'S1\n120'), (5.5, 4.5, 'S2\n85'), (8.0, 2.5, 'S3\n60')]
for sx, sy, sl in sensors:
    ax.plot(sx, sy, 'o', color=ORANGE, ms=12, zorder=6)
    ax.text(sx, sy, sl, fontsize=7, color=WHITE, ha='center', va='center',
            fontweight='bold', zorder=7)

# Formula
ax.text(5, 0.5,
        'weight(i) = 1/dist(i)²   →   cell_value = Σ(weight×station) / Σ(weight)',
        fontsize=8, color=DIM, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor=BORDER, edgecolor=TEAL, alpha=0.8))
ax.text(5, 9.5, 'Closer station = more influence on grid cell value',
        fontsize=9, color=TEAL, ha='center', va='center', style='italic')

# ── Panel (0,1): 24-Channel Tensor ──────────────────────────────────
ax = axes[0][1]
ax.set_facecolor(CARD)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('HOW: 24-Channel Feature Tensor\n"Each grid cell carries 24 numbers"',
             color=ORANGE, fontsize=12, fontweight='bold', pad=8)

channel_groups = [
    ('Ch 0–5',  'Pollutants × LUR\nPM2.5, PM10, CO, NO2, O3, Dust', ORANGE,   8.5),
    ('Ch 6',    'Aerosol Optical Depth', ORANGE,                                  7.3),
    ('Ch 7–14', 'Weather\nTemp, Humidity, Rain,\nPressure, Cloud, Radiation, Wind U/V', BLUE, 5.8),
    ('Ch 15–20','Time Encoding\nsin/cos(hour, day, month)', GREEN,               4.0),
    ('Ch 21',   'Industrial Prior\n(Gazipur, Narayanganj, Savar…)', PINK,        2.9),
    ('Ch 22',   'Water Prior\n(Buriganga, Turag, Balu rivers)', TEAL,            2.0),
    ('Ch 23',   'Road Prior\n(Major highways)', DIM,                              1.2),
]
for ch_label, desc, col, ypos in channel_groups:
    rounded_box(ax, 5, ypos, 9.5, 0.75 if '\n' not in desc else 1.0,
                col, alpha=0.15, border=col, lw=1.5)
    ax.text(1.2, ypos, ch_label, fontsize=8, color=col,
            fontweight='bold', ha='left', va='center')
    ax.text(9.2, ypos, desc, fontsize=7.5, color=WHITE,
            ha='right', va='center')

ax.text(5, 9.5, 'Shape per timestep: (32, 32, 24)  =  24,576 numbers',
        fontsize=9, color=ORANGE, ha='center', va='center',
        fontweight='bold')

# ── Panel (0,2): ConvLSTM Architecture ──────────────────────────────
ax = axes[0][2]
ax.set_facecolor(CARD)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('HOW: ConvLSTM U-Net Architecture\n"Learns WHERE and WHEN pollution moves"',
             color=PURPLE, fontsize=12, fontweight='bold', pad=8)

arch_steps = [
    (5, 9.0, '(batch, 24hrs, 32, 32, 24)', 'INPUT', BLUE,   9.5),
    (5, 7.5, 'ConvLSTM2D (64 filters)\nreturn_sequences=True\n→ learns spatial pattern each hour', 'ENCODER 1', PURPLE, 8.0),
    (5, 5.8, 'ConvLSTM2D (128 filters)\nreturn_sequences=False\n→ compresses 24hrs into 1 frame', 'ENCODER 2', PURPLE, 6.3),
    (5, 4.1, 'Attention Gate\n→ tells decoder WHERE to focus\n(high-pollution zones get more weight)', 'ATTENTION', PINK, 4.6),
    (5, 2.5, 'Concatenate + Conv2D(128) + Conv2D(64)\n→ refines spatial prediction', 'DECODER', ORANGE, 3.0),
    (5, 1.0, '(batch, 32, 32, 1)  →  PM2.5 heatmap', 'OUTPUT', GOLD, 1.5),
]
for bx, by, desc, lbl, col, _ in arch_steps:
    h = 0.9 if '\n' not in desc else 1.1
    rounded_box(ax, bx, by, 9.5, h, col, alpha=0.2, border=col, lw=2)
    ax.text(0.5, by, lbl, fontsize=7, color=col, fontweight='bold',
            ha='left', va='center')
    ax.text(5, by, desc, fontsize=7.5, color=WHITE, ha='center', va='center')

# arrows
for y1, y2 in [(8.55, 8.0), (7.0, 6.35), (5.35, 4.65), (3.65, 3.0), (2.0, 1.5)]:
    ax.annotate('', xy=(5, y2), xytext=(5, y1),
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=2), zorder=6)

# skip connection
ax.annotate('', xy=(8.5, 4.1), xytext=(8.5, 7.5),
            arrowprops=dict(arrowstyle='->', color=PINK, lw=1.5,
                            connectionstyle='arc3,rad=-0.3'), zorder=6)
ax.text(9.3, 5.8, 'skip\nconn.', fontsize=7, color=PINK, ha='center', va='center')

# ── Panel (1,0): Custom Loss ─────────────────────────────────────────
ax = axes[1][0]
ax.set_facecolor(CARD)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('HOW: Custom Weighted Huber Loss\n"Model learns dangerous spikes, not just averages"',
             color=PINK, fontsize=12, fontweight='bold', pad=8)

# Draw loss curve
x_err = np.linspace(-4, 4, 200)
huber  = np.where(np.abs(x_err) < 0.5, 0.5*x_err**2, 0.5*(np.abs(x_err) - 0.25))
mse    = 0.5 * x_err**2

lax_x = np.interp(x_err, [-4, 4], [1, 9])
lax_h = np.interp(huber, [0, huber.max()], [1, 5.5])
lax_m = np.interp(mse,   [0, mse.max()],   [1, 5.5])

ax.plot(lax_x, lax_h, color=PINK,  lw=2.5, label='Huber (our loss)')
ax.plot(lax_x, lax_m, color=DIM,   lw=1.5, label='MSE', ls='--')
ax.axvline(5, color=WHITE, lw=1, ls=':', alpha=0.5)
ax.text(5, 0.6, 'zero error', fontsize=8, color=DIM, ha='center')

ax.text(2.5, 5.8, 'Huber: robust\nto outliers', fontsize=9, color=PINK,
        ha='center', bbox=dict(boxstyle='round', facecolor=BORDER, alpha=0.8))
ax.text(7.5, 5.8, 'MSE: explodes\non spikes', fontsize=9, color=DIM,
        ha='center', bbox=dict(boxstyle='round', facecolor=BORDER, alpha=0.8))

# Weight explanation
rounded_box(ax, 5, 8.0, 9, 2.5, PINK, alpha=0.15, border=PINK, lw=1.5)
ax.text(5, 8.8, 'Pollution > 85th percentile (107 µg/m³) → 3× weight',
        fontsize=10, color=PINK, ha='center', va='center', fontweight='bold')
ax.text(5, 8.1, 'Pollution ≤ 85th percentile              → 1× weight',
        fontsize=10, color=WHITE, ha='center', va='center')
ax.text(5, 7.4, '→ Model can\'t ignore dangerous spikes by predicting "average"',
        fontsize=9, color=DIM, ha='center', va='center', style='italic')

# ── Panel (1,1): Normalization ────────────────────────────────────────
ax = axes[1][1]
ax.set_facecolor(CARD)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('HOW: Normalization & Time Encoding\n"Making data ML-friendly"',
             color=BLUE, fontsize=12, fontweight='bold', pad=8)

# Normalization
rounded_box(ax, 5, 8.2, 9.5, 2.8, BLUE, alpha=0.15, border=BLUE, lw=1.5)
ax.text(5, 9.4, 'Z-Score Normalization', fontsize=11, color=BLUE,
        ha='center', va='center', fontweight='bold')
ax.text(5, 8.7, 'normalized = (value − mean) / std', fontsize=10,
        color=WHITE, ha='center', va='center',
        fontfamily='monospace')
ax.text(5, 8.0, 'PM2.5 mean=58.26  std=50.43  →  all values roughly −2 to +2',
        fontsize=8.5, color=DIM, ha='center', va='center')
ax.text(5, 7.5, 'Computed on TRAINING set only. Applied to both train & val.',
        fontsize=8, color=TEAL, ha='center', va='center', style='italic')

# Cyclical encoding diagram
rounded_box(ax, 5, 5.0, 9.5, 3.5, GREEN, alpha=0.12, border=GREEN, lw=1.5)
ax.text(5, 6.5, 'Cyclical Time Encoding', fontsize=11, color=GREEN,
        ha='center', va='center', fontweight='bold')

theta = np.linspace(0, 2*np.pi, 100)
cx, cy, r = 3.0, 4.8, 1.2
ax.plot(1.2 + r*np.cos(theta), cy + r*np.sin(theta), color=GREEN, lw=1.5, alpha=0.5)
ax.plot(1.2 + r*np.cos(0),    cy + r*np.sin(0),    'o', color=ORANGE, ms=8)   # hour 0
ax.plot(1.2 + r*np.cos(23*2*np.pi/24), cy + r*np.sin(23*2*np.pi/24), 'o',
        color=ORANGE, ms=8)  # hour 23
ax.text(1.2, cy + r + 0.3, 'Hr 0', fontsize=8, color=ORANGE, ha='center')
ax.text(1.2 + r*np.cos(23*2*np.pi/24) + 0.3,
        cy + r*np.sin(23*2*np.pi/24), 'Hr 23', fontsize=8, color=ORANGE)
ax.text(1.2, cy - r - 0.3, 'Clock analogy:', fontsize=7, color=DIM, ha='center')

ax.text(6.5, 5.6, 'hour → sin(2π·h/24)', fontsize=9, color=WHITE, ha='center',
        fontfamily='monospace')
ax.text(6.5, 5.1, 'hour → cos(2π·h/24)', fontsize=9, color=WHITE, ha='center',
        fontfamily='monospace')
ax.text(6.5, 4.5, 'Hour 23 & Hour 0 become', fontsize=8.5, color=DIM, ha='center')
ax.text(6.5, 4.0, 'neighbours on the circle', fontsize=8.5, color=GREEN,
        ha='center', fontweight='bold')

# LUR
rounded_box(ax, 5, 1.8, 9.5, 2.5, ORANGE, alpha=0.12, border=ORANGE, lw=1.5)
ax.text(5, 2.9, 'LUR Multiplier (Land Use Regression)', fontsize=10, color=ORANGE,
        ha='center', va='center', fontweight='bold')
ax.text(5, 2.2, 'cell_PM25 × (1 + 0.5×industry + 0.2×roads − 0.3×water)',
        fontsize=8.5, color=WHITE, ha='center', va='center', fontfamily='monospace')
ax.text(5, 1.5, 'Industrial zones: +50% | Roads: +20% | Rivers: −30%',
        fontsize=8.5, color=DIM, ha='center', va='center')

# ── Panel (1,2): Results vs Proposal ─────────────────────────────────
ax = axes[1][2]
ax.set_facecolor(CARD)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('HOW FAR WE EXCEEDED THE PROPOSAL\n"Promised vs Delivered"',
             color=GOLD, fontsize=12, fontweight='bold', pad=8)

comparisons = [
    # (topic, proposed, delivered, color)
    ('Sensors',    '3–4 CAMS',        '9 CAMS stations',            TEAL),
    ('Grid',       '10×10',           '32×32 (10× more cells)',     ORANGE),
    ('Date range', '2023–2025',       '2022–2026 (4.3 years)',      BLUE),
    ('Features',   'PM2.5 + weather', '24 channels incl. OSM priors',PURPLE),
    ('Model',      'Basic ConvLSTM',  'ConvLSTM + Attention Gate',  PINK),
    ('Loss',       'RMSE',            'Weighted Huber (3× on spikes)',GOLD),
    ('Forecast',   '24h heatmap',     '24h + 7-day autoregressive', GREEN),
    ('Metric R',   'Not specified',   'R = 0.9729',                  TEAL),
    ('MAE',        'Not specified',   '3.39 µg/m³ (±6% of mean)',   ORANGE),
    ('Within ±10', 'Not specified',   '96.9% of all predictions',    PINK),
]

for i, (topic, proposed, delivered, col) in enumerate(comparisons):
    yy = 9.3 - i * 0.88
    rounded_box(ax, 5, yy, 9.6, 0.72, col, alpha=0.12, border=col, lw=1)
    ax.text(0.3, yy, topic,     fontsize=8,   color=col,   ha='left',  va='center', fontweight='bold')
    ax.text(3.3, yy, proposed,  fontsize=7.5, color=DIM,   ha='left',  va='center')
    ax.text(5.5, yy, '→',       fontsize=10,  color=col,   ha='center',va='center')
    ax.text(6.0, yy, delivered, fontsize=7.5, color=WHITE, ha='left',  va='center', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96], pad=1.5)
p2 = os.path.join(OUT, 'methodology_diagram.png')
plt.savefig(p2, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'✓ Saved: {p2}')

print('\n✅ Both diagrams saved.')
print('   pipeline_diagram.png    — the 5-step flow')
print('   methodology_diagram.png — the technical details inside each step')