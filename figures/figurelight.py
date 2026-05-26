"""
DeepAir-Dhaka: Full Pipeline & Methodology Visualization (Light Presentation Theme)
Run this in any Python environment (Windows, Mac, Linux).

Outputs 7 high-quality images to the current directory:
  1. pipeline_diagram.png       — the step-by-step data flow
  2. methodology_1_idw.png      — Step 1 deep dive
  3. methodology_2_tensor.png   — Step 2 deep dive
  4. methodology_3_convlstm.png — Step 3 deep dive
  5. methodology_4_loss.png     — Step 4 deep dive
  6. methodology_5_norm.png     — Step 5 deep dive
  7. methodology_6_results.png  — Step 6 deep dive
"""

import matplotlib
matplotlib.use('Agg') # Use headless backend so windows don't pop up
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = '.'
os.makedirs(OUT, exist_ok=True)

# ── PREMIUM LIGHT THEME COLORS ──────────────────────────────────────
BG        = '#f4f6f9'  # Soft presentation background
CARD      = '#ffffff'  # Clean white for boxes/cards
BORDER    = '#cbd5e1'  # Light slate border
TEXT_MAIN = '#1e293b'  # Dark slate for primary text (replacing white)
DIM       = '#64748b'  # Medium slate for secondary text
# Accents (deepened slightly for great contrast on light backgrounds)
TEAL      = '#0d9488'
ORANGE    = '#c2410c'
PURPLE    = '#6d28d9'
PINK      = '#be185d'
GOLD      = '#a16207'
BLUE      = '#1d4ed8'
GREEN     = '#15803d'

def rounded_box(ax, x, y, w, h, color, alpha=0.1, border=None, lw=2, zorder=3):
    border = border or color
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02",
                          facecolor=color, alpha=alpha,
                          edgecolor=border, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

def label(ax, x, y, text, size=11, color=TEXT_MAIN, weight='bold', ha='center', va='center', zorder=5):
    # White outline to make text readable when crossing lines
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder,
            path_effects=[pe.withStroke(linewidth=3, foreground=CARD)])

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: PIPELINE DIAGRAM (Wide Format)
# ═══════════════════════════════════════════════════════════════════
print("Generating Pipeline Diagram...")
fig1, ax1 = plt.subplots(figsize=(22, 10))
fig1.patch.set_facecolor(BG)
ax1.set_facecolor(BG)
ax1.set_xlim(0, 22)
ax1.set_ylim(0, 10)
ax1.axis('off')

# Title
ax1.text(11, 9.4, 'DeepAir-Dhaka: Complete Pipeline',
         fontsize=24, color=TEXT_MAIN, fontweight='bold', ha='center', va='center',
         path_effects=[pe.withStroke(linewidth=5, foreground=BG)])
ax1.text(11, 8.9, 'From 9 Raw Sensors → Neighborhood-Level PM2.5 Forecast Maps',
         fontsize=14, color=TEAL, ha='center', va='center')

# Step Boxes Data
steps = [
    (2.0,  5.0, TEAL,   '① DATA\nCOLLECTION',
     ['9 CAMS Stations', 'Open-Meteo API', '2022 → 2026', '38,304 hours', '7 pollutants', '8 weather vars']),
    (6.0,  5.0, ORANGE, '② PREPROCESSING\n& GRID BUILDING',
     ['IDW Interpolation', '9 pts → 32×32 grid', '1,024 virtual sensors', 'LUR multiplier', 'OSM spatial priors', '3.7 GB tensor']),
    (10.5, 5.0, PURPLE, '③ FEATURE\nENGINEERING',
     ['24 channels/cell', 'Cyclical time encode', 'Wind U/V vectors', 'Industrial prior', 'Water prior', 'Road density prior']),
    (15.0, 5.0, PINK,   '④ ConvLSTM\nTRAINING',
     ['Input: 24h history', 'ConvLSTM-64 encoder', 'Attention gate', 'ConvLSTM-128 decoder', 'Weighted Huber loss', '50 epochs, P100 GPU']),
    (19.5, 5.0, GOLD,   '⑤ FORECAST\n& EVALUATION',
     ['Output: PM2.5 map', '24h autoregressive', '7-day forecast', 'R=0.9729', 'MAE=3.39 µg/m³', '96.9% within ±10']),
]

for bx, by, col, title, details in steps:
    rounded_box(ax1, bx, by, 3.4, 7.0, col, alpha=0.08, border=col, lw=2) # Main card
    rounded_box(ax1, bx, by + 2.8, 3.4, 1.2, col, alpha=0.2, border=col, lw=0) # Header bar
    label(ax1, bx, by + 2.8, title, size=12, color=col, weight='bold') # Colored Title
    
    for i, d in enumerate(details):
        yy = by + 1.7 - i * 0.65
        ax1.plot([bx - 1.4, bx - 1.1], [yy, yy], color=col, lw=2, alpha=0.7, zorder=4)
        label(ax1, bx + 0.1, yy, d, size=10, color=TEXT_MAIN, weight='normal')

# Arrows between steps
for x1, x2, col in [(3.72, 4.28, TEAL), (7.72, 8.78, ORANGE), (12.22, 13.28, PURPLE), (16.72, 17.78, PINK)]:
    ax1.annotate('', xy=(x2, 5.0), xytext=(x1, 5.0),
                arrowprops=dict(arrowstyle='->', color=col, lw=3.5, mutation_scale=22), zorder=6)

# Bottom Section: Promises vs Delivery
ax1.text(11, 1.2, 'PROPOSAL PROMISED:', fontsize=11, color=DIM, ha='center', fontweight='bold')
promises = [
    (3.5,  '✓ IDW virtual sensor grid', TEAL),
    (8.5,  '✓ ConvLSTM spatiotemporal model', PURPLE),
    (13.5, '✓ 24h PM2.5 heatmap forecast', PINK),
    (18.5, '✓ Exceeded all metrics', GOLD),
]
for px, pt, pc in promises:
    rounded_box(ax1, px, 0.6, 4.8, 0.7, pc, alpha=0.1, border=pc, lw=1.5)
    label(ax1, px, 0.6, pt, size=10, color=pc, weight='bold')

plt.tight_layout(pad=0.5)
p1 = os.path.join(OUT, 'pipeline_diagram.png')
plt.savefig(p1, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()


# ═══════════════════════════════════════════════════════════════════
# INDIVIDUAL METHODOLOGY PANELS (Perfect for Presentations)
# ═══════════════════════════════════════════════════════════════════
def setup_panel(title, color):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(CARD) # Individual images use white backgrounds
    ax.set_facecolor(CARD)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(5, 9.6, title, color=color, fontsize=14, fontweight='bold', ha='center', va='center')
    return fig, ax

print("Generating Methodology Panel 1: IDW...")
fig, ax = setup_panel('HOW: IDW Interpolation\n"9 stations → 1,024 grid cells"', TEAL)
for i in range(5):
    for j in range(5):
        ax.add_patch(FancyBboxPatch((1.5 + j*1.4, 1.5 + i*1.3), 1.2, 1.1, boxstyle='round,pad=0.05',
                          facecolor=TEAL, alpha=0.05 + 0.05*np.random.random(), edgecolor=TEAL, linewidth=1))
# Sensors
for sx, sy, sl in [(3.0, 7.0, 'S1\n120'), (5.8, 4.5, 'S2\n85'), (8.0, 2.5, 'S3\n60')]:
    ax.plot(sx, sy, 'o', color=ORANGE, ms=16, zorder=6)
    ax.text(sx, sy, sl, fontsize=9, color='#ffffff', ha='center', va='center', fontweight='bold', zorder=7)
# Formula box
ax.text(5, 0.6, 'weight(i) = 1/dist(i)²   →   cell_value = Σ(weight×station) / Σ(weight)',
        fontsize=10, color=TEXT_MAIN, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=BG, edgecolor=TEAL, alpha=1.0, lw=1.5))
ax.text(5, 8.8, 'Closer station = more influence on grid cell value', fontsize=11, color=TEAL, ha='center', style='italic')
plt.savefig(os.path.join(OUT, 'methodology_1_idw.png'), dpi=200, bbox_inches='tight', facecolor=CARD)
plt.close()


print("Generating Methodology Panel 2: Tensor...")
fig, ax = setup_panel('HOW: 24-Channel Feature Tensor\n"Each grid cell carries 24 numbers"', ORANGE)
groups = [
    ('Ch 0–5',  'Pollutants × LUR\nPM2.5, PM10, CO, NO2, O3, Dust', ORANGE,   7.8),
    ('Ch 6',    'Aerosol Optical Depth', ORANGE,                                  6.5),
    ('Ch 7–14', 'Weather\nTemp, Humidity, Rain, Pressure,\nCloud, Radiation, Wind U/V', BLUE, 5.0),
    ('Ch 15–20','Time Encoding\nsin/cos(hour, day, month)', GREEN,               3.2),
    ('Ch 21',   'Industrial Prior\n(Gazipur, Narayanganj, Savar…)', PINK,        2.0),
    ('Ch 22',   'Water & Road Priors\n(Rivers & Major Highways)', TEAL,          0.8),
]
for ch_label, desc, col, ypos in groups:
    rounded_box(ax, 5, ypos, 9.5, 1.0, col, alpha=0.1, border=col, lw=1.5)
    ax.text(0.8, ypos, ch_label, fontsize=10, color=col, fontweight='bold', ha='left', va='center')
    ax.text(9.2, ypos, desc, fontsize=9, color=TEXT_MAIN, ha='right', va='center')
ax.text(5, 8.8, 'Shape per timestep: (32, 32, 24)  =  24,576 numbers', fontsize=11, color=ORANGE, ha='center', fontweight='bold')
plt.savefig(os.path.join(OUT, 'methodology_2_tensor.png'), dpi=200, bbox_inches='tight', facecolor=CARD)
plt.close()


print("Generating Methodology Panel 3: ConvLSTM...")
fig, ax = setup_panel('HOW: ConvLSTM U-Net Architecture\n"Learns WHERE and WHEN pollution moves"', PURPLE)
arch_steps = [
    (5, 8.5, '(batch, 24hrs, 32, 32, 24)', 'INPUT', BLUE),
    (5, 7.0, 'ConvLSTM2D (64 filters)\n→ learns spatial pattern each hour', 'ENCODER 1', PURPLE),
    (5, 5.4, 'ConvLSTM2D (128 filters)\n→ compresses 24hrs into 1 frame', 'ENCODER 2', PURPLE),
    (5, 3.8, 'Attention Gate\n→ tells decoder WHERE to focus', 'ATTENTION', PINK),
    (5, 2.2, 'Concatenate + Conv2D Decoder\n→ refines spatial prediction', 'DECODER', ORANGE),
    (5, 0.8, '(batch, 32, 32, 1)  →  PM2.5 heatmap', 'OUTPUT', GOLD),
]
for bx, by, desc, lbl, col in arch_steps:
    rounded_box(ax, bx, by, 9.5, 1.1, col, alpha=0.1, border=col, lw=2)
    ax.text(0.5, by, lbl, fontsize=9, color=col, fontweight='bold', ha='left', va='center')
    ax.text(5, by, desc, fontsize=9.5, color=TEXT_MAIN, ha='center', va='center')
for y1, y2 in [(7.9, 7.6), (6.4, 6.0), (4.8, 4.4), (3.2, 2.8), (1.6, 1.4)]:
    ax.annotate('', xy=(5, y2), xytext=(5, y1), arrowprops=dict(arrowstyle='->', color=PURPLE, lw=2.5), zorder=6)
ax.annotate('', xy=(8.5, 3.8), xytext=(8.5, 7.0), arrowprops=dict(arrowstyle='->', color=PINK, lw=2, connectionstyle='arc3,rad=-0.4'), zorder=6)
ax.text(9.4, 5.4, 'Skip\nConn.', fontsize=9, color=PINK, ha='center', va='center', fontweight='bold')
plt.savefig(os.path.join(OUT, 'methodology_3_convlstm.png'), dpi=200, bbox_inches='tight', facecolor=CARD)
plt.close()


print("Generating Methodology Panel 4: Loss Function...")
fig, ax = setup_panel('HOW: Custom Weighted Huber Loss\n"Model learns dangerous spikes, not just averages"', PINK)
x_err = np.linspace(-4, 4, 200)
huber  = np.where(np.abs(x_err) < 0.5, 0.5*x_err**2, 0.5*(np.abs(x_err) - 0.25))
mse    = 0.5 * x_err**2
lax_x = np.interp(x_err, [-4, 4], [1, 9])
lax_h = np.interp(huber, [0, huber.max()], [1, 5.0])
lax_m = np.interp(mse,   [0, mse.max()],   [1, 5.0])

ax.plot(lax_x, lax_h, color=PINK,  lw=3.5, label='Huber (our loss)')
ax.plot(lax_x, lax_m, color=DIM,   lw=2, label='MSE', ls='--')
ax.axvline(5, color=BORDER, lw=2, ls=':')
ax.text(5, 0.5, 'Zero Error', fontsize=10, color=DIM, ha='center')
ax.text(2.5, 5.5, 'Huber: Robust\nto outliers', fontsize=11, color=PINK, ha='center', fontweight='bold', bbox=dict(boxstyle='round', facecolor=CARD, edgecolor=PINK, alpha=0.9))
ax.text(7.5, 5.5, 'MSE: Explodes\non spikes', fontsize=11, color=DIM, ha='center', fontweight='bold', bbox=dict(boxstyle='round', facecolor=CARD, edgecolor=DIM, alpha=0.9))

rounded_box(ax, 5, 7.8, 9, 2.0, PINK, alpha=0.1, border=PINK, lw=1.5)
ax.text(5, 8.4, 'Pollution > 85th percentile (107 µg/m³) → 3× weight', fontsize=11, color=PINK, ha='center', va='center', fontweight='bold')
ax.text(5, 7.8, 'Pollution ≤ 85th percentile              → 1× weight', fontsize=11, color=TEXT_MAIN, ha='center', va='center')
ax.text(5, 7.1, '→ Model cannot ignore dangerous spikes to achieve a "safe average"', fontsize=10, color=DIM, ha='center', va='center', style='italic')
plt.savefig(os.path.join(OUT, 'methodology_4_loss.png'), dpi=200, bbox_inches='tight', facecolor=CARD)
plt.close()


print("Generating Methodology Panel 5: Normalization...")
fig, ax = setup_panel('HOW: Normalization & Features\n"Making data ML-friendly"', BLUE)
rounded_box(ax, 5, 7.8, 9.5, 2.2, BLUE, alpha=0.1, border=BLUE, lw=1.5)
ax.text(5, 8.5, 'Z-Score Normalization', fontsize=12, color=BLUE, ha='center', fontweight='bold')
ax.text(5, 7.8, 'normalized = (value − mean) / std', fontsize=11, color=TEXT_MAIN, ha='center', fontfamily='monospace')
ax.text(5, 7.1, 'PM2.5 mean=58.26  std=50.43  →  puts values in standard range', fontsize=10, color=DIM, ha='center')

rounded_box(ax, 5, 4.4, 9.5, 3.8, GREEN, alpha=0.1, border=GREEN, lw=1.5)
ax.text(5, 5.8, 'Cyclical Time Encoding', fontsize=12, color=GREEN, ha='center', fontweight='bold')
cx, cy, r = 2.5, 4.2, 1.3
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), color=GREEN, lw=2, alpha=0.5)
ax.plot(cx + r, cy, 'o', color=ORANGE, ms=10) # Hr 0
ax.plot(cx + r*np.cos(23*2*np.pi/24), cy + r*np.sin(23*2*np.pi/24), 'o', color=ORANGE, ms=10) # Hr 23
ax.text(cx+1.8, cy + 0.3, 'Hr 0', fontsize=10, color=ORANGE, fontweight='bold')
ax.text(cx+1.8, cy - 0.5, 'Hr 23', fontsize=10, color=ORANGE, fontweight='bold')
ax.text(6.8, 4.8, 'hour → sin(2π·h/24)', fontsize=11, color=TEXT_MAIN, ha='center', fontfamily='monospace')
ax.text(6.8, 4.1, 'hour → cos(2π·h/24)', fontsize=11, color=TEXT_MAIN, ha='center', fontfamily='monospace')
ax.text(6.8, 3.2, 'Hour 23 and Hour 0 become\nneighbors on the circle', fontsize=10, color=GREEN, ha='center', fontweight='bold')

rounded_box(ax, 5, 1.4, 9.5, 1.6, ORANGE, alpha=0.1, border=ORANGE, lw=1.5)
ax.text(5, 1.8, 'LUR Multipliers (Industry, Roads, Water)', fontsize=11, color=ORANGE, ha='center', fontweight='bold')
ax.text(5, 1.0, 'Industry boosts PM2.5 (+50%), Rivers decrease it (−30%)', fontsize=10, color=TEXT_MAIN, ha='center')
plt.savefig(os.path.join(OUT, 'methodology_5_norm.png'), dpi=200, bbox_inches='tight', facecolor=CARD)
plt.close()


print("Generating Methodology Panel 6: Results...")
fig, ax = setup_panel('HOW FAR WE EXCEEDED EXPECTATIONS\n"Promised vs Delivered"', GOLD)
comparisons = [
    ('Sensors',    '3–4 CAMS',        '9 CAMS stations',            TEAL),
    ('Grid',       '10×10',           '32×32 (10× more cells)',     ORANGE),
    ('Date range', '2023–2025',       '2022–2026 (4.3 years)',      BLUE),
    ('Features',   'PM2.5 + weather', '24 channels incl. OSM priors',PURPLE),
    ('Model',      'Basic ConvLSTM',  'ConvLSTM + Attention Gate',  PINK),
    ('Loss',       'RMSE',            'Weighted Huber (3× on spikes)',GOLD),
    ('Forecast',   '24h heatmap',     '24h + 7-day autoregressive', GREEN),
    ('Metric R',   'Not specified',   'R = 0.9729',                  TEAL),
    ('MAE',        'Not specified',   '3.39 µg/m³ (±6% of mean)',   ORANGE),
    ('Accuracy',   'Not specified',   '96.9% within ±10 µg/m³',      PINK),
]
for i, (topic, proposed, delivered, col) in enumerate(comparisons):
    yy = 8.5 - i * 0.8
    rounded_box(ax, 5, yy, 9.6, 0.65, col, alpha=0.1, border=col, lw=1)
    ax.text(0.3, yy, topic,     fontsize=10, color=col,       ha='left',  va='center', fontweight='bold')
    ax.text(3.0, yy, proposed,  fontsize=9,  color=DIM,       ha='left',  va='center')
    ax.text(5.5, yy, '→',       fontsize=12, color=col,       ha='center',va='center', fontweight='bold')
    ax.text(6.0, yy, delivered, fontsize=10, color=TEXT_MAIN, ha='left',  va='center', fontweight='bold')
plt.savefig(os.path.join(OUT, 'methodology_6_results.png'), dpi=200, bbox_inches='tight', facecolor=CARD)
plt.close()

print('\n✅ Success! All 7 diagrams saved locally.')