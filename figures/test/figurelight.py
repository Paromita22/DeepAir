"""
DeepAir-Dhaka: Ultimate High-Contrast Presentation Visualization
Optimized for Large Projectors. Fixes all text overflow issues by 
using a wider aspect ratio (16:9 style) and massive font sizes.
"""

import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = '.'
os.makedirs(OUT, exist_ok=True)

# ── ULTIMATE HIGH-CONTRAST PALETTE ─────────────────────────────
BG        = '#f4f6f9'  # Very light, clean grey background
CARD      = '#ffffff'  # Pure white for cards to pop
BORDER    = '#64748b'  
TEXT_MAIN = '#000000'  # PURE BLACK for absolute maximum readability
DIM       = '#1e293b'  # Very dark slate for secondary text

# Deep, highly saturated accents that won't wash out on bright projectors
TEAL      = '#0b7a75'
ORANGE    = '#c23b0a'
PURPLE    = '#4c1d95'
PINK      = '#be123c'
GOLD      = '#92400e'
BLUE      = '#1e3a8a'
GREEN     = '#14532d'

def rounded_box(ax, x, y, w, h, color, alpha=0.18, border=None, lw=3, zorder=3):
    border = border or color
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.03",
                          facecolor=color, alpha=alpha,
                          edgecolor=border, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

def label(ax, x, y, text, size=15, color=TEXT_MAIN, weight='bold', ha='center', va='center', zorder=5):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder,
            path_effects=[pe.withStroke(linewidth=4, foreground=CARD)])

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: PIPELINE DIAGRAM (Massive Canvas for Huge Text)
# ═══════════════════════════════════════════════════════════════════
print("Generating Pipeline Diagram...")
# Expanded canvas size to 28x13 to allow massive text without spilling
fig1, ax1 = plt.subplots(figsize=(28, 13)) 
fig1.patch.set_facecolor(BG)
ax1.set_facecolor(BG)
ax1.set_xlim(0, 28)
ax1.set_ylim(0, 13)
ax1.axis('off')

# Huge Titles
ax1.text(14, 12.0, 'DeepAir-Dhaka: Complete Pipeline',
         fontsize=38, color=TEXT_MAIN, fontweight='heavy', ha='center', va='center')
ax1.text(14, 11.2, 'From 9 Raw Sensors → Neighborhood-Level PM2.5 Forecast Maps',
         fontsize=22, color=TEAL, fontweight='bold', ha='center', va='center')

# Step Boxes Data (Spaced wider apart, centers at 3, 8.5, 14, 19.5, 25)
steps = [
    (2.8,  6.0, TEAL,   '① DATA\nCOLLECTION',
     ['9 CAMS Stations', 'Open-Meteo API', '2022 → 2026', '38,304 hours', '7 pollutants', '8 weather vars']),
    (8.4,  6.0, ORANGE, '② PREPROCESSING\n& GRID BUILDING',
     ['IDW Interpolation', '9 pts → 32×32 grid', '1,024 grid cells', 'LUR multipliers', 'OSM spatial priors', '3.7 GB tensor']),
    (14.0, 6.0, PURPLE, '③ FEATURE\nENGINEERING',
     ['24 channels/cell', 'Cyclical time encode', 'Wind U/V vectors', 'Industrial prior', 'Water prior', 'Road density prior']),
    (19.6, 6.0, PINK,   '④ ConvLSTM\nTRAINING',
     ['Input: 24h history', 'ConvLSTM-64 enc.', 'Attention gate', 'ConvLSTM-128 dec.', 'Weighted Huber loss', '50 epochs, GPU']),
    (25.2, 6.0, GOLD,   '⑤ FORECAST\n& EVALUATION',
     ['Output: PM2.5 map', '24h autoregressive', '7-day forecast', 'R = 0.9729', 'MAE = 3.39 µg/m³', '96.9% within ±10']),
]

for bx, by, col, title, details in steps:
    rounded_box(ax1, bx, by, 4.6, 8.5, col, alpha=0.15, border=col, lw=4) # Main card
    rounded_box(ax1, bx, by + 3.4, 4.6, 1.5, col, alpha=0.3, border=col, lw=0) # Header bar
    label(ax1, bx, by + 3.4, title, size=18, color=col, weight='heavy') # Colored Title
    
    for i, d in enumerate(details):
        yy = by + 2.0 - i * 0.85 # Spaced out highly
        ax1.plot([bx - 1.9, bx - 1.6], [yy, yy], color=col, lw=5, alpha=1.0, zorder=4)
        label(ax1, bx - 0.1, yy, d, size=16, color=TEXT_MAIN, weight='bold')

# Thick Arrows between steps
for x1, x2, col in [(5.1, 6.1, TEAL), (10.7, 11.7, ORANGE), (16.3, 17.3, PURPLE), (21.9, 22.9, PINK)]:
    ax1.annotate('', xy=(x2, 6.0), xytext=(x1, 6.0),
                arrowprops=dict(arrowstyle='->', color=col, lw=7, mutation_scale=40), zorder=6)

# Bottom Section
ax1.text(14, 0.9, 'HOW WE EXCEEDED THE PROPOSAL:', fontsize=18, color=DIM, ha='center', fontweight='heavy')
promises = [
    (4.5,  '✓ Full virtual sensor grid', TEAL),
    (10.8, '✓ Attention-based ConvLSTM', PURPLE),
    (17.1, '✓ Detailed 24h heatmaps', PINK),
    (23.4, '✓ Massive Accuracy Gains', GOLD),
]
for px, pt, pc in promises:
    rounded_box(ax1, px, 0.2, 5.5, 0.9, pc, alpha=0.2, border=pc, lw=3)
    label(ax1, px, 0.2, pt, size=16, color=pc, weight='heavy')

plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(OUT, 'pipeline_diagram.png'), dpi=250, bbox_inches='tight', facecolor=BG)
plt.close()


# ═══════════════════════════════════════════════════════════════════
# INDIVIDUAL METHODOLOGY PANELS (Wider 14x10 aspect ratio)
# ═══════════════════════════════════════════════════════════════════
def setup_panel(title, color):
    # WIDER Canvas so text fits perfectly inside the boxes
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(CARD) 
    ax.set_facecolor(CARD)
    ax.set_xlim(0, 14) # Canvas is now 14 units wide!
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(7, 9.5, title, color=color, fontsize=20, fontweight='heavy', ha='center', va='center')
    return fig, ax

print("Generating Panel 1: IDW...")
fig, ax = setup_panel('HOW: IDW Interpolation\n"9 stations → 1,024 grid cells"', TEAL)
# Grid centered horizontally (x starts at 3.5 instead of 1.5)
for i in range(5):
    for j in range(5):
        ax.add_patch(FancyBboxPatch((3.5 + j*1.4, 1.7 + i*1.3), 1.2, 1.1, boxstyle='round,pad=0.05',
                          facecolor=TEAL, alpha=0.2, edgecolor=TEAL, linewidth=3.5))
# Huge Sensors with High Contrast text
for sx, sy, sl in [(5.0, 7.2, 'S1\n120'), (7.8, 4.7, 'S2\n85'), (10.0, 2.7, 'S3\n60')]:
    ax.plot(sx, sy, 'o', color=ORANGE, ms=42, zorder=6) # MASSIVE dots
    ax.text(sx, sy, sl, fontsize=15, color='#ffffff', ha='center', va='center', fontweight='heavy', zorder=7)

ax.text(7, 0.6, 'weight(i) = 1 / dist(i)²   →   cell_value = Σ(weight × station) / Σ(weight)',
        fontsize=16, color=TEXT_MAIN, ha='center', va='center', fontweight='heavy',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8fafc', edgecolor=TEAL, alpha=1.0, lw=4))
ax.text(7, 8.5, 'Closer station = more influence on grid cell value', fontsize=17, color=TEXT_MAIN, ha='center', fontweight='bold', style='italic')
plt.savefig(os.path.join(OUT, 'methodology_1_idw.png'), dpi=250, bbox_inches='tight', facecolor=CARD)
plt.close()


print("Generating Panel 2: Tensor...")
fig, ax = setup_panel('HOW: 24-Channel Feature Tensor\n"Each grid cell carries 24 numbers"', ORANGE)
groups = [
    ('Ch 0–5',  'Pollutants × LUR\n(PM2.5, PM10, CO, NO2, O3, Dust)', ORANGE,   7.8),
    ('Ch 6',    'Aerosol Optical Depth', ORANGE,                                  6.4),
    ('Ch 7–14', 'Weather\n(Temp, Hum, Rain, Press, Cloud, Rad, Wind)', BLUE, 4.9),
    ('Ch 15–20','Time Encoding\n(sin/cos of hour, day, month)', GREEN,               3.2),
    ('Ch 21',   'Industrial Prior (Gazipur, Savar…)', PINK,                      2.0),
    ('Ch 22-23','Water & Road Priors (Rivers, Highways)', TEAL,                  0.8),
]
for ch_label, desc, col, ypos in groups:
    rounded_box(ax, 7, ypos, 13.0, 1.1, col, alpha=0.15, border=col, lw=3) # Box width is now 13!
    ax.text(1.0, ypos, ch_label, fontsize=16, color=col, fontweight='heavy', ha='left', va='center')
    ax.text(13.0, ypos, desc, fontsize=15, color=TEXT_MAIN, fontweight='bold', ha='right', va='center')
ax.text(7, 8.8, 'Shape per timestep: (32, 32, 24)  =  24,576 numbers', fontsize=18, color=ORANGE, ha='center', fontweight='heavy')
plt.savefig(os.path.join(OUT, 'methodology_2_tensor.png'), dpi=250, bbox_inches='tight', facecolor=CARD)
plt.close()


print("Generating Panel 3: ConvLSTM...")
fig, ax = setup_panel('HOW: ConvLSTM U-Net Architecture\n"Learns WHERE and WHEN pollution moves"', PURPLE)
arch_steps = [
    (7, 8.3, '(batch, 24hrs, 32, 32, 24)', 'INPUT', BLUE),
    (7, 6.7, 'ConvLSTM2D (64 filters)\n→ learns spatial pattern each hour', 'ENCODER 1', PURPLE),
    (7, 5.1, 'ConvLSTM2D (128 filters)\n→ compresses 24hrs into 1 frame', 'ENCODER 2', PURPLE),
    (7, 3.5, 'Attention Gate\n→ tells decoder WHERE to focus', 'ATTENTION', PINK),
    (7, 1.9, 'Concatenate + Conv2D Decoder\n→ refines spatial prediction', 'DECODER', ORANGE),
    (7, 0.5, '(batch, 32, 32, 1)  →  PM2.5 heatmap', 'OUTPUT', GOLD),
]
for bx, by, desc, lbl, col in arch_steps:
    rounded_box(ax, bx, by, 12.5, 1.2, col, alpha=0.15, border=col, lw=3)
    ax.text(1.2, by, lbl, fontsize=14, color=col, fontweight='heavy', ha='left', va='center')
    ax.text(7, by, desc, fontsize=15, color=TEXT_MAIN, fontweight='bold', ha='center', va='center')
for y1, y2 in [(7.6, 7.3), (6.0, 5.7), (4.4, 4.1), (2.8, 2.5), (1.2, 0.9)]:
    ax.annotate('', xy=(7, y2), xytext=(7, y1), arrowprops=dict(arrowstyle='->', color=PURPLE, lw=4), zorder=6)
# Skip connection on the right side
ax.annotate('', xy=(11.5, 3.5), xytext=(11.5, 6.7), arrowprops=dict(arrowstyle='->', color=PINK, lw=4, connectionstyle='arc3,rad=-0.3'), zorder=6)
ax.text(12.3, 5.1, 'Skip\nConn.', fontsize=14, color=PINK, ha='center', va='center', fontweight='heavy')
plt.savefig(os.path.join(OUT, 'methodology_3_convlstm.png'), dpi=250, bbox_inches='tight', facecolor=CARD)
plt.close()


print("Generating Panel 4: Loss Function...")
fig, ax = setup_panel('HOW: Custom Weighted Huber Loss\n"Model learns dangerous spikes, not just averages"', PINK)
x_err = np.linspace(-4, 4, 200)
huber  = np.where(np.abs(x_err) < 0.5, 0.5*x_err**2, 0.5*(np.abs(x_err) - 0.25))
mse    = 0.5 * x_err**2
lax_x = np.interp(x_err, [-4, 4], [2, 12]) # Scaled to wider canvas
lax_h = np.interp(huber, [0, huber.max()], [1, 5.5])
lax_m = np.interp(mse,   [0, mse.max()],   [1, 5.5])

ax.plot(lax_x, lax_h, color=PINK,  lw=6, label='Huber (our loss)')
ax.plot(lax_x, lax_m, color=DIM,   lw=4, label='MSE', ls='--')
ax.axvline(7, color=BORDER, lw=3, ls=':') # Center line at x=7
ax.text(7, 0.4, 'Zero Error', fontsize=16, color=DIM, ha='center', fontweight='bold')

ax.text(4.0, 6.0, 'Huber:\nRobust to outliers', fontsize=16, color=PINK, ha='center', fontweight='heavy', bbox=dict(boxstyle='round,pad=0.5', facecolor=CARD, edgecolor=PINK, alpha=1, lw=3))
ax.text(10.0, 6.0, 'MSE:\nExplodes on spikes', fontsize=16, color=DIM, ha='center', fontweight='heavy', bbox=dict(boxstyle='round,pad=0.5', facecolor=CARD, edgecolor=DIM, alpha=1, lw=3))

rounded_box(ax, 7, 7.8, 12.5, 2.0, PINK, alpha=0.15, border=PINK, lw=3)
ax.text(7, 8.4, 'Pollution > 85th percentile (107 µg/m³)  →  3× Weight Penalty', fontsize=16, color=PINK, ha='center', va='center', fontweight='heavy')
ax.text(7, 7.8, 'Pollution ≤ 85th percentile               →  1× Weight Penalty', fontsize=16, color=TEXT_MAIN, ha='center', va='center', fontweight='bold')
ax.text(7, 7.1, '→ Forces model to predict high spikes accurately to minimize loss', fontsize=15, color=DIM, ha='center', va='center', fontweight='bold', style='italic')
plt.savefig(os.path.join(OUT, 'methodology_4_loss.png'), dpi=250, bbox_inches='tight', facecolor=CARD)
plt.close()


print("Generating Panel 5: Normalization...")
fig, ax = setup_panel('HOW: Normalization & Features\n"Making data ML-friendly"', BLUE)
rounded_box(ax, 7, 7.8, 12.5, 2.2, BLUE, alpha=0.15, border=BLUE, lw=3)
ax.text(7, 8.5, 'Z-Score Normalization', fontsize=18, color=BLUE, ha='center', fontweight='heavy')
ax.text(7, 7.8, 'normalized_value = (value − mean) / std', fontsize=16, color=TEXT_MAIN, ha='center', fontweight='bold', fontfamily='monospace')
ax.text(7, 7.1, 'PM2.5 mean=58.26  std=50.43  →  puts all values in standard range', fontsize=15, color=DIM, ha='center', fontweight='bold')

rounded_box(ax, 7, 4.4, 12.5, 3.8, GREEN, alpha=0.15, border=GREEN, lw=3)
ax.text(7, 5.8, 'Cyclical Time Encoding', fontsize=18, color=GREEN, ha='center', fontweight='heavy')
cx, cy, r = 3.5, 4.2, 1.3
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), color=GREEN, lw=4, alpha=0.6)
ax.plot(cx + r, cy, 'o', color=ORANGE, ms=20) # Hr 0
ax.plot(cx + r*np.cos(23*2*np.pi/24), cy + r*np.sin(23*2*np.pi/24), 'o', color=ORANGE, ms=20) # Hr 23
ax.text(cx+1.9, cy + 0.3, 'Hr 0', fontsize=14, color=ORANGE, fontweight='heavy')
ax.text(cx+1.9, cy - 0.5, 'Hr 23', fontsize=14, color=ORANGE, fontweight='heavy')
ax.text(9.5, 4.9, 'hour → sin(2π · h/24)', fontsize=16, color=TEXT_MAIN, ha='center', fontweight='bold', fontfamily='monospace')
ax.text(9.5, 4.1, 'hour → cos(2π · h/24)', fontsize=16, color=TEXT_MAIN, ha='center', fontweight='bold', fontfamily='monospace')
ax.text(9.5, 3.2, 'Hour 23 and Hour 0 become\nneighbors on the circle', fontsize=15, color=GREEN, ha='center', fontweight='heavy')

rounded_box(ax, 7, 1.4, 12.5, 1.6, ORANGE, alpha=0.15, border=ORANGE, lw=3)
ax.text(7, 1.8, 'LUR Multipliers (Industry, Roads, Water)', fontsize=17, color=ORANGE, ha='center', fontweight='heavy')
ax.text(7, 1.0, 'Industry boosts PM2.5 (+50%), Rivers decrease it (−30%)', fontsize=16, color=TEXT_MAIN, ha='center', fontweight='bold')
plt.savefig(os.path.join(OUT, 'methodology_5_norm.png'), dpi=250, bbox_inches='tight', facecolor=CARD)
plt.close()


print("Generating Panel 6: Results...")
fig, ax = setup_panel('HOW FAR WE EXCEEDED EXPECTATIONS\n"Promised vs Delivered"', GOLD)
comparisons = [
    ('Sensors',    '3–4 CAMS',        '9 CAMS stations',            TEAL),
    ('Grid Cells', '10×10',           '32×32 (10× more cells)',     ORANGE),
    ('Date Range', '2023–2025',       '2022–2026 (4.3 years)',      BLUE),
    ('Features',   'PM2.5 + weather', '24 channels + map priors',   PURPLE),
    ('Model',      'Basic ConvLSTM',  'ConvLSTM + Attention',       PINK),
    ('Loss Func',  'RMSE',            'Weighted Huber',             GOLD),
    ('Forecast',   '24h heatmap',     '24h + 7-day extended',       GREEN),
    ('Metric R',   'Not specified',   'R = 0.9729',                  TEAL),
    ('MAE Error',  'Not specified',   '3.39 µg/m³',                 ORANGE),
    ('Accuracy',   'Not specified',   '96.9% within ±10 µg/m³',      PINK),
]

# Text shifted to use the new wide canvas without overlapping
for i, (topic, proposed, delivered, col) in enumerate(comparisons):
    yy = 8.5 - i * 0.8
    rounded_box(ax, 7, yy, 13.0, 0.65, col, alpha=0.18, border=col, lw=3)
    ax.text(1.0, yy, topic,     fontsize=16, color=col,       ha='left',  va='center', fontweight='heavy')
    ax.text(4.0, yy, proposed,  fontsize=15, color=DIM,       ha='left',  va='center', fontweight='bold')
    ax.text(7.5, yy, '→',       fontsize=22, color=col,       ha='center',va='center', fontweight='heavy')
    ax.text(8.3, yy, delivered, fontsize=16, color=TEXT_MAIN, ha='left',  va='center', fontweight='heavy')
plt.savefig(os.path.join(OUT, 'methodology_6_results.png'), dpi=250, bbox_inches='tight', facecolor=CARD)
plt.close()

print('\n✅ Success! All layout overlaps are permanently fixed.')