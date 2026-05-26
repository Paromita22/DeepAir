"""
DeepAir-Dhaka: Technical Presentation Assets Generator
Generates exactly what graphic design tools cannot:
1. Data-driven technical methodology panels (Loss curves, exact IDW maps)
2. A geographically accurate, animated PM2.5 forecast GIF.
"""

import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import matplotlib.animation as animation
import numpy as np
import os

OUT = '.'
os.makedirs(OUT, exist_ok=True)

# ── PRESENTATION PALETTE ─────────────────────────────
CARD      = '#ffffff'  
BORDER    = '#64748b'  
TEXT_MAIN = '#000000'  
DIM       = '#1e293b'  

C1_BLUE   = '#3b82f6' 
C2_TEAL   = '#14b8a6' 
C3_DTEAL  = '#0f766e' 
C4_PURP   = '#8b5cf6' 
C5_IND    = '#5b21b6' 
C6_ORAN   = '#f97316' 
C7_NAVY   = '#1e40af' 

STATIONS = {
    "Gazipur": (23.9941, 90.4223), "Savar": (23.9537, 90.2798), "Tongi": (23.8942, 90.4112),
    "Darussalam": (23.7808, 90.3554), "Agargaon": (23.7774, 90.3726), "Farmgate": (23.7594, 90.3887),
    "BUET": (23.7276, 90.3928), "NagarBhaban": (23.7241, 90.4091), "Narayanganj": (23.6261, 90.5072),
}

def rounded_box(ax, x, y, w, h, color, alpha=0.18, border=None, lw=3, zorder=3):
    border = border or color
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.03",
                          facecolor=color, alpha=alpha, edgecolor=border, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

def setup_panel(title, color):
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')
    ax.text(7, 9.5, title, color=color, fontsize=22, fontweight='heavy', ha='center', va='center')
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
# 1. IDW MAP (With Exact Coordinates & Starburst Labels)
# ═══════════════════════════════════════════════════════════════════
print("Generating 1. Panel 1 (IDW Map)...")
fig, ax = setup_panel('HOW: IDW Interpolation\n"9 stations → 1,024 grid cells"', C2_TEAL)
for i in range(16):
    for j in range(16):
        ax.add_patch(FancyBboxPatch((3.0 + j*0.5, 1.4 + i*0.45), 0.45, 0.4, boxstyle='round,pad=0.02', facecolor=C2_TEAL, alpha=0.1, edgecolor=C2_TEAL, linewidth=1.5))

offsets = {"Gazipur": (0, 25), "Savar": (0, 25), "Tongi": (-45, 15), "Darussalam": (-50, 25), "Agargaon": (55, 30), "Farmgate": (60, 0), "BUET": (-40, -35), "NagarBhaban": (45, -35), "Narayanganj": (0, -25)}
for name, (lat, lon) in STATIONS.items():
    px = 3.2 + ((lon - 90.27) / 0.25) * 7.6
    py = 1.6 + ((lat - 23.62) / 0.38) * 6.8
    ax.plot(px, py, 'o', color=C6_ORAN, ms=16, mec='white', mew=2.5, zorder=6)
    ox, oy = offsets[name]
    ax.annotate(name, xy=(px, py), xytext=(ox, oy), textcoords='offset points', fontsize=13, color=TEXT_MAIN, fontweight='heavy', ha='center', va='center', zorder=7, arrowprops=dict(arrowstyle='-', color=TEXT_MAIN, alpha=0.6, lw=1.5), path_effects=[pe.withStroke(linewidth=4, foreground=CARD)])

ax.text(7, 0.6, 'weight(i) = 1 / dist(i)²   →   cell_value = Σ(weight × station) / Σ(weight)', fontsize=18, color=TEXT_MAIN, ha='center', va='center', fontweight='heavy', bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8fafc', edgecolor=C2_TEAL, alpha=1.0, lw=4))
plt.savefig(os.path.join(OUT, 'methodology_1_idw.png'), dpi=200, bbox_inches='tight', facecolor=CARD); plt.close()


# ═══════════════════════════════════════════════════════════════════
# 2. TENSOR EXPLANATION
# ═══════════════════════════════════════════════════════════════════
print("Generating 2. Panel 2 (Tensor)...")
fig, ax = setup_panel('HOW: 24-Channel Feature Tensor\n"Each grid cell carries 24 numbers"', C3_DTEAL)
groups = [('Ch 0–5', 'Pollutants × LUR (PM2.5, PM10, CO, NO2, O3, Dust)', C3_DTEAL, 7.8), ('Ch 6', 'Aerosol Optical Depth', C3_DTEAL, 6.4), ('Ch 7–14', 'Weather (Temp, Hum, Rain, Press, Cloud, Rad, Wind)', C1_BLUE, 4.9), ('Ch 15–20', 'Time Encoding (sin/cos of hour, day, month)', '#15803d', 3.2), ('Ch 21', 'Industrial Prior (Gazipur, Savar, N\'ganj)', C4_PURP, 2.0), ('Ch 22-23', 'Water & Road Priors (Rivers, Highways)', C2_TEAL, 0.8)]
for ch_label, desc, col, ypos in groups:
    rounded_box(ax, 7, ypos, 13.0, 1.1, col, alpha=0.15, border=col, lw=3)
    ax.text(1.0, ypos, ch_label, fontsize=16, color=col, fontweight='heavy', ha='left', va='center')
    ax.text(13.0, ypos, desc, fontsize=16, color=TEXT_MAIN, fontweight='bold', ha='right', va='center')
ax.text(7, 8.8, 'Shape per timestep: (32, 32, 24)  =  24,576 numbers', fontsize=18, color=C3_DTEAL, ha='center', fontweight='heavy')
plt.savefig(os.path.join(OUT, 'methodology_2_tensor.png'), dpi=200, bbox_inches='tight', facecolor=CARD); plt.close()


# ═══════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTION MATH
# ═══════════════════════════════════════════════════════════════════
print("Generating 3. Panel 3 (Loss Function)...")
fig, ax = setup_panel('HOW: Custom Weighted Huber Loss\n"Model learns dangerous spikes, not just averages"', '#be123c')
x_err = np.linspace(-4, 4, 200); huber = np.where(np.abs(x_err) < 0.5, 0.5*x_err**2, 0.5*(np.abs(x_err) - 0.25)); mse = 0.5 * x_err**2
lax_x = np.interp(x_err, [-4, 4], [2, 12]); lax_h = np.interp(huber, [0, huber.max()], [1, 5.5]); lax_m = np.interp(mse, [0, mse.max()], [1, 5.5])
ax.plot(lax_x, lax_h, color='#be123c', lw=6, label='Huber (our loss)'); ax.plot(lax_x, lax_m, color=DIM, lw=4, label='MSE', ls='--')
ax.axvline(7, color=BORDER, lw=3, ls=':'); ax.text(7, 0.4, 'Zero Error', fontsize=16, color=DIM, ha='center', fontweight='bold')
ax.text(4.0, 6.0, 'Huber:\nRobust to outliers', fontsize=18, color='#be123c', ha='center', fontweight='heavy', bbox=dict(boxstyle='round,pad=0.5', facecolor=CARD, edgecolor='#be123c', alpha=1, lw=3))
ax.text(10.0, 6.0, 'MSE:\nExplodes on spikes', fontsize=18, color=DIM, ha='center', fontweight='heavy', bbox=dict(boxstyle='round,pad=0.5', facecolor=CARD, edgecolor=DIM, alpha=1, lw=3))
rounded_box(ax, 7, 7.8, 12.5, 2.0, '#be123c', alpha=0.15, border='#be123c', lw=3)
ax.text(7, 8.4, 'Pollution > 85th percentile (107.58 µg/m³)  →  3× Weight Penalty', fontsize=17, color='#be123c', ha='center', va='center', fontweight='heavy')
ax.text(7, 7.8, 'Pollution ≤ 85th percentile                  →  1× Weight Penalty', fontsize=17, color=TEXT_MAIN, ha='center', va='center', fontweight='bold')
ax.text(7, 7.1, '→ Forces model to predict high spikes accurately to minimize loss', fontsize=16, color=DIM, ha='center', va='center', fontweight='bold', style='italic')
plt.savefig(os.path.join(OUT, 'methodology_3_loss.png'), dpi=200, bbox_inches='tight', facecolor=CARD); plt.close()


# ═══════════════════════════════════════════════════════════════════
# 4. NORMALIZATION & ENCODING
# ═══════════════════════════════════════════════════════════════════
print("Generating 4. Panel 4 (Normalization)...")
fig, ax = setup_panel('HOW: Normalization & Features\n"Making data ML-friendly"', C4_PURP)
rounded_box(ax, 7, 7.8, 12.5, 2.2, C1_BLUE, alpha=0.15, border=C1_BLUE, lw=3)
ax.text(7, 8.5, 'Z-Score Normalization', fontsize=18, color=C1_BLUE, ha='center', fontweight='heavy')
ax.text(7, 7.8, 'normalized_value = (value − mean) / std', fontsize=18, color=TEXT_MAIN, ha='center', fontweight='bold', fontfamily='monospace')
ax.text(7, 7.1, 'PM2.5 mean = 58.26  |  std = 50.43', fontsize=16, color=DIM, ha='center', fontweight='bold')
rounded_box(ax, 7, 4.4, 12.5, 3.8, '#15803d', alpha=0.15, border='#15803d', lw=3)
ax.text(7, 5.8, 'Cyclical Time Encoding', fontsize=18, color='#15803d', ha='center', fontweight='heavy')
cx, cy, r = 3.5, 4.2, 1.3; theta = np.linspace(0, 2*np.pi, 100)
ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), color='#15803d', lw=4, alpha=0.6); ax.plot(cx + r, cy, 'o', color=C6_ORAN, ms=20); ax.plot(cx + r*np.cos(23*2*np.pi/24), cy + r*np.sin(23*2*np.pi/24), 'o', color=C6_ORAN, ms=20) 
ax.text(cx+1.9, cy + 0.3, 'Hr 0', fontsize=16, color=C6_ORAN, fontweight='heavy'); ax.text(cx+1.9, cy - 0.5, 'Hr 23', fontsize=16, color=C6_ORAN, fontweight='heavy')
ax.text(9.5, 4.9, 'hour → sin(2π · h/24)', fontsize=18, color=TEXT_MAIN, ha='center', fontweight='bold', fontfamily='monospace'); ax.text(9.5, 4.1, 'hour → cos(2π · h/24)', fontsize=18, color=TEXT_MAIN, ha='center', fontweight='bold', fontfamily='monospace')
ax.text(9.5, 3.2, 'Hour 23 and Hour 0 become\nneighbors on the circle', fontsize=16, color='#15803d', ha='center', fontweight='heavy')
rounded_box(ax, 7, 1.4, 12.5, 1.6, C2_TEAL, alpha=0.15, border=C2_TEAL, lw=3)
ax.text(7, 1.8, 'LUR Multipliers (Industry, Roads, Water)', fontsize=17, color=C2_TEAL, ha='center', fontweight='heavy')
ax.text(7, 1.0, 'Industry boosts PM2.5 (+50%), Rivers decrease it (−30%)', fontsize=16, color=TEXT_MAIN, ha='center', fontweight='bold')
plt.savefig(os.path.join(OUT, 'methodology_4_norm.png'), dpi=200, bbox_inches='tight', facecolor=CARD); plt.close()


# ═══════════════════════════════════════════════════════════════════
# 5. CONTEXT-RICH ANIMATED MAP GIF
# ═══════════════════════════════════════════════════════════════════
print("Generating 5. Context-Rich Animated Map GIF...")
def generate_animation():
    fig_anim, ax_anim = plt.subplots(figsize=(10, 9))
    fig_anim.patch.set_facecolor('#0d0d0d'); ax_anim.set_facecolor('#0d0d0d')
    
    # 1. Geographic Context & Axes
    ax_anim.set_xlim(90.28, 90.58); ax_anim.set_ylim(23.65, 23.93)
    ax_anim.set_xlabel('Longitude (°E)', color='white', fontsize=12, fontweight='bold')
    ax_anim.set_ylabel('Latitude (°N)', color='white', fontsize=12, fontweight='bold')
    ax_anim.tick_params(colors='white'); ax_anim.grid(color='#333333', linestyle=':', linewidth=1)
    for spine in ax_anim.spines.values(): spine.set_edgecolor('#444444')

    # 2. Station Overlays
    st_offsets = {"Gazipur": (0, 7), "Savar": (0, 7), "Tongi": (0, 7), "Darussalam": (-15, -15), "Agargaon": (5, 7), "Farmgate": (5, 7), "BUET": (0, -15), "NagarBhaban": (20, -10), "Narayanganj": (0, 7)}
    for name, (lat, lon) in STATIONS.items():
        ax_anim.plot(lon, lat, '^', color='white', ms=9, mec='black', zorder=6)
        ox, oy = st_offsets[name]
        ax_anim.annotate(name, xy=(lon, lat), xytext=(ox, oy), textcoords='offset points', color='white', fontsize=9, fontweight='bold', zorder=7, path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # 3. Simulate Base Grid
    x = np.linspace(90.28, 90.58, 32); y = np.linspace(23.93, 23.65, 32); X, Y = np.meshgrid(x, y)
    base = np.exp(-(((X-90.40)**2 + (Y-23.75)**2) / 0.015)) * 60
    im = ax_anim.imshow(base, cmap='turbo', vmin=0, vmax=150, interpolation='bicubic', extent=[90.28, 90.58, 23.65, 23.93], origin='upper', alpha=0.85, zorder=3)
    
    # 4. Colorbar & Text
    cbar = plt.colorbar(im, ax=ax_anim, shrink=0.8, pad=0.04)
    cbar.set_label('PM2.5 Concentration (µg/m³)', color='white', fontsize=12, fontweight='bold')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontweight='bold')
    
    ax_anim.set_title('DeepAir-Dhaka: 24h Dispersion Forecast', color='white', fontsize=18, fontweight='heavy', pad=15)
    time_text = ax_anim.text(0.03, 0.94, '+0h Forecast', transform=ax_anim.transAxes, color='white', fontsize=16, fontweight='heavy', zorder=8, bbox=dict(facecolor='black', alpha=0.7, edgecolor='white', lw=1))
    ax_anim.text(0.03, 0.04, 'Simulation:\nIndustrial Plume moving South-East\nfrom Gazipur & Tongi over 24 hours.', transform=ax_anim.transAxes, color='white', fontsize=12, fontweight='bold', zorder=8, bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

    # 5. Animation Loop
    def update(frame):
        px = 90.42 + (frame * 0.003); py = 23.99 - (frame * 0.008)
        plume1 = np.exp(-(((X-px)**2 + (Y-py)**2) / 0.008)) * 95
        rush = np.sin(frame/24 * np.pi) * 30
        city = np.exp(-(((X-90.39)**2 + (Y-23.75)**2) / 0.01)) * rush
        im.set_array(base + plume1 + city)
        time_text.set_text(f'+{frame}h Forecast')
        return [im, time_text]

    ani = animation.FuncAnimation(fig_anim, update, frames=24, interval=250, blit=True)
    ani.save(os.path.join(OUT, 'presentation_forecast.gif'), writer='pillow', fps=4)
    plt.close()

generate_animation()
print('\n✅ Success! Technical deep-dive images and animated GIF saved.')