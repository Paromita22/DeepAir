import pandas as pd
import numpy as np
import requests
import time
from sklearn.neighbors import KNeighborsRegressor

print("=== DEEPAIR-DHAKA: ULTIMATE 32x32 SPATIOTEMPORAL DATA BUILDER ===")

# 1. Official CAMS Stations + US Embassy
STATIONS = {
    "CAMS1_Sangsad": {"latitude": 23.7600, "longitude": 90.3900},
    "CAMS3_Mirpur":  {"latitude": 23.7800, "longitude": 90.3600},
    "CAMS4_Gazipur": {"latitude": 23.9900, "longitude": 90.4200},
    "CAMS5_Narayanganj": {"latitude": 23.6300, "longitude": 90.5100},
    "US_Embassy":    {"latitude": 23.8100, "longitude": 90.4100}
}

def fetch_station_data(station_name, coords, year):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31" if year < 2026 else "2026-05-15"
    lat, lon = coords["latitude"], coords["longitude"]

    # Fetch Air Quality (Added Dust & Aerosol Optical Depth)
    aq_url = (f"https://air-quality-api.open-meteo.com/v1/air-quality?"
              f"latitude={lat}&longitude={lon}&"
              f"hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,dust,aerosol_optical_depth&"
              f"start_date={start_date}&end_date={end_date}&timezone=Asia/Dhaka")

    aq_response = requests.get(aq_url)
    if aq_response.status_code != 200: return None
    df_aq = pd.DataFrame(aq_response.json().get('hourly', {}))

    # Fetch Weather (Added Solar Radiation)
    wx_url = (f"https://archive-api.open-meteo.com/v1/archive?"
              f"latitude={lat}&longitude={lon}&"
              f"hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,"
              f"precipitation,surface_pressure,cloud_cover,shortwave_radiation&"
              f"start_date={start_date}&end_date={end_date}&timezone=Asia/Dhaka")

    wx_response = requests.get(wx_url)
    if wx_response.status_code != 200:
        wx_url = wx_url.replace("archive-api", "api").replace("/archive", "/forecast")
        wx_response = requests.get(wx_url)
        if wx_response.status_code != 200: return None

    df_wx = pd.DataFrame(wx_response.json().get('hourly', {}))
    if df_aq.empty or df_wx.empty: return None

    # Merge and inject Spatial Meta-data
    df = pd.merge(df_aq, df_wx, on="time", how="inner")
    df['monitoring_station'] = station_name
    df['latitude'] = lat
    df['longitude'] = lon
    return df

# --- STEP 1: DOWNLOAD RAW STATION DATA ---
all_data = []
for year in range(2022, 2027):
    print(f"[+] Fetching Hourly Multi-Channel Data for {year}...")
    for name, coords in STATIONS.items():
        df_station = fetch_station_data(name, coords, year)
        if df_station is not None:
            all_data.append(df_station)
        time.sleep(1) # Prevent API rate limit

df_massive = pd.concat(all_data, ignore_index=True)
df_massive['time'] = pd.to_datetime(df_massive['time'])

# --- STEP 2: MATHEMATICAL CLEANUP & FULL COLUMN NAMES ---
print("\n[+] Processing Physics, Cyclic Time, and Column Standardization...")

# Calculate proper Wind Vectors
wd_rad = df_massive['wind_direction_10m'] * np.pi / 180.0
df_massive['wind_vector_u'] = df_massive['wind_speed_10m'] * np.sin(wd_rad)
df_massive['wind_vector_v'] = df_massive['wind_speed_10m'] * np.cos(wd_rad)

# Cyclical Time Signatures
df_massive['hour_sine'] = np.sin(2 * np.pi * df_massive['time'].dt.hour / 24.0)
df_massive['hour_cosine'] = np.cos(2 * np.pi * df_massive['time'].dt.hour / 24.0)
df_massive['day_sine'] = np.sin(2 * np.pi * df_massive['time'].dt.dayofweek / 7.0)
df_massive['day_cosine'] = np.cos(2 * np.pi * df_massive['time'].dt.dayofweek / 7.0)
df_massive['month_sine'] = np.sin(2 * np.pi * df_massive['time'].dt.month / 12.0)
df_massive['month_cosine'] = np.cos(2 * np.pi * df_massive['time'].dt.month / 12.0)

# Full Form Mapping
rename_mapping = {
    'time': 'timestamp',
    'pm2_5': 'particulate_matter_2_5_ug_m3',
    'pm10': 'particulate_matter_10_ug_m3',
    'carbon_monoxide': 'carbon_monoxide_ug_m3',
    'nitrogen_dioxide': 'nitrogen_dioxide_ug_m3',
    'ozone': 'ozone_ug_m3',
    'dust': 'dust_ug_m3',
    'temperature_2m': 'temperature_celsius',
    'relative_humidity_2m': 'relative_humidity_percent',
    'precipitation': 'precipitation_millimetres',
    'surface_pressure': 'surface_pressure_hpa',
    'cloud_cover': 'cloud_cover_percent',
    'shortwave_radiation': 'shortwave_radiation_w_m2'
}

df_massive.rename(columns=rename_mapping, inplace=True)
df_massive.dropna(subset=['particulate_matter_2_5_ug_m3'], inplace=True)

# Select final columns in logical order
cols = [
    'timestamp', 'monitoring_station', 'latitude', 'longitude',
    'particulate_matter_2_5_ug_m3', 'particulate_matter_10_ug_m3', 'carbon_monoxide_ug_m3',
    'nitrogen_dioxide_ug_m3', 'ozone_ug_m3', 'dust_ug_m3', 'aerosol_optical_depth',
    'temperature_celsius', 'relative_humidity_percent', 'precipitation_millimetres',
    'surface_pressure_hpa', 'cloud_cover_percent', 'shortwave_radiation_w_m2',
    'wind_vector_u', 'wind_vector_v',
    'hour_sine', 'hour_cosine', 'day_sine', 'day_cosine', 'month_sine', 'month_cosine'
]

df_hourly = df_massive[cols].round(3)

# Save the readable CSV
csv_filename = "DeepAir_Dhaka_5_Stations_Full_Hourly.csv"
df_hourly.to_csv(csv_filename, index=False)


# --- STEP 3: GENERATE THE 32x32 SPATIOTEMPORAL GRID ---
print("\n[+] Generating 32x32 Deep Learning Tensor Maps...")

lat_min, lat_max = 23.65, 23.93
lon_min, lon_max = 90.28, 90.58
grid_size = 32

lats = np.linspace(lat_max, lat_min, grid_size)
lons = np.linspace(lon_min, lon_max, grid_size)
grid_lons, grid_lats = np.meshgrid(lons, lats)
grid_points = np.column_stack([grid_lats.ravel(), grid_lons.ravel()])

unique_times = sorted(df_hourly['timestamp'].unique())

# Features that need to be mapped (Exclude metadata)
map_features = cols[4:]
tensor_list = []

for hour_time in unique_times:
    hour_data = df_hourly[df_hourly['timestamp'] == hour_time]
    if len(hour_data) < 3: continue

    X_train = hour_data[['latitude', 'longitude']].values
    y_train_matrix = hour_data[map_features].values

    # IDW Interpolation mapped across all 21 data channels instantly
    model = KNeighborsRegressor(n_neighbors=len(X_train), weights='distance')
    model.fit(X_train, y_train_matrix)
    preds = model.predict(grid_points)

    # Reshape from flat list into (32, 32, 21) map image
    hour_image = preds.reshape((grid_size, grid_size, len(map_features)))
    tensor_list.append(hour_image)

final_tensor = np.array(tensor_list, dtype=np.float32)

npy_filename = "DeepAir_Dhaka_32x32_21_Channels.npy"
np.save(npy_filename, final_tensor)

print(f"\n✅ SUCCESS! Machine Learning Tensor built.")
print(f"Tensor Shape: {final_tensor.shape}")
print(f"Representing: [Hours], [Pixels Height], [Pixels Width], [Data Channels]")

try:
    from google.colab import files
    files.download(csv_filename)
    files.download(npy_filename)
except:
    print("\nScript executed perfectly. Please check your file explorer to download.")