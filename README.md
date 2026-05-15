# DeepAir-Dhaka: Spatiotemporal Air Quality Forecasting

![DeepAir Banner](https://img.shields.io/badge/Project-DeepAir--Dhaka-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Development-orange?style=for-the-badge)
![AI for Social Good](https://img.shields.io/badge/Focus-AI%20for%20Social%20Good-green?style=for-the-badge)

**DeepAir-Dhaka** is a machine learning-driven system designed to provide granular, neighborhood-level air quality forecasts for Dhaka, Bangladesh. Moving beyond simple city-wide averages, this project utilizes spatiotemporal deep learning to visualize and predict the diffusion of pollutants as dynamic heatmaps.

---

## 🌟 Project Overview

Dhaka consistently ranks as one of the most polluted cities globally. However, existing monitoring systems are sparse, leaving many neighborhoods without accurate data. DeepAir-Dhaka bridges this gap by:
- **Constructing "Virtual Sensors"**: Using mathematical interpolation to estimate pollution in unmonitored zones.
- **Spatiotemporal Forecasting**: Employing ConvLSTM networks to learn how pollution "clouds" travel across the city.
- **Dynamic Visualization**: Generating high-resolution heatmaps (32x32 grids) for real-time and future air quality insights.

## 🛠️ Technology Stack

- **Data Processing**: Python, Pandas, NumPy
- **Spatial Interpolation**: Scikit-Learn (IDW - Inverse Distance Weighting)
- **Deep Learning**: TensorFlow/Keras (ConvLSTM Architecture)
- **APIs**: Open-Meteo Air Quality & Archive APIs
- **Visualization**: Matplotlib, Folium

## 📊 Dataset Structure

The project currently generates a comprehensive spatiotemporal dataset:
- **Spatial Resolution**: 32x32 Grid (Dhaka City)
- **Temporal Resolution**: Hourly
- **Channels (21 total)**: 
  - PM2.5, PM10, CO, NO2, Ozone, Dust, AOD
  - Temperature, Humidity, Precipitation, Pressure, Cloud Cover, Radiation
  - Wind Vectors (U, V)
  - Cyclical Time Signatures (Hour, Day, Month)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required Libraries: `pandas`, `numpy`, `requests`, `scikit-learn`

### Data Generation
To build the 32x32 spatiotemporal tensor maps, run:
```bash
python "Scipts/createDeepAir datasetinNPY.py"
```
This script will:
1. Fetch hourly data from 5 major CAMS stations (Sangsad, Mirpur, Gazipur, Narayanganj, US Embassy).
2. Clean and standardize the data with physical vectors.
3. Perform IDW Interpolation to generate the 32x32x21 tensor.
4. Save the results as `DeepAir_Dhaka_5_Stations_Full_Hourly.csv` and `DeepAir_Dhaka_32x32_21_Channels.npy`.

## 📂 Project Structure

- `Docs/`: Project proposal and technical documentation.
- `Scipts/`: Core scripts for data collection, preprocessing, and model training.
- `scratch/`: Experimental scripts and temporary logs.

## 📜 References

1. Islam, M. S., et al. "Air Quality Prediction in Dhaka City using Deep Learning Approaches." IEEE, 2023.
2. Shi, X., et al. "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting." NIPS, 2015.

---

*This project contributes to the field of AI for Social Good, offering a scalable solution for cities with limited sensor infrastructure.*
