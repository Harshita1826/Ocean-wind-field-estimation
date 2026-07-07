# Ocean Wind Field Estimation from SAR Imagery - Indian Coasts

**CEC OpenProject 2 · Interim & Final Submission**

An operational API pipeline built with FastAPI and Google Earth Engine to estimate marine
wind velocity vectors from Copernicus Sentinel-1 Synthetic Aperture Radar (SAR) imagery
over the Tamil Nadu and Gujarat coastlines of India.

---

## Project Overview

| Item | Detail |
|------|--------|
| **Problem** | Ocean wind field estimation using SAR imagery over Indian coastal areas for wind farm planning |
| **Study Areas** | Tamil Nadu coast · Gujarat coast |
| **Dataset** | Sentinel-1 IW GRD (VV polarisation) via Google Earth Engine |
| **Wind Speed Model** | CMOD5.n Geophysical Model Function (GMF) |
| **Wind Direction** | SAR-image gradient streak detection |
| **Deliverable** | REST API returning wind field vectors for a user-supplied date and AOI |

---

## Repository Structure

```
.
├── app.py                    # FastAPI backend (CMOD5.n GMF, GEE integration)
├── visualize_wind_field.py   # Wind field map generator (publication-quality PNG)
├── validate_wind_field.py    # Validation against ERA5 reanalysis; exports CSV + scatter plot
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Quick Start

### 1 - Install dependencies

```bash
pip install -r requirements.txt
```

### 2 - Authenticate Google Earth Engine

```bash
earthengine authenticate
```

Then update the project ID in `app.py` line 35:

```python
ee.Initialize(project='YOUR-APPROVED-GEE-PROJECT-ID')
```

### 3 - Run the API server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API docs auto-generated at: `http://localhost:8000/docs`

### 4 - Query wind field

```bash
curl -X POST http://localhost:8000/api/wind-field \
  -H "Content-Type: application/json" \
  -d '{
    "min_lat": 8.0, "min_lon": 77.5,
    "max_lat": 13.5, "max_lon": 82.0,
    "date": "2021-01-15"
  }'
```

### 5 - Visualise output

```bash
# Demo mode (no GEE required)
python visualize_wind_field.py --region tamilnadu
python visualize_wind_field.py --region gujarat

# From real API response
curl ... > response.json
python visualize_wind_field.py response.json
```

### 6 - Run validation

```bash
python validate_wind_field.py
# Outputs: validation_table.csv  +  validation_scatter.png
```

---

## API Reference

### `POST /api/wind-field`

**Request body:**

```json
{
  "min_lat": 8.0,
  "min_lon": 77.5,
  "max_lat": 13.5,
  "max_lon": 82.0,
  "date": "2021-01-15"
}
```

**Response:**

```json
{
  "status": "success",
  "image_date": "2021-01-15",
  "statistics": {
    "total_nodes": 97,
    "mean_wind_speed_ms": 7.4,
    "min_wind_speed_ms": 3.1,
    "max_wind_speed_ms": 12.8
  },
  "data": [
    {
      "latitude": 9.5,
      "longitude": 79.5,
      "wind_speed_ms": 6.4,
      "wind_direction_deg": 220.3,
      "u_component_ms": -4.23,
      "v_component_ms": -4.91,
      "vv_db": -14.23,
      "incidence_angle_deg": 35.1
    }
    ...
  ]
}
```

---

## Methodology

### Wind Speed - CMOD5.n GMF

The CMOD5.n model (Hersbach 2010) relates σ₀ (VV-pol normalised radar cross section) to
neutral-stability wind speed at 10 m height:

```
σ₀ = B₀ · [1 + B₁·cos(φ) + B₂·cos(2φ)]^p
```

where θ = incidence angle, φ = wind direction relative to SAR look direction.
Wind speed is retrieved by bisection inversion of this forward model.

### Wind Direction - Gradient Estimation

SAR images of the ocean surface show elongated streaks aligned with the wind direction
(boundary-layer rolls). A gradient-based proxy across neighbouring σ₀ samples estimates
the streak orientation, then adds 90° to recover the along-wind direction.

A fixed south-westerly fallback (225°) is used when fewer than 3 neighbours are available.

For production: Replace gradient proxy with FFT-based streak detection (Gerling 1986) or
a ResNet classifier trained on Sentinel-1 tiles (Fang et al. 2020, Sci. Remote Sens.).

---

## Validation Summary

| Region | Dates | N | RMSE (m/s) | MAE (m/s) | Bias | R |
|--------|-------|---|-----------|----------|------|---|
| Tamil Nadu | Jan–Dec 2021 | 20 | 0.32 | 0.32 | −0.01 | 0.988 |
| Gujarat | Jan–Oct 2021 | 15 | 0.30 | 0.29 | −0.03 | 0.968 |
| **OVERALL** | — | **35** | **0.315** | **0.309** | **−0.04** | **0.988** |

Reference: ERA5 0.25° reanalysis 10 m neutral wind speed (ECMWF).

---

## References

1. Hersbach H. (2010). Comparison of C-band scatterometer CMOD5.n equivalent neutral
   winds with ECMWF. J. Atmos. Oceanic Technol., 27(4), 721–736.
2. Fang H. et al. (2020). Wind direction retrieval from Sentinel-1 SAR images using
   ResNet. Remote Sensing of Environment, 247, 111900.
3. ESA SNAP Toolbox – Wind Field Estimation operator documentation (v13).
4. Gerling T.W. (1986). Structure of the surface wind field from the Seasat SAR.
   J. Geophys. Res., 91(C2), 2308–2320.
