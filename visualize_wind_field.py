"""
visualize_wind_field.py
========================
Generates a publication-quality wind-field map from:
  (a) A JSON response file produced by the /api/wind-field endpoint, OR
  (b) Simulated data (when run without arguments) for demonstration.

Output: wind_field_<region>_<date>.png  (matches sample-results style)

Usage:
  python visualize_wind_field.py                        # demo mode
  python visualize_wind_field.py response.json          # real API data
  python visualize_wind_field.py --region gujarat       # Tamil Nadu or Gujarat demo
"""

import sys
import json
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter

# ── Colour map matching the sample result (blue→cyan→green→yellow→red) ──
WIND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "wind_speed",
    ["#0000FF", "#00AAFF", "#00FFAA", "#AAFF00", "#FFAA00", "#FF0000"],
)

# ── Preset study areas ──
PRESETS = {
    "tamilnadu": {
        "name": "Tamil Nadu Coast",
        "min_lat": 8.0, "max_lat": 13.5,
        "min_lon": 77.5, "max_lon": 82.0,
        "date": "2021-01-15",
    },
    "gujarat": {
        "name": "Gujarat Coast",
        "min_lat": 20.0, "max_lat": 24.5,
        "min_lon": 68.0, "max_lon": 73.5,
        "date": "2021-01-18",
    },
}


# ────────────────────────────────────────────────────────────────
# Synthetic data generator (reproduces expected wind patterns)
# ────────────────────────────────────────────────────────────────
def generate_synthetic_wind_field(preset_key: str = "tamilnadu") -> dict:
    """
    Produce a physically plausible synthetic wind field for the given region.
    Speeds 3–12 m/s, dominant south-westerly (225°) with spatial variation.
    """
    p = PRESETS[preset_key]
    np.random.seed(42)

    n_lat, n_lon = 18, 22
    lats = np.linspace(p["min_lat"] + 0.2, p["max_lat"] - 0.2, n_lat)
    lons = np.linspace(p["min_lon"] + 0.2, p["max_lon"] - 0.2, n_lon)

    # Smooth spatial wind speed field
    raw = np.random.rand(n_lat, n_lon) * 4 + 5  # 5–9 m/s base
    speed_field = gaussian_filter(raw, sigma=2.5)

    # Add a coastal jet feature
    for i, lat in enumerate(lats):
        coastal_boost = 2.0 * math.exp(-((lat - (p["min_lat"] + 2.5)) ** 2) / 4)
        speed_field[i, :] += coastal_boost

    speed_field = np.clip(speed_field, 2.0, 14.0)

    # Wind direction: dominant 210–240° with spatial swirl
    dir_base = 220.0 if preset_key == "tamilnadu" else 235.0
    dir_noise = gaussian_filter(np.random.randn(n_lat, n_lon) * 25, sigma=3)
    dir_field = (dir_base + dir_noise) % 360

    data = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            spd = float(speed_field[i, j])
            direction = float(dir_field[i, j])
            dir_math = (270.0 - direction) % 360.0
            rad = math.radians(dir_math)
            u = spd * math.cos(rad)
            v = spd * math.sin(rad)
            data.append({
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "wind_speed_ms": round(spd, 2),
                "wind_direction_deg": round(direction, 1),
                "u_component_ms": round(u, 2),
                "v_component_ms": round(v, 2),
            })

    return {
        "status": "success",
        "image_date": p["date"],
        "study_area": {
            "min_lat": p["min_lat"], "max_lat": p["max_lat"],
            "min_lon": p["min_lon"], "max_lon": p["max_lon"],
        },
        "statistics": {
            "total_nodes": len(data),
            "mean_wind_speed_ms": round(float(np.mean(speed_field)), 2),
            "min_wind_speed_ms": round(float(np.min(speed_field)), 2),
            "max_wind_speed_ms": round(float(np.max(speed_field)), 2),
        },
        "data": data,
        "_region_name": p["name"],
        "_region_key": preset_key,
    }


# ────────────────────────────────────────────────────────────────
# Interpolate scattered nodes onto a regular grid for background
# ────────────────────────────────────────────────────────────────
def interpolate_to_grid(lons, lats, values, res: int = 200):
    from scipy.interpolate import griddata
    lon_min, lon_max = min(lons) - 0.1, max(lons) + 0.1
    lat_min, lat_max = min(lats) - 0.1, max(lats) + 0.1
    grid_lon = np.linspace(lon_min, lon_max, res)
    grid_lat = np.linspace(lat_min, lat_max, res)
    GL, GT = np.meshgrid(grid_lon, grid_lat)
    points = np.column_stack([lons, lats])
    grid_vals = griddata(points, values, (GL, GT), method="cubic")
    # Fill NaNs at edges
    grid_vals_nn = griddata(points, values, (GL, GT), method="nearest")
    grid_vals = np.where(np.isnan(grid_vals), grid_vals_nn, grid_vals)
    return GL, GT, grid_vals


# ────────────────────────────────────────────────────────────────
# Main plotting routine
# ────────────────────────────────────────────────────────────────
def plot_wind_field(api_response: dict, output_path: str = None):
    data = api_response["data"]
    area = api_response["study_area"]
    date = api_response.get("image_date", api_response.get("request_date", ""))
    region_name = api_response.get("_region_name", "Indian Coast")
    stats = api_response.get("statistics", {})

    lats = np.array([d["latitude"] for d in data])
    lons = np.array([d["longitude"] for d in data])
    speeds = np.array([d["wind_speed_ms"] for d in data])
    dirs = np.array([d["wind_direction_deg"] for d in data])

    # Derived u/v for quiver (pointing TO direction wind blows)
    # Convention: arrows show where wind is going (oceanographic style)
    u = np.array([d["u_component_ms"] for d in data])
    v = np.array([d["v_component_ms"] for d in data])
    # Normalize for uniform-length arrows (speed encoded by colour)
    mag = np.sqrt(u**2 + v**2)
    mag = np.where(mag < 0.1, 1.0, mag)
    u_norm = u / mag
    v_norm = v / mag

    # Clamp speed range for colour scale
    vmin, vmax = 2.0, 14.0

    # ── Colour mapping per arrow ────────────────────────────────
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = WIND_CMAP(norm(speeds))

    # ── Interpolated background ─────────────────────────────────
    GL, GT, grid_spd = interpolate_to_grid(lons, lats, speeds)
    grid_spd = np.clip(grid_spd, vmin, vmax)

    # ── Figure setup ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0d0d1a")

    # ── Background speed heatmap ────────────────────────────────
    ax.pcolormesh(GL, GT, grid_spd, cmap=WIND_CMAP, norm=norm,
                  alpha=0.45, shading="gouraud")

    # ── Wind vector arrows ──────────────────────────────────────
    qv = ax.quiver(
        lons, lats, u_norm, v_norm,
        color=colors,
        scale=30,
        width=0.003,
        headwidth=4,
        headlength=5,
        headaxislength=4,
        alpha=0.92,
        zorder=5,
    )

    # ── Scatter dots at each node ────────────────────────────────
    sc = ax.scatter(lons, lats, c=speeds, cmap=WIND_CMAP, norm=norm,
                    s=18, zorder=6, linewidths=0.3, edgecolors="white", alpha=0.85)

    # ── Colour bar ──────────────────────────────────────────────
    sm = ScalarMappable(cmap=WIND_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        pad=0.10, fraction=0.04, aspect=40)
    cbar.set_label("Wind Speed (m/s)", color="white", fontsize=12, labelpad=6)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white", fontsize=10)
    cbar.outline.set_edgecolor("white")

    # ── Grid lines ──────────────────────────────────────────────
    ax.grid(color="white", linestyle=":", linewidth=0.4, alpha=0.3)

    # ── Axes formatting ─────────────────────────────────────────
    ax.set_xlim(area["min_lon"] - 0.1, area["max_lon"] + 0.1)
    ax.set_ylim(area["min_lat"] - 0.1, area["max_lat"] + 0.1)

    def fmt_lon(x, _=None):
        return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"
    def fmt_lat(y, _=None):
        return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_lat))
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # ── Title & subtitles ────────────────────────────────────────
    ax.set_title(
        f"Ocean Wind Field — {region_name}",
        color="white", fontsize=15, fontweight="bold", pad=14,
    )
    subtitle = (
        f"Sentinel-1 SAR · CMOD5.n GMF  ·  {date}  ·  "
        f"Nodes: {stats.get('total_nodes', len(data))}  ·  "
        f"Mean: {stats.get('mean_wind_speed_ms', '—')} m/s  ·  "
        f"Range: {stats.get('min_wind_speed_ms', '—')}–{stats.get('max_wind_speed_ms', '—')} m/s"
    )
    ax.set_xlabel(subtitle, color="#aaaacc", fontsize=9, labelpad=18)

    # ── Reference arrow ─────────────────────────────────────────
    ax.quiverkey(qv, X=0.88, Y=1.04, U=1, label="Wind Vector", labelpos="E",
                 color="white", labelcolor="white", fontproperties={"size": 9})

    plt.tight_layout()

    if output_path is None:
        region_key = api_response.get("_region_key", "coast")
        safe_date = date.replace("-", "")
        output_path = f"wind_field_{region_key}_{safe_date}.png"

    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Wind field map saved → {output_path}")
    return output_path


# ────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise Sentinel-1 wind field")
    parser.add_argument("json_file", nargs="?", help="API response JSON file")
    parser.add_argument("--region", choices=["tamilnadu", "gujarat"],
                        default="tamilnadu", help="Demo region (used when no JSON provided)")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    if args.json_file:
        with open(args.json_file) as fh:
            api_resp = json.load(fh)
    else:
        print(f"[i] No JSON file supplied — generating synthetic wind field for '{args.region}'")
        api_resp = generate_synthetic_wind_field(args.region)

    plot_wind_field(api_resp, output_path=args.output)
