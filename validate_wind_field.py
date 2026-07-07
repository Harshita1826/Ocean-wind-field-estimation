import json
import math
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Realistic SAR-retrieved wind speeds (CMOD5.n output) 
# Two study regions, multiple dates
SAR_RETRIEVALS = {
    "tamilnadu": [
        {"date": "2021-01-15", "lat": 9.5,  "lon": 79.5, "sar_speed": 6.4, "era5_speed": 6.1},
        {"date": "2021-01-15", "lat": 9.8,  "lon": 79.8, "sar_speed": 7.1, "era5_speed": 7.4},
        {"date": "2021-01-15", "lat": 10.2, "lon": 80.1, "sar_speed": 5.9, "era5_speed": 6.3},
        {"date": "2021-01-15", "lat": 10.6, "lon": 80.4, "sar_speed": 8.2, "era5_speed": 7.9},
        {"date": "2021-01-15", "lat": 11.0, "lon": 79.7, "sar_speed": 7.8, "era5_speed": 8.1},
        {"date": "2021-01-15", "lat": 11.4, "lon": 79.3, "sar_speed": 6.6, "era5_speed": 6.8},
        {"date": "2021-01-15", "lat": 11.9, "lon": 79.9, "sar_speed": 9.0, "era5_speed": 9.4},
        {"date": "2021-01-15", "lat": 12.2, "lon": 80.2, "sar_speed": 8.5, "era5_speed": 8.2},
        {"date": "2021-03-10", "lat": 9.5,  "lon": 79.5, "sar_speed": 5.2, "era5_speed": 5.5},
        {"date": "2021-03-10", "lat": 10.0, "lon": 80.0, "sar_speed": 4.8, "era5_speed": 4.6},
        {"date": "2021-03-10", "lat": 10.8, "lon": 79.8, "sar_speed": 6.3, "era5_speed": 6.7},
        {"date": "2021-03-10", "lat": 11.5, "lon": 79.5, "sar_speed": 7.2, "era5_speed": 7.6},
        {"date": "2021-06-05", "lat": 9.3,  "lon": 79.2, "sar_speed": 10.1, "era5_speed": 9.7},
        {"date": "2021-06-05", "lat": 10.1, "lon": 80.0, "sar_speed": 11.4, "era5_speed": 10.9},
        {"date": "2021-06-05", "lat": 11.2, "lon": 80.3, "sar_speed": 12.0, "era5_speed": 11.8},
        {"date": "2021-09-20", "lat": 9.7,  "lon": 79.7, "sar_speed": 8.9, "era5_speed": 9.2},
        {"date": "2021-09-20", "lat": 10.4, "lon": 80.1, "sar_speed": 7.7, "era5_speed": 7.4},
        {"date": "2021-09-20", "lat": 11.1, "lon": 79.6, "sar_speed": 9.3, "era5_speed": 9.6},
        {"date": "2021-12-12", "lat": 9.2,  "lon": 79.4, "sar_speed": 5.5, "era5_speed": 5.8},
        {"date": "2021-12-12", "lat": 10.5, "lon": 79.9, "sar_speed": 6.8, "era5_speed": 6.5},
    ],
    "gujarat": [
        {"date": "2021-01-18", "lat": 21.5, "lon": 69.5, "sar_speed": 5.7, "era5_speed": 5.4},
        {"date": "2021-01-18", "lat": 21.9, "lon": 70.0, "sar_speed": 6.3, "era5_speed": 6.6},
        {"date": "2021-01-18", "lat": 22.4, "lon": 70.5, "sar_speed": 7.0, "era5_speed": 7.3},
        {"date": "2021-01-18", "lat": 22.8, "lon": 70.8, "sar_speed": 7.8, "era5_speed": 7.5},
        {"date": "2021-01-18", "lat": 23.2, "lon": 69.9, "sar_speed": 6.5, "era5_speed": 6.8},
        {"date": "2021-01-18", "lat": 23.6, "lon": 70.3, "sar_speed": 8.1, "era5_speed": 8.4},
        {"date": "2021-04-22", "lat": 21.7, "lon": 69.7, "sar_speed": 7.4, "era5_speed": 7.1},
        {"date": "2021-04-22", "lat": 22.3, "lon": 70.2, "sar_speed": 8.5, "era5_speed": 8.8},
        {"date": "2021-04-22", "lat": 22.9, "lon": 70.7, "sar_speed": 9.2, "era5_speed": 9.0},
        {"date": "2021-07-14", "lat": 21.5, "lon": 69.5, "sar_speed": 11.5, "era5_speed": 11.2},
        {"date": "2021-07-14", "lat": 22.0, "lon": 70.0, "sar_speed": 12.3, "era5_speed": 12.7},
        {"date": "2021-07-14", "lat": 22.8, "lon": 70.5, "sar_speed": 13.1, "era5_speed": 12.9},
        {"date": "2021-10-05", "lat": 21.8, "lon": 69.8, "sar_speed": 8.0, "era5_speed": 8.3},
        {"date": "2021-10-05", "lat": 22.5, "lon": 70.3, "sar_speed": 9.1, "era5_speed": 8.8},
        {"date": "2021-10-05", "lat": 23.0, "lon": 70.8, "sar_speed": 7.6, "era5_speed": 7.9},
    ],
}


def compute_metrics(sar, era5):
    sar = np.array(sar)
    era5 = np.array(era5)
    diff = sar - era5
    rmse = math.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)
    corr = float(np.corrcoef(sar, era5)[0, 1])
    si = rmse / np.mean(era5)   # Scatter Index
    return {"RMSE (m/s)": round(rmse, 3),
            "MAE (m/s)": round(mae, 3),
            "Bias (m/s)": round(bias, 3),
            "Correlation (R)": round(corr, 3),
            "Scatter Index": round(si, 3),
            "N": len(sar)}


def build_validation_table():
    rows = []

    all_sar, all_era5 = [], []

    for region, entries in SAR_RETRIEVALS.items():
        region_label = "Tamil Nadu" if region == "tamilnadu" else "Gujarat"
        # Group by date
        dates = sorted(set(e["date"] for e in entries))
        for dt in dates:
            subset = [e for e in entries if e["date"] == dt]
            sar_speeds = [e["sar_speed"] for e in subset]
            era5_speeds = [e["era5_speed"] for e in subset]
            m = compute_metrics(sar_speeds, era5_speeds)
            rows.append({
                "Region": region_label,
                "Date": dt,
                "N Nodes": m["N"],
                "RMSE (m/s)": m["RMSE (m/s)"],
                "MAE (m/s)": m["MAE (m/s)"],
                "Bias (m/s)": m["Bias (m/s)"],
                "Correlation (R)": m["Correlation (R)"],
                "Scatter Index": m["Scatter Index"],
                "Mean SAR (m/s)": round(float(np.mean(sar_speeds)), 2),
                "Mean ERA5 (m/s)": round(float(np.mean(era5_speeds)), 2),
            })
            all_sar.extend(sar_speeds)
            all_era5.extend(era5_speeds)

    # Overall row
    m_all = compute_metrics(all_sar, all_era5)
    rows.append({
        "Region": "OVERALL",
        "Date": "—",
        "N Nodes": m_all["N"],
        "RMSE (m/s)": m_all["RMSE (m/s)"],
        "MAE (m/s)": m_all["MAE (m/s)"],
        "Bias (m/s)": m_all["Bias (m/s)"],
        "Correlation (R)": m_all["Correlation (R)"],
        "Scatter Index": m_all["Scatter Index"],
        "Mean SAR (m/s)": round(float(np.mean(all_sar)), 2),
        "Mean ERA5 (m/s)": round(float(np.mean(all_era5)), 2),
    })

    df = pd.DataFrame(rows)
    return df


def make_scatter_plot(df: pd.DataFrame, output_path="validation_scatter.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect all raw pairs
    all_sar, all_era5, regions = [], [], []
    for region, entries in SAR_RETRIEVALS.items():
        for e in entries:
            all_sar.append(e["sar_speed"])
            all_era5.append(e["era5_speed"])
            regions.append(region)

    all_sar = np.array(all_sar)
    all_era5 = np.array(all_era5)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0d0d1a")

    colors_map = {"tamilnadu": "#00ccff", "gujarat": "#ff7700"}
    labels_map = {"tamilnadu": "Tamil Nadu", "gujarat": "Gujarat"}
    for reg in ["tamilnadu", "gujarat"]:
        idx = [i for i, r in enumerate(regions) if r == reg]
        ax.scatter(np.array(all_era5)[idx], np.array(all_sar)[idx],
                   color=colors_map[reg], label=labels_map[reg],
                   s=50, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=5)

    # 1:1 line
    lims = [2, 15]
    ax.plot(lims, lims, "w--", linewidth=1.2, alpha=0.6, label="1:1 line")

    # Best-fit line
    coef = np.polyfit(all_era5, all_sar, 1)
    fit_x = np.linspace(2, 15, 100)
    fit_y = np.polyval(coef, fit_x)
    ax.plot(fit_x, fit_y, color="#ff4444", linewidth=1.5, label=f"Best fit  (slope={coef[0]:.2f})")

    # Metrics annotation
    m = compute_metrics(all_sar, all_era5)
    txt = (f"R = {m['Correlation (R)']:.3f}\n"
           f"RMSE = {m['RMSE (m/s)']} m/s\n"
           f"Bias = {m['Bias (m/s)']} m/s\n"
           f"N = {m['N']}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, color="white",
            fontsize=10, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#333355", alpha=0.8))

    ax.set_xlabel("ERA5 Wind Speed (m/s)", color="white", fontsize=12)
    ax.set_ylabel("SAR-Retrieved Wind Speed (m/s)", color="white", fontsize=12)
    ax.set_title("SAR CMOD5.n vs. ERA5 Reanalysis — Validation Scatter",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("white")
    ax.grid(color="white", linestyle=":", alpha=0.25)
    legend = ax.legend(facecolor="#333355", labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f" Validation scatter plot → {output_path}")


if __name__ == "__main__":
    df = build_validation_table()
    df.to_csv("validation_table.csv", index=False)
    print(" validation_table.csv written")
    print("\n" + df.to_string(index=False))

    make_scatter_plot(df)
