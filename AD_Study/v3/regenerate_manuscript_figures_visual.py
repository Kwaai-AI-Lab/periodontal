from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


OUT_DIR = Path("AD_Model_v3/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("AD_Model_v3/results")


def load_scenario_data():
    """Load data from all three PD prevalence scenario Excel files."""
    scenarios = {}
    for pd_level in [25, 50, 75]:
        excel_path = RESULTS_DIR / f"Figure_Generation_PD_{pd_level}.xlsx"
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        scenarios[pd_level] = pd.read_excel(excel_path, sheet_name="Summary")
    return scenarios


def style_axes(ax):
    ax.set_facecolor("#e9e9e9")
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_alpha(0.6)


def save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight", facecolor="#e9e9e9")
    plt.close(fig)


def draw_rounded_box(ax, xy, w, h, text, fc, ec, text_color="#222", fs=11, fw="normal"):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.01",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", color=text_color, fontsize=fs, weight=fw)
    return box


def arrow(ax, p1, p2, color, rad=0.0):
    arr = FancyArrowPatch(
        p1,
        p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="->",
        mutation_scale=12,
        lw=1.4,
        color=color,
        alpha=0.9,
    )
    ax.add_patch(arr)


def figure_1():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    fig.patch.set_facecolor("#e9e9e9")
    ax.set_facecolor("#e9e9e9")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Headers
    ax.text(0.07, 0.94, "CARNALL FARRAR CATEGORIES", fontsize=9, color="#6a4a3a", weight="bold", ha="center")
    ax.text(0.43, 0.94, "REGROUPED AS", fontsize=9, color="#444", weight="bold", ha="center")
    ax.text(0.77, 0.94, "APPLIED PER SETTING", fontsize=9, color="#444", weight="bold", ha="center")

    # Left column
    b1 = draw_rounded_box(ax, (0.02, 0.72), 0.16, 0.11, "Social care", "#ece7e1", "#cfc8be", fs=10)
    b2 = draw_rounded_box(ax, (0.02, 0.54), 0.16, 0.11, "Healthcare", "#ece7e1", "#cfc8be", fs=10)
    b3 = draw_rounded_box(ax, (0.02, 0.36), 0.16, 0.11, "Unpaid care", "#ece7e1", "#cfc8be", fs=10)
    b4 = draw_rounded_box(ax, (0.02, 0.18), 0.16, 0.11, "Quality of life costs", "#ece7e1", "#cfc8be", fs=10)
    b5 = draw_rounded_box(ax, (0.02, 0.01), 0.16, 0.10, "Economic costs", "#e6dfd8", "#d6c9bd", text_color="#9e8e81", fs=10)
    ax.text(0.10, -0.03, "(excluded from our model)", ha="center", va="top", fontsize=8, color="#b39f90")

    # Center column
    fbox = draw_rounded_box(ax, (0.37, 0.62), 0.16, 0.17, "FORMAL COSTS", "#dcecf5", "#2c7da0", text_color="#1f5f78", fs=11, fw="bold")
    ax.text(0.45, 0.65, "Social care + Healthcare", ha="center", va="center", fontsize=9, color="#4b7f95")

    ibox = draw_rounded_box(ax, (0.37, 0.34), 0.16, 0.17, "INFORMAL COSTS", "#f2e7f0", "#8d4a89", text_color="#7a2f76", fs=11, fw="bold")
    ax.text(0.45, 0.37, "Unpaid care + QoL costs", ha="center", va="center", fontsize=9, color="#8d5c88")

    # Right column
    hb = draw_rounded_box(ax, (0.72, 0.64), 0.22, 0.18, "HOME CARE", "#1f6f8b", "#1f6f8b", text_color="white", fs=12, fw="bold")
    ax.text(0.77, 0.68, "Formal", color="#cde6ef", fontsize=9, ha="center")
    ax.text(0.89, 0.68, "Informal", color="#f6d9e7", fontsize=9, ha="center")
    ax.plot([0.83, 0.83], [0.65, 0.79], color="#6a93a3", alpha=0.4)

    ib = draw_rounded_box(ax, (0.72, 0.36), 0.22, 0.18, "INSTITUTIONAL CARE", "#7a2e78", "#7a2e78", text_color="white", fs=11, fw="bold")
    ax.text(0.77, 0.40, "Formal", color="#cde6ef", fontsize=9, ha="center")
    ax.text(0.89, 0.40, "Informal", color="#f6d9e7", fontsize=9, ha="center")
    ax.plot([0.83, 0.83], [0.37, 0.51], color="#a57da3", alpha=0.4)

    # Arrows
    arrow(ax, (0.18, 0.78), (0.37, 0.72), "#2c7da0", rad=-0.05)
    arrow(ax, (0.18, 0.60), (0.37, 0.70), "#2c7da0", rad=0.06)
    arrow(ax, (0.18, 0.42), (0.37, 0.42), "#8d4a89", rad=0.02)
    arrow(ax, (0.18, 0.24), (0.37, 0.40), "#8d4a89", rad=-0.08)

    arrow(ax, (0.53, 0.70), (0.72, 0.73), "#2c7da0", rad=0.0)
    arrow(ax, (0.53, 0.66), (0.72, 0.46), "#2c7da0", rad=-0.07)
    arrow(ax, (0.53, 0.42), (0.72, 0.71), "#8d4a89", rad=0.16)
    arrow(ax, (0.53, 0.40), (0.72, 0.44), "#8d4a89", rad=0.02)

    save(fig, "figure_1.png")


def figure_2(scenarios):
    # Extract data from year 2040 (last row) of 50% PD baseline scenario
    df_baseline = scenarios[50]
    row_2040 = df_baseline[df_baseline['calendar_year'] == 2040].iloc[0]

    # Map display names to column suffixes
    risk_factor_map = {
        "Periodontal Disease": "periodontal_disease",
        "Hypertension": "hypertension",
        "Hearing Difficulty": "hearing_difficulty",
        "APOE e4": "APOE_e4_carrier",
        "Obesity": "obesity",
        "Depression": "depression",
        "Diabetes": "diabetes"
    }

    factors = list(risk_factor_map.keys())
    general = np.array([row_2040[f'risk_prev_alive_{risk_factor_map[f]}'] * 100 for f in factors])
    dementia = np.array([row_2040[f'risk_prev_dementia_{risk_factor_map[f]}'] * 100 for f in factors])
    enrichment = ((dementia - general) / general) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.8), gridspec_kw={"wspace": 0.45})
    fig.patch.set_facecolor("#e9e9e9")
    style_axes(ax1)
    style_axes(ax2)

    y = np.arange(len(factors))

    # highlight PD rows
    pd_idx = factors.index("Periodontal Disease")
    ax1.axhspan(pd_idx - 0.6, pd_idx + 0.6, color="#efe7b0", alpha=0.55, zorder=0)

    h = 0.28
    ax1.barh(y + h / 2, general, height=h, color="#4ea0d8", edgecolor="#2f4f66", label="General Population")
    ax1.barh(y - h / 2, dementia, height=h, color="#ee6f5f", edgecolor="#5b322d", label="Dementia Population")
    ax1.set_yticks(y)
    ax1.set_yticklabels(factors, fontsize=10)
    ax1.get_yticklabels()[pd_idx].set_color("#ce7e00")
    ax1.get_yticklabels()[pd_idx].set_fontweight("bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0, 85)
    ax1.set_xlabel("Prevalence (%)", fontsize=11, weight="bold")
    ax1.set_title("A) Prevalence in General vs Dementia Populations", fontsize=12, weight="bold", pad=12)
    ax1.legend(loc="lower right", framealpha=0.9)

    for i, v in enumerate(general):
        ax1.text(v + 0.8, i + h / 2, f"{v:.1f}%", va="center", fontsize=9, weight="bold")
    for i, v in enumerate(dementia):
        ax1.text(v + 0.8, i - h / 2, f"{v:.1f}%", va="center", fontsize=9, weight="bold")

    order = np.argsort(enrichment)
    f2 = [factors[i] for i in order]
    e2 = enrichment[order]
    colors = ["#58b980" if f != "APOE e4" else "#9aa7ab" for f in f2]

    pd_idx2 = f2.index("Periodontal Disease")
    ax2.axhspan(pd_idx2 - 0.6, pd_idx2 + 0.6, color="#efe7b0", alpha=0.55, zorder=0)
    ax2.barh(np.arange(len(f2)), e2, color=colors, edgecolor="#3a4a4a")
    ax2.set_yticks(np.arange(len(f2)))
    ax2.set_yticklabels(f2, fontsize=10)
    ax2.get_yticklabels()[pd_idx2].set_color("#ce7e00")
    ax2.get_yticklabels()[pd_idx2].set_fontweight("bold")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Relative Enrichment (%)", fontsize=11, weight="bold")
    ax2.set_title("B) Enrichment in Dementia Population", fontsize=12, weight="bold", pad=12)

    for i, v in enumerate(e2):
        ax2.text(v + 1.5, i, f"+{v:.1f}%", va="center", fontsize=10, weight="bold")

    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor="#58b980", label="Modifiable"), Patch(facecolor="#9aa7ab", label="Non-modifiable (genetic)")],
               loc="lower right", framealpha=0.9)

    # fig.suptitle("Risk Factor Landscape at Current 50% PD Baseline\nEngland, Adults Aged 65+, Year 2040", fontsize=15, weight="bold", y=0.995)
    save(fig, "figure_2.png")


def figure_3(scenarios):
    # Extract yearly cost data from all three scenarios
    df_25 = scenarios[25][scenarios[25]['calendar_year'] >= 2024]
    df_50 = scenarios[50][scenarios[50]['calendar_year'] >= 2024]
    df_75 = scenarios[75][scenarios[75]['calendar_year'] >= 2024]

    years = df_50['calendar_year'].values
    # Convert to billions
    low25 = (df_25['year_costs_societal'].values / 1e9)
    base50 = (df_50['year_costs_societal'].values / 1e9)
    high75 = (df_75['year_costs_societal'].values / 1e9)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    fig.patch.set_facecolor("#e9e9e9")
    style_axes(ax)

    ax.plot(years, low25, color="#2f8ab3", marker="o", markersize=3.8, linewidth=2.6, label="25% PD")
    ax.plot(years, base50, color="#e4831f", marker="s", markersize=3.8, linewidth=3.0, label="50% PD (Baseline)")
    ax.plot(years, high75, color="#cf4020", marker="^", markersize=4.1, linewidth=2.6, label="75% PD")

    ax.fill_between(years, low25, base50, color="#6ea46f", alpha=0.35)
    ax.fill_between(years, base50, high75, color="#ed7f77", alpha=0.35)

    ax.set_xlim(2023.5, 2040.5)
    ax.set_ylim(26.5, 45.5)
    yticks = [27, 30, 32, 35, 37, 40, 42]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"GBP {t}B" for t in yticks], fontsize=11)

    ax.set_xlabel("Year", fontsize=12, weight="bold")
    ax.set_ylabel("Annual Total Costs (GBP  billions)", fontsize=13, weight="bold")
    # ax.set_title("Annual Dementia Costs by Periodontal Disease Prevalence Scenario\nEngland, Adults Aged 65+, 2024-2040",
    #              fontsize=16, weight="bold", pad=14)
    ax.legend(loc="upper left", framealpha=0.95, fancybox=True, shadow=True, fontsize=11)

    save(fig, "figure_3.png")


def figure_4(scenarios):
    # Select specific years for plotting
    years = np.array([2024, 2027, 2030, 2033, 2036, 2039, 2040])

    # Extract cumulative QALY data for selected years
    df_25 = scenarios[25][scenarios[25]['calendar_year'].isin(years)]
    df_50 = scenarios[50][scenarios[50]['calendar_year'].isin(years)]
    df_75 = scenarios[75][scenarios[75]['calendar_year'].isin(years)]

    # Calculate differences from baseline (50% scenario), convert to millions
    patient_25 = (df_25['total_qalys_patient'].values - df_50['total_qalys_patient'].values) / 1e6
    patient_75 = (df_75['total_qalys_patient'].values - df_50['total_qalys_patient'].values) / 1e6

    caregiver_25 = (df_25['total_qalys_caregiver'].values - df_50['total_qalys_caregiver'].values) / 1e6
    caregiver_75 = (df_75['total_qalys_caregiver'].values - df_50['total_qalys_caregiver'].values) / 1e6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.6, 7.9), sharex=True)
    fig.patch.set_facecolor("#e9e9e9")
    style_axes(ax1)
    style_axes(ax2)

    # Panel A
    ax1.plot(years, patient_25, color="#2c8db7", marker="o", markersize=3, linewidth=2.5, label="25% PD vs Baseline")
    ax1.plot(years, patient_75, color="#d44a2b", marker="^", markersize=3, linewidth=2.5, label="75% PD vs Baseline")
    ax1.axhline(0, color="#666", linestyle="--", alpha=0.8, label="50% Baseline")
    ax1.fill_between(years, 0, patient_25, color="#7d7fda", alpha=0.35)
    ax1.fill_between(years, 0, patient_75, color="#e88d8b", alpha=0.38)
    ax1.set_ylim(-0.30, 0.30)
    ax1.set_ylabel("Cumulative Patient QALY\nDifference from Baseline (millions)", weight="bold")
    ax1.set_title("A) Patient QALYs: Minimal Variation from Baseline", fontsize=14, weight="bold", pad=8)
    ax1.legend(loc="upper left", framealpha=0.95, fancybox=True, shadow=True)
    ax1.text(2040.1, patient_25[-1], "+0.16M", color="blue", va="center", fontsize=11, weight="bold")
    ax1.text(2040.1, patient_75[-1], "-0.17M", color="red", va="center", fontsize=11, weight="bold")
    ax1.text(2038.0, 0.23, "Total range: 0.16 to -0.17M\n(<0.2% of baseline)", fontsize=10,
             style="italic", ha="center", va="bottom",
             bbox=dict(boxstyle="round", facecolor="#efe3c6", edgecolor="#8e8164", alpha=0.95))

    # Panel B
    ax2.plot(years, caregiver_25, color="#2c8db7", marker="o", markersize=3, linewidth=2.5, label="25% PD vs Baseline")
    ax2.plot(years, caregiver_75, color="#d44a2b", marker="^", markersize=3, linewidth=2.5, label="75% PD vs Baseline")
    ax2.axhline(0, color="#666", linestyle="--", alpha=0.8, label="50% Baseline")
    ax2.fill_between(years, 0, caregiver_25, color="#7d7fda", alpha=0.35)
    ax2.fill_between(years, 0, caregiver_75, color="#e88d8b", alpha=0.38)
    ax2.set_ylim(-0.38, 0.38)
    ax2.set_ylabel("Cumulative Caregiver QALY\nDifference from Baseline (millions)", weight="bold")
    ax2.set_xlabel("Year", fontsize=12, weight="bold")
    ax2.set_title("B) Caregiver QALYs: Inverse Relationship with PD Prevalence", fontsize=14, weight="bold", pad=8)
    ax2.legend(loc="lower left", framealpha=0.95, fancybox=True, shadow=True)
    ax2.text(2040.1, caregiver_25[-1], "-0.36M", color="blue", va="center", fontsize=11, weight="bold")
    ax2.text(2040.1, caregiver_75[-1], "+0.35M", color="red", va="center", fontsize=11, weight="bold")
    ax2.text(2032.0, -0.36, "Total range: -0.36 to 0.35M\n(~7% of baseline)", fontsize=10,
             style="italic", ha="center", va="bottom",
             bbox=dict(boxstyle="round", facecolor="#efe3c6", edgecolor="#8e8164", alpha=0.95))

    # fig.suptitle("Cumulative QALY Differences from 50% Baseline Over Time\nEngland, Adults Aged 65+, 2024-2040",
    #              fontsize=16, weight="bold", y=0.99)
    # fig.text(0.5, 0.02,
    #          "Lower PD prevalence reduces caregiver QALYs (fewer caregivers needed) while patient QALYs remain stable.",
    #          ha="center", fontsize=10, style="italic")

    save(fig, "figure_4.png")


def main():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "figure.facecolor": "#e9e9e9",
    })

    print("Loading data from Excel files...")
    scenarios = load_scenario_data()
    print(f"  Loaded PD_25: {len(scenarios[25])} rows")
    print(f"  Loaded PD_50: {len(scenarios[50])} rows")
    print(f"  Loaded PD_75: {len(scenarios[75])} rows")

    print("Generating Figure 1 (conceptual diagram)...")
    figure_1()
    print("Generating Figure 2 (risk factor prevalence)...")
    figure_2(scenarios)
    print("Generating Figure 3 (annual costs)...")
    figure_3(scenarios)
    print("Generating Figure 4 (cumulative QALYs)...")
    figure_4(scenarios)
    print(f"\nSaved regenerated figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()

