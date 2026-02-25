"""
Generate manuscript Figure 2, Figure 3, and Figure 4 by re-running the model.

This script runs three scenarios by changing periodontal disease prevalence:
- 25% PD
- 50% PD (baseline)
- 75% PD

It then creates:
- figure_2.png  (risk factor landscape at 2040 for 50% PD baseline)
- figure_3.png  (annual societal costs by scenario, 2024-2040)
- figure_4.png  (cumulative QALY differences vs baseline, 2024-2040)

Optional: exports underlying data to Excel.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IBM_PD_AD_v3 import general_config, run_model, summaries_to_dataframe


FIGURE_RISK_FACTORS = [
    ("Periodontal Disease", "periodontal_disease", True),
    ("Hypertension", "hypertension", True),
    ("Hearing Difficulty", "hearing_difficulty", True),
    ("APOE e4", "APOE_e4_carrier", False),
    ("Obesity", "obesity", True),
    ("Depression", "depression", True),
    ("Diabetes", "diabetes", True),
]


def _style_axes(ax) -> None:
    ax.set_facecolor("#e9e9e9")
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_alpha(0.6)


def _prepare_config(pd_prevalence: float, population_fraction: float) -> dict:
    cfg = copy.deepcopy(general_config)

    if not (0 < population_fraction <= 1.0):
        raise ValueError("population_fraction must be in (0, 1].")
    cfg["population"] = max(1, int(round(float(cfg["population"]) * population_fraction)))

    pd_cfg = cfg["risk_factors"]["periodontal_disease"]
    pd_cfg["prevalence"] = {"female": pd_prevalence, "male": pd_prevalence}
    return cfg


def _run_scenario(pd_prevalence: float, seed: int, population_fraction: float) -> pd.DataFrame:
    cfg = _prepare_config(pd_prevalence=pd_prevalence, population_fraction=population_fraction)
    result = run_model(cfg, seed=seed)

    # Export full results to Excel with unique filename (for journal reproducibility)
    from IBM_PD_AD_v3 import export_results_to_excel
    from pathlib import Path
    excel_file = Path("results") / f"Figure_Generation_PD_{int(pd_prevalence*100)}.xlsx"
    excel_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_results_to_excel(result, path=str(excel_file))
        print(f"  Exported full results to: {excel_file}")
    except Exception as e:
        print(f"  Warning: Excel export failed for {pd_prevalence:.0%} PD: {e}")

    df = summaries_to_dataframe(result).sort_values("time_step").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"Model returned no summaries for scenario {pd_prevalence:.2f}.")
    return df


def _extract_figure_2_data(baseline_df: pd.DataFrame, baseline_year: int) -> pd.DataFrame:
    row = baseline_df.loc[baseline_df["calendar_year"] == baseline_year]
    if row.empty:
        available = sorted(int(y) for y in baseline_df["calendar_year"].unique())
        raise ValueError(f"Year {baseline_year} not found. Available years: {available}")
    row = row.iloc[0]

    records = []
    for label, key, modifiable in FIGURE_RISK_FACTORS:
        g = float(row.get(f"risk_prev_alive_{key}", 0.0) or 0.0) * 100.0
        d = float(row.get(f"risk_prev_dementia_{key}", 0.0) or 0.0) * 100.0
        enrichment = ((d / g) - 1.0) * 100.0 if g > 0 else 0.0
        records.append(
            {
                "risk_factor": label,
                "general_prevalence_pct": g,
                "dementia_prevalence_pct": d,
                "enrichment_pct": enrichment,
                "modifiable": bool(modifiable),
            }
        )
    return pd.DataFrame(records)


def _extract_figure_3_data(scenario_dfs: Dict[str, pd.DataFrame], start_year: int, end_year: int) -> pd.DataFrame:
    rows = []
    for scenario_name, df in scenario_dfs.items():
        d = df[(df["calendar_year"] >= start_year) & (df["calendar_year"] <= end_year)].copy()
        for _, r in d.iterrows():
            rows.append(
                {
                    "year": int(r["calendar_year"]),
                    "scenario": scenario_name,
                    "annual_total_societal_cost_gbp_bn": float(r.get("year_costs_societal", 0.0) or 0.0) / 1e9,
                }
            )
    out = pd.DataFrame(rows).sort_values(["year", "scenario"]).reset_index(drop=True)
    return out


def _extract_figure_4_data(
    scenario_dfs: Dict[str, pd.DataFrame], start_year: int, end_year: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = scenario_dfs["50% PD"].copy()
    base = base[(base["calendar_year"] >= start_year) & (base["calendar_year"] <= end_year)]
    base = base.set_index("calendar_year")

    rows = []
    for scenario in ("25% PD", "75% PD"):
        cur = scenario_dfs[scenario].copy()
        cur = cur[(cur["calendar_year"] >= start_year) & (cur["calendar_year"] <= end_year)]
        cur = cur.set_index("calendar_year")
        merged = cur.join(
            base[["total_qalys_patient", "total_qalys_caregiver"]],
            how="inner",
            rsuffix="_base",
        )
        for year, r in merged.iterrows():
            rows.append(
                {
                    "year": int(year),
                    "qaly_type": "patient",
                    "scenario": scenario,
                    "cumulative_difference_millions": (
                        float(r["total_qalys_patient"] - r["total_qalys_patient_base"]) / 1e6
                    ),
                }
            )
            rows.append(
                {
                    "year": int(year),
                    "qaly_type": "caregiver",
                    "scenario": scenario,
                    "cumulative_difference_millions": (
                        float(r["total_qalys_caregiver"] - r["total_qalys_caregiver_base"]) / 1e6
                    ),
                }
            )

    plot_df = pd.DataFrame(rows).sort_values(["year", "qaly_type", "scenario"]).reset_index(drop=True)
    return plot_df, base.reset_index()


def _plot_figure_2(df: pd.DataFrame, out_path: Path, baseline_year: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.8), gridspec_kw={"wspace": 0.33})
    fig.patch.set_facecolor("#e9e9e9")
    _style_axes(ax1)
    _style_axes(ax2)

    factors = df["risk_factor"].tolist()
    general = df["general_prevalence_pct"].to_numpy()
    dementia = df["dementia_prevalence_pct"].to_numpy()
    enrichment = df["enrichment_pct"].to_numpy()

    y = np.arange(len(factors))
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
    ax1.set_xlim(0, max(float(np.max(dementia)), float(np.max(general))) * 1.55)
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
    max_enrichment = max(100.0, float(np.max(e2) * 1.15))
    ax2.set_xlim(0, max_enrichment)
    ax2.set_xlabel("Relative Enrichment (%)", fontsize=11, weight="bold")
    ax2.set_title("B) Enrichment in Dementia Population", fontsize=12, weight="bold", pad=12)

    for i, v in enumerate(e2):
        ax2.text(v + 1.2, i, f"+{v:.1f}%", va="center", fontsize=10, weight="bold")

    from matplotlib.patches import Patch

    ax2.legend(
        handles=[
            Patch(facecolor="#58b980", label="Modifiable"),
            Patch(facecolor="#9aa7ab", label="Non-modifiable (genetic)"),
        ],
        loc="lower right",
        framealpha=0.9,
    )

    fig.suptitle(
        f"Risk Factor Landscape at Current 50% PD Baseline\nEngland, Adults Aged 65+, Year {baseline_year}",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#e9e9e9")
    plt.close(fig)


def _plot_figure_3(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot(index="year", columns="scenario", values="annual_total_societal_cost_gbp_bn").sort_index()
    years = pivot.index.to_numpy()
    low = pivot["25% PD"].to_numpy()
    base = pivot["50% PD"].to_numpy()
    high = pivot["75% PD"].to_numpy()

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    fig.patch.set_facecolor("#e9e9e9")
    _style_axes(ax)

    ax.plot(years, low, color="#2f8ab3", marker="o", markersize=3.8, linewidth=2.6, label="25% PD")
    ax.plot(years, base, color="#e4831f", marker="s", markersize=3.8, linewidth=3.0, label="50% PD (Baseline)")
    ax.plot(years, high, color="#cf4020", marker="^", markersize=4.1, linewidth=2.6, label="75% PD")
    ax.fill_between(years, low, base, color="#6ea46f", alpha=0.35)
    ax.fill_between(years, base, high, color="#ed7f77", alpha=0.35)

    ax.set_xlim(float(np.min(years)) - 0.5, float(np.max(years)) + 0.5)
    y_min = min(float(np.min(low)), float(np.min(base)), float(np.min(high)))
    y_max = max(float(np.max(low)), float(np.max(base)), float(np.max(high)))
    ax.set_ylim(y_min * 0.98, y_max * 1.02)
    ax.set_xlabel("Year", fontsize=12, weight="bold")
    ax.set_ylabel("Annual Total Societal Costs (GBP billions)", fontsize=13, weight="bold")
    ax.set_title(
        "Annual Dementia Costs by Periodontal Disease Prevalence Scenario\nEngland, Adults Aged 65+, 2024-2040",
        fontsize=16,
        weight="bold",
        pad=14,
    )
    ax.legend(loc="upper left", framealpha=0.95, fancybox=True, shadow=True, fontsize=11)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#e9e9e9")
    plt.close(fig)


def _plot_figure_4(df: pd.DataFrame, out_path: Path) -> None:
    p25 = df[(df["qaly_type"] == "patient") & (df["scenario"] == "25% PD")]
    p75 = df[(df["qaly_type"] == "patient") & (df["scenario"] == "75% PD")]
    c25 = df[(df["qaly_type"] == "caregiver") & (df["scenario"] == "25% PD")]
    c75 = df[(df["qaly_type"] == "caregiver") & (df["scenario"] == "75% PD")]

    years = p25["year"].to_numpy()
    patient_25 = p25["cumulative_difference_millions"].to_numpy()
    patient_75 = p75["cumulative_difference_millions"].to_numpy()
    caregiver_25 = c25["cumulative_difference_millions"].to_numpy()
    caregiver_75 = c75["cumulative_difference_millions"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.6, 7.9), sharex=True)
    fig.patch.set_facecolor("#e9e9e9")
    _style_axes(ax1)
    _style_axes(ax2)

    ax1.plot(years, patient_25, color="#2c8db7", marker="o", markersize=3, linewidth=2.5, label="25% PD vs Baseline")
    ax1.plot(years, patient_75, color="#d44a2b", marker="^", markersize=3, linewidth=2.5, label="75% PD vs Baseline")
    ax1.axhline(0, color="#666", linestyle="--", alpha=0.8, label="50% Baseline")
    ax1.fill_between(years, 0, patient_25, color="#7d7fda", alpha=0.35)
    ax1.fill_between(years, 0, patient_75, color="#e88d8b", alpha=0.38)
    ax1.set_ylabel("Cumulative Patient QALY\nDifference from Baseline (millions)", weight="bold")
    ax1.set_title("A) Patient QALYs: Minimal Variation from Baseline", fontsize=14, weight="bold", pad=8)
    ax1.legend(loc="upper left", framealpha=0.95, fancybox=True, shadow=True)

    ax2.plot(years, caregiver_25, color="#2c8db7", marker="o", markersize=3, linewidth=2.5, label="25% PD vs Baseline")
    ax2.plot(years, caregiver_75, color="#d44a2b", marker="^", markersize=3, linewidth=2.5, label="75% PD vs Baseline")
    ax2.axhline(0, color="#666", linestyle="--", alpha=0.8, label="50% Baseline")
    ax2.fill_between(years, 0, caregiver_25, color="#7d7fda", alpha=0.35)
    ax2.fill_between(years, 0, caregiver_75, color="#e88d8b", alpha=0.38)
    ax2.set_ylabel("Cumulative Caregiver QALY\nDifference from Baseline (millions)", weight="bold")
    ax2.set_xlabel("Year", fontsize=12, weight="bold")
    ax2.set_title("B) Caregiver QALYs: Inverse Relationship with PD Prevalence", fontsize=14, weight="bold", pad=8)
    ax2.legend(loc="lower left", framealpha=0.95, fancybox=True, shadow=True)

    min_y = min(float(np.min(patient_25)), float(np.min(patient_75)), float(np.min(caregiver_25)), float(np.min(caregiver_75)))
    max_y = max(float(np.max(patient_25)), float(np.max(patient_75)), float(np.max(caregiver_25)), float(np.max(caregiver_75)))
    ax1.set_ylim(min(-0.30, min_y * 1.1), max(0.30, max_y * 1.1))
    ax2.set_ylim(min(-0.38, min_y * 1.1), max(0.38, max_y * 1.1))

    fig.suptitle(
        "Cumulative QALY Differences from 50% Baseline Over Time\nEngland, Adults Aged 65+, 2024-2040",
        fontsize=16,
        weight="bold",
        y=0.99,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#e9e9e9")
    plt.close(fig)


def _export_data_excel(
    out_path: Path,
    figure2_df: pd.DataFrame,
    figure3_df: pd.DataFrame,
    figure4_df: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(out_path) as writer:
        figure2_df.to_excel(writer, sheet_name="figure_2_risk_landscape", index=False)
        figure3_df.to_excel(writer, sheet_name="figure_3_annual_costs", index=False)
        figure4_df.to_excel(writer, sheet_name="figure_4_qaly_differences", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 25/50/75 PD scenarios and generate manuscript Figure 2, 3, and 4."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("AD_Model_v3/images_regenerated_model"),
        help="Directory for output PNGs/data (default: AD_Model_v3/images_regenerated_model)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for all scenarios (default: 42)",
    )
    parser.add_argument(
        "--population-fraction",
        type=float,
        default=1.0,
        help="Fraction of configured population to simulate, in (0,1] (default: 1.0)",
    )
    parser.add_argument(
        "--baseline-year",
        type=int,
        default=2040,
        help="Year used for Figure 2 risk landscape (default: 2040)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2024,
        help="Start year for Figure 3 and Figure 4 (default: 2024)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2040,
        help="End year for Figure 3 and Figure 4 (default: 2040)",
    )
    parser.add_argument(
        "--export-excel",
        action="store_true",
        help="Also export figure source data to outdir/figure_2_3_4_data.xlsx",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.facecolor": "#e9e9e9",
        }
    )

    scenarios = {
        "25% PD": 0.25,
        "50% PD": 0.50,
        "75% PD": 0.75,
    }

    scenario_dfs: Dict[str, pd.DataFrame] = {}
    for scenario_name, prevalence in scenarios.items():
        print(f"Running scenario: {scenario_name} (PD prevalence={prevalence:.2f})")
        scenario_dfs[scenario_name] = _run_scenario(
            pd_prevalence=prevalence,
            seed=args.seed,
            population_fraction=args.population_fraction,
        )

    figure2_df = _extract_figure_2_data(
        baseline_df=scenario_dfs["50% PD"],
        baseline_year=args.baseline_year,
    )
    figure3_df = _extract_figure_3_data(
        scenario_dfs=scenario_dfs,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    figure4_df, _ = _extract_figure_4_data(
        scenario_dfs=scenario_dfs,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    fig2_path = args.outdir / "figure_2.png"
    fig3_path = args.outdir / "figure_3.png"
    fig4_path = args.outdir / "figure_4.png"

    _plot_figure_2(figure2_df, fig2_path, baseline_year=args.baseline_year)
    _plot_figure_3(figure3_df, fig3_path)
    _plot_figure_4(figure4_df, fig4_path)

    if args.export_excel:
        data_path = args.outdir / "figure_2_3_4_data.xlsx"
        _export_data_excel(
            out_path=data_path,
            figure2_df=figure2_df,
            figure3_df=figure3_df,
            figure4_df=figure4_df,
        )
        print(f"Data workbook written: {data_path}")

    print("Generated:")
    print(f"  {fig2_path}")
    print(f"  {fig3_path}")
    print(f"  {fig4_path}")


if __name__ == "__main__":
    main()

