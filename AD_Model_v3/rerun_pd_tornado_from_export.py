"""
Regenerate periodontal-disease tornado diagrams using an exported Excel file.

This avoids re-running the full model. It expects the Excel produced by
`pd_sensitivity_analysis.export_pd_sensitivity_results`, specifically the
`Raw_Results` sheet.

Example:
    python rerun_pd_tornado_from_export.py \\
        --excel pd_sensitivity_analysis.xlsx \\
        --outdir plots
"""

from pathlib import Path
import argparse
import pandas as pd

from pd_sensitivity_analysis import create_pd_tornado_diagram


def load_raw_results(excel_path: Path) -> pd.DataFrame:
    """Load the Raw_Results sheet from the export."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name="Raw_Results")
    expected_cols = {"parameter", "value_type", "replicate"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in Raw_Results; "
            "ensure the file was created by export_pd_sensitivity_results."
        )
    return df


def regenerate_diagrams(df: pd.DataFrame, outdir: Path) -> None:
    """Create the main and per-metric tornado diagrams from the DataFrame."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Main combined plot
    create_pd_tornado_diagram(
        df,
        metrics=["total_qalys_combined", "incident_onsets_total", "total_costs_all"],
        save_path=str(outdir / "pd_tornado_main.png"),
        show=False,
    )

    # Separate plots per metric
    per_metric = [
        ("total_qalys_combined", "pd_tornado_qalys.png"),
        ("incident_onsets_total", "pd_tornado_incidence.png"),
        ("total_costs_all", "pd_tornado_costs.png"),
    ]
    for metric, filename in per_metric:
        if metric not in df.columns:
            continue
        create_pd_tornado_diagram(
            df,
            metrics=[metric],
            save_path=str(outdir / filename),
            show=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-create periodontal disease tornado diagrams from an exported Excel file."
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("pd_sensitivity_analysis.xlsx"),
        help="Path to Excel file containing Raw_Results (default: pd_sensitivity_analysis.xlsx)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("plots"),
        help="Directory to save regenerated plots (default: plots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_raw_results(args.excel)
    regenerate_diagrams(df, args.outdir)
    print(f"OK Tornado diagrams regenerated from {args.excel} into {args.outdir}")


if __name__ == "__main__":
    main()
