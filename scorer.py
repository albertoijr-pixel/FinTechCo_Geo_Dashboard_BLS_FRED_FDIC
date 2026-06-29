import os
import pandas as pd
from dotenv import load_dotenv

from data_fetcher import fetch_fdic_deposits, fetch_bls_state_wages

load_dotenv()

US_STATES = {
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO",
    "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO",
    "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA",
    "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA",
    "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA",
    "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA", "OREGON",
    "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA",
    "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON",
    "WEST VIRGINIA", "WISCONSIN", "WYOMING", "DISTRICT OF COLUMBIA",
}

SECTOR_LABELS = {
    "Tech (NAICS 5415)": "Tech",
    "Physicians (NAICS 6211)": "Healthcare",
    "Legal (NAICS 5411)": "Legal",
}


def _normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(50.0, index=series.index)
    return (series - lo) / (hi - lo) * 100.0


def compute_scores() -> pd.DataFrame:
    bls_key = os.getenv("BLS_API_KEY", "")

    fdic_df = fetch_fdic_deposits()
    bls_df  = fetch_bls_state_wages(bls_key)

    # --- filter to 50 states + DC ---
    fdic_df = fdic_df[fdic_df["state"].str.strip().str.upper().isin(US_STATES)].copy().reset_index(drop=True)

    # --- FDIC-derived features ---
    fdic_df["nonint_per_inst"] = fdic_df.apply(
        lambda r: r["nonint_deposits_millions"] / r["institution_count"]
        if r["institution_count"] > 0 else 0.0,
        axis=1,
    )
    fdic_df["nonint_to_total_ratio"] = fdic_df.apply(
        lambda r: r["nonint_deposits_millions"] / r["total_deposits_millions"]
        if r["total_deposits_millions"] > 0 else 0.0,
        axis=1,
    )

    # --- BLS: pivot to one row per state, one column per sector ---
    if not bls_df.empty:
        bls_df["sector_label"] = bls_df["sector_name"].map(SECTOR_LABELS)
        wage_pivot = (
            bls_df.pivot_table(
                index="state_name",
                columns="sector_label",
                values="mean_annual_wage",
                aggfunc="first",
            )
            .reset_index()
            .rename(columns={"state_name": "state"})
        )
        wage_cols = [c for c in ["Tech", "Healthcare", "Legal"] if c in wage_pivot.columns]

        # max wage and dominant sector per state
        wage_pivot["max_annual_wage"] = wage_pivot[wage_cols].max(axis=1)
        wage_pivot["dominant_sector"] = wage_pivot[wage_cols].idxmax(axis=1)
        # idxmax returns NaN when all sector wages are NaN
        wage_pivot["dominant_sector"] = wage_pivot["dominant_sector"].fillna("Unknown")

        df = fdic_df.merge(
            wage_pivot[["state", "max_annual_wage", "dominant_sector"] + wage_cols],
            on="state",
            how="left",
        )
    else:
        df = fdic_df.copy()
        df["max_annual_wage"] = float("nan")
        df["dominant_sector"] = "Unknown"

    df["max_annual_wage"]  = df["max_annual_wage"].fillna(df["max_annual_wage"].median())
    df["dominant_sector"]  = df["dominant_sector"].fillna("Unknown")

    # --- normalize components ---
    df["norm_total_deposits"]   = _normalize(df["total_deposits_millions"])
    df["norm_nonint_per_inst"]  = _normalize(df["nonint_per_inst"])
    df["norm_max_wage"]         = _normalize(df["max_annual_wage"])
    df["norm_nonint_ratio"]     = _normalize(df["nonint_to_total_ratio"])

    # --- weighted target score ---
    df["target_score"] = (
        0.35 * df["norm_total_deposits"] +
        0.25 * df["norm_nonint_per_inst"] +
        0.25 * df["norm_max_wage"] +
        0.15 * df["norm_nonint_ratio"]
    ).round(2)

    return df.sort_values("target_score", ascending=False).reset_index(drop=True)


def main():
    df = compute_scores()

    display_cols = [
        "state", "target_score", "dominant_sector",
        "total_deposits_millions", "nonint_per_inst",
        "max_annual_wage", "nonint_to_total_ratio", "institution_count",
    ]

    print("=== Top 15 States by Target Score ===")
    print(df[display_cols].head(15).to_string(index=False))

    df.to_csv("target_scores.csv", index=False)
    print(f"\nFull results saved to target_scores.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
