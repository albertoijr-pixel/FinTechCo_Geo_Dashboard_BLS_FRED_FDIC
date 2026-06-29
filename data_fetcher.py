import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BLS_SERIES = {
    "5415": "Tech (NAICS 5415)",
    "6211": "Physicians (NAICS 6211)",
    "5411": "Legal (NAICS 5411)",
}

# OEWS MSA-level series IDs follow the pattern:
# OEUM<area_code><industry_code>000000<data_type>
# We use "area_code" 0000000 (national) as a fallback;
# for real MSA pulls the caller would supply specific area codes.
# The helper below queries the BLS public data API for a given set of series.

def _bls_post(api_key: str, series_ids: list[str]) -> dict:
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {
        "seriesid": series_ids,
        "registrationkey": api_key,
        "latest": True,
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_bls_wages(api_key: str) -> pd.DataFrame:
    """
    Pull MSA-level OEWS employment and mean annual wages for NAICS 5415, 6211, 5411.

    BLS OEWS series format:
      OEU <area> <industry> <occupation> <datatype>
      area:       7-digit MSA FIPS (0000000 = national aggregate used here)
      industry:   6-digit NAICS
      occupation: 000000 = all occupations
      datatype:   01 = employment, 04 = mean annual wage

    Returns columns: msa_code, msa_name, naics_code, sector_name,
                     employment, mean_annual_wage
    """
    rows = []
    try:
        series_ids = []
        for naics in BLS_SERIES:
            naics6 = naics.ljust(6, "0")  # BLS series requires 6-digit industry code
            series_ids.append(f"OEUN0000000{naics6}00000001")  # employment
            series_ids.append(f"OEUN0000000{naics6}00000004")  # mean annual wage

        data = _bls_post(api_key, series_ids)

        if data.get("status") != "REQUEST_SUCCEEDED":
            raise ValueError(f"BLS API error: {data.get('message', data)}")

        # Index results by series id for easy lookup
        by_series = {s["seriesID"]: s for s in data.get("Results", {}).get("series", [])}

        for naics, sector_name in BLS_SERIES.items():
            naics6 = naics.ljust(6, "0")
            emp_sid = f"OEUN0000000{naics6}00000001"
            wage_sid = f"OEUN0000000{naics6}00000004"

            emp_value = None
            wage_value = None

            if emp_sid in by_series:
                pts = by_series[emp_sid].get("data", [])
                if pts:
                    emp_value = float(pts[0]["value"].replace(",", ""))

            if wage_sid in by_series:
                pts = by_series[wage_sid].get("data", [])
                if pts:
                    wage_value = float(pts[0]["value"].replace(",", ""))

            rows.append({
                "msa_code": "0000000",
                "msa_name": "National",
                "naics_code": naics,
                "sector_name": sector_name,
                "employment": emp_value,
                "mean_annual_wage": wage_value,
            })

    except Exception as exc:
        print(f"[fetch_bls_wages] Error: {exc}")
        return pd.DataFrame(columns=[
            "msa_code", "msa_name", "naics_code", "sector_name",
            "employment", "mean_annual_wage",
        ])

    return pd.DataFrame(rows, columns=[
        "msa_code", "msa_name", "naics_code", "sector_name",
        "employment", "mean_annual_wage",
    ])


def fetch_fdic_deposits() -> pd.DataFrame:
    """
    Pull total deposits and non-interest-bearing deposits by state from the FDIC API.

    Returns columns: state, total_deposits_millions, nonint_deposits_millions,
                     institution_count
    """
    try:
        url = "https://banks.data.fdic.gov/api/financials"
        params = {
            "filters": "REPDTE:20231231",  # most recent year-end call report
            "fields": "STNAME,DEP,DEPNI,CERT",
            "agg_by": "STNAME",
            "agg_sum_fields": "DEP,DEPNI",
            "agg_count_fields": "CERT",
            "output": "json",
            "limit": 100,
            "offset": 0,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        records = payload.get("data", [])
        if not records:
            raise ValueError("FDIC API returned no data")

        rows = []
        for rec in records:
            d = rec.get("data", rec)
            rows.append({
                "state": d.get("STNAME"),
                # FDIC reports deposits in thousands; convert to millions
                "total_deposits_millions": round(float(d.get("sum_DEP", 0)) / 1_000, 2),
                "nonint_deposits_millions": round(float(d.get("sum_DEPNI", 0)) / 1_000, 2),
                "institution_count": int(d.get("count", 0)),
            })

        return pd.DataFrame(rows, columns=[
            "state", "total_deposits_millions", "nonint_deposits_millions",
            "institution_count",
        ])

    except Exception as exc:
        print(f"[fetch_fdic_deposits] Error: {exc}")
        return pd.DataFrame(columns=[
            "state", "total_deposits_millions", "nonint_deposits_millions",
            "institution_count",
        ])


def fetch_fred_deposits(api_key: str) -> pd.DataFrame:
    """
    Pull FRED series DPSACBW027SBOG (deposits at all commercial banks, weekly, SA).

    Returns columns: date, deposits_billions
    """
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "DPSACBW027SBOG",
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if "error_message" in payload:
            raise ValueError(f"FRED API error: {payload['error_message']}")

        observations = payload.get("observations", [])
        if not observations:
            raise ValueError("FRED API returned no observations")

        rows = []
        for obs in observations:
            value = obs.get("value", ".")
            if value == ".":
                continue
            rows.append({
                "date": obs["date"],
                "deposits_billions": float(value),
            })

        df = pd.DataFrame(rows, columns=["date", "deposits_billions"])
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    except Exception as exc:
        print(f"[fetch_fred_deposits] Error: {exc}")
        return pd.DataFrame(columns=["date", "deposits_billions"])


_STATE_FIPS = {
    "01": "ALABAMA", "02": "ALASKA", "04": "ARIZONA", "05": "ARKANSAS",
    "06": "CALIFORNIA", "08": "COLORADO", "09": "CONNECTICUT", "10": "DELAWARE",
    "11": "DISTRICT OF COLUMBIA", "12": "FLORIDA", "13": "GEORGIA", "15": "HAWAII",
    "16": "IDAHO", "17": "ILLINOIS", "18": "INDIANA", "19": "IOWA",
    "20": "KANSAS", "21": "KENTUCKY", "22": "LOUISIANA", "23": "MAINE",
    "24": "MARYLAND", "25": "MASSACHUSETTS", "26": "MICHIGAN", "27": "MINNESOTA",
    "28": "MISSISSIPPI", "29": "MISSOURI", "30": "MONTANA", "31": "NEBRASKA",
    "32": "NEVADA", "33": "NEW HAMPSHIRE", "34": "NEW JERSEY", "35": "NEW MEXICO",
    "36": "NEW YORK", "37": "NORTH CAROLINA", "38": "NORTH DAKOTA", "39": "OHIO",
    "40": "OKLAHOMA", "41": "OREGON", "42": "PENNSYLVANIA", "44": "RHODE ISLAND",
    "45": "SOUTH CAROLINA", "46": "SOUTH DAKOTA", "47": "TENNESSEE", "48": "TEXAS",
    "49": "UTAH", "50": "VERMONT", "51": "VIRGINIA", "53": "WASHINGTON",
    "54": "WEST VIRGINIA", "55": "WISCONSIN", "56": "WYOMING",
}

# OEWS does not publish state × NAICS cross-tabs via the v2 series API.
# We approximate sector wages using SOC occupation major-group codes instead:
#   15-0000  Computer and Mathematical  → Tech (NAICS 5415) proxy
#   29-0000  Healthcare Practitioners   → Physicians (NAICS 6211) proxy
#   23-0000  Legal                      → Legal (NAICS 5411) proxy
#
# Series format:
#   OEUS + fips(2) + 00000(5-pad → 7-char area) + 000000(all-NAICS) + occ6(6) + 04(wage) = 25 chars
_STATE_SECTORS = {
    "5415": ("150000", "Tech (NAICS 5415)"),
    "6211": ("290000", "Physicians (NAICS 6211)"),
    "5411": ("230000", "Legal (NAICS 5411)"),
}


def fetch_bls_state_wages(api_key: str) -> pd.DataFrame:
    """
    Pull state-level OEWS mean annual wages as proxies for NAICS 5415, 6211, 5411.

    Uses SOC occupation major groups (15-0000/29-0000/23-0000) since OEWS does not
    publish state-level data broken out by NAICS industry via the v2 series API.
    51 states × 3 sectors = 153 series; batched into groups of 50.

    Returns columns: state_fips, state_name, naics_code, sector_name,
                     mean_annual_wage
    """
    try:
        all_series = [
            (f"OEUS{fips}00000000000{occ6}04", fips, naics)
            for fips in _STATE_FIPS
            for naics, (occ6, _) in _STATE_SECTORS.items()
        ]

        by_series: dict = {}
        batch_size = 50
        for i in range(0, len(all_series), batch_size):
            batch_ids = [s[0] for s in all_series[i : i + batch_size]]
            data = _bls_post(api_key, batch_ids)
            if data.get("status") != "REQUEST_SUCCEEDED":
                raise ValueError(f"BLS API error: {data.get('message', data)}")
            for s in data.get("Results", {}).get("series", []):
                by_series[s["seriesID"]] = s

        rows = []
        for series_id, fips, naics in all_series:
            occ6, sector_name = _STATE_SECTORS[naics]
            wage = None
            if series_id in by_series:
                pts = by_series[series_id].get("data", [])
                if pts and pts[0].get("value", ".") != ".":
                    try:
                        wage = float(pts[0]["value"].replace(",", ""))
                    except ValueError:
                        pass
            rows.append({
                "state_fips": fips,
                "state_name": _STATE_FIPS[fips],
                "naics_code": naics,
                "sector_name": sector_name,
                "mean_annual_wage": wage,
            })

        return pd.DataFrame(rows, columns=[
            "state_fips", "state_name", "naics_code", "sector_name",
            "mean_annual_wage",
        ])

    except Exception as exc:
        print(f"[fetch_bls_state_wages] Error: {exc}")
        return pd.DataFrame(columns=[
            "state_fips", "state_name", "naics_code", "sector_name",
            "mean_annual_wage",
        ])


def main():
    bls_key = os.getenv("BLS_API_KEY", "")
    fred_key = os.getenv("FRED_API_KEY", "")

    print("=== BLS Wages (OEWS) ===")
    bls_df = fetch_bls_wages(bls_key)
    print(bls_df.head())

    print("\n=== FDIC Deposits by State ===")
    fdic_df = fetch_fdic_deposits()
    print(fdic_df.head())

    print("\n=== FRED Commercial Bank Deposits ===")
    fred_df = fetch_fred_deposits(fred_key)
    print(fred_df.head())


if __name__ == "__main__":
    main()
