# rainfall_and_tide_generator.py
from __future__ import annotations
from typing import Tuple, Optional
import re
import numpy as np
import pandas as pd

from scipy.signal import find_peaks

__all__ = [
    "pf_df",
    "generate_rainfall",
    "moon_tide_ranges",
    "generate_tide_curve",
    "find_tide_extrema",
    "align_rainfall_to_tide",
    "fetch_RT_tide_dataframe",
    "build_timestep_and_resample_15min",
    "get_tide_real_or_synthetic",
    "get_aligned_rainfall",
]

pf_df = pd.DataFrame({
    "Duration_Minutes": [120, 240, 360, 480, 600, 720],

    "1":  [1.68, 1.99, 2.18, 2.38, 2.44, 2.57],
    "2":  [2.02, 2.38, 2.62, 2.85, 2.93, 3.08],
    "5":  [2.49, 2.87, 3.25, 3.55, 3.65, 3.85],
    "10": [2.98, 3.41, 3.91, 4.28, 4.41, 4.66],
    "25": [3.58, 4.07, 4.77, 5.25, 5.41, 5.73],
})

SCS_TYPE_III_CUM = np.array([
    0.0000, 0.0050, 0.0110, 0.0150, 0.0200, 0.0232, 0.0308, 0.0367, 0.0430, 0.0497,
    0.0568, 0.0642, 0.0720, 0.0806, 0.0905, 0.1016, 0.1140, 0.1284, 0.1458, 0.1659,
    0.1899, 0.2165, 0.2500, 0.2980, 0.5000, 0.7020, 0.7500, 0.7835, 0.8110, 0.8341,
    0.8542, 0.8716, 0.8860, 0.8984, 0.9095, 0.9194, 0.9280, 0.9358, 0.9432, 0.9503,
    0.9570, 0.9634, 0.9694, 0.9752, 0.9808, 0.9860, 0.9900, 0.9956, 1.0000
], dtype=float)

def generate_rainfall(total_inches: float,
                      duration_minutes: int,
                      curve: np.ndarray = SCS_TYPE_III_CUM) -> np.ndarray:
    """
    SCS Type III hyetograph generator.
    Returns INCREMENTAL 15-min depths that sum to `total_inches` over `duration_minutes`.
    `duration_minutes` must be divisible by 15.
    `curve` is the dimensionless cumulative P/Ptotal vs. time-fraction (0..1), monotone 0->1.
    """
    if duration_minutes % 15 != 0:
        raise ValueError("duration_minutes must be divisible by 15.")
    if curve.ndim != 1 or len(curve) < 2:
        raise ValueError("curve must be a 1D array with at least 2 points.")
    if not (np.isclose(curve[0], 0.0) and np.isclose(curve[-1], 1.0)):
        raise ValueError("curve must start at 0.0 and end at 1.0 (dimensionless cumulative).")
    if np.any(np.diff(curve) < -1e-12):
        raise ValueError("curve must be nondecreasing.")

    intervals = duration_minutes // 15

    x_src = np.linspace(0.0, 1.0, len(curve))


    edges = np.linspace(0.0, 1.0, intervals + 1)

    cum_edges = np.interp(edges, x_src, curve)

    inc_dimless = np.diff(cum_edges)
    inc_dimless = np.clip(inc_dimless, 0.0, None)

    y = total_inches * inc_dimless

    err = total_inches - y.sum()
    if abs(err) > 1e-10:
        y[-1] += err

    return y  

moon_tide_ranges = {
    "🌓 First Quarter": (-0.4, 3.58),   
    "🌕 Full Moon":   (0, 3.9),   
    "🌗 Last Quarter":  (1.5, 3.5),   
    "🌑 New Moon":    (1.73, 3.83),   
}

def generate_tide_curve(moon_phase: str, unit: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic tide generator.
    Always returns tide in FEET for SWMM compatibility.
    Metric conversion will be handled ONLY in the Streamlit display layer.
    """
    if moon_phase not in moon_tide_ranges:
        raise ValueError(f"Unknown moon phase: {moon_phase}")


    low_ft, high_ft = moon_tide_ranges[moon_phase]

    minutes_15 = np.arange(0, 48 * 60, 15)

    mid = (low_ft + high_ft) / 2
    amp = (high_ft - low_ft) / 2
    tide_15 = mid + amp * np.sin(2 * np.pi * minutes_15 / (12.42 * 60))  

    return minutes_15, tide_15

def find_tide_extrema(
    tide_curve_15min: np.ndarray,
    distance_bins: int = 40,      
    prominence: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return indices of tide highs (peaks) and lows (troughs) on a 15-min series.
    Tune 'distance_bins' and 'prominence' to control detection strictness.
    """
    peaks, _   = find_peaks(tide_curve_15min, distance=distance_bins, prominence=prominence)
    troughs, _ = find_peaks(-tide_curve_15min, distance=distance_bins, prominence=prominence)
    return peaks, troughs

def align_rainfall_to_tide(total_inches: float,
                           duration_minutes: int,
                           tide_curve_15min: np.ndarray,
                           align: str = "peak",
                           method: str = "SCS_TypeIII",
                           target_index: Optional[int] = None,
                           prominence: Optional[float] = None  
                           ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build an SCS Type III hyetograph (incremental, 15-min bins) and align it to a tide series.
    - total_inches: storm depth [in]
    - duration_minutes: storm duration (multiple of 15)
    - tide_curve_15min: array of tide elevations at 15-min steps (length N)
    - align: "peak" or "low" (align to nearest peak/low near the series mid), or use target_index
    - method: only "SCS_TypeIII" is supported (kept to avoid breaking callers)
    - target_index: if set, center the storm at this index (overrides 'align')
    Returns (minutes_15, rain_15, center_index_used)
    """
    if duration_minutes % 15 != 0:
        raise ValueError("duration_minutes must be divisible by 15.")
    n = int(len(tide_curve_15min))
    if n == 0:
        raise ValueError("Empty tide curve.")
    intervals = duration_minutes // 15
    if intervals > n:
        raise ValueError(f"Storm of {intervals} bins does not fit into tide series of length {n}.")

    rain_profile = generate_rainfall(total_inches, duration_minutes)


    if target_index is not None:
        center_index = int(target_index)
        if not (0 <= center_index < n):
            raise ValueError("target_index out of range.")
    else:
        try:
            from rainfall_and_tide_generator import find_tide_extrema  # noqa
            peaks, troughs = find_tide_extrema(tide_curve_15min, prominence=prominence)
            candidates = peaks if align == "peak" else troughs
            candidates = np.asarray(candidates, dtype=int)
        except Exception:
            candidates = generate_rainfall(tide_curve_15min, "peak" if align == "peak" else "trough")

        if candidates.size == 0:
            center_index = n // 2
        else:
            mid = n // 2
            center_index = int(candidates[np.argmin(np.abs(candidates - mid))])

    start = center_index - (intervals // 2)
    start = max(0, min(start, n - intervals))  
    end = start + intervals

    rain = np.zeros(n, dtype=float)
    rain[start:end] = rain_profile  

    minutes_15 = np.arange(n, dtype=int) * 15
    return minutes_15, rain, center_index

GREENSTREAM_URL = "https://dashboard.greenstream.cloud/detail?id=SITE#d935fec2-7a0b-4df0-986c-76f25d773070"
WATER_COL_LIVE = "Water Level NAVD88 (ft)"  


X_FILTER   = 1565   
X_DOWNLOAD = 1470   
TOL        = 40
TOP_Y_MAX  = 100
ICON_MIN_W, ICON_MAX_W = 24, 56

def _find_icon_by_x(page, x_target, tol, top_y_max, min_w, max_w):
    nodes = page.locator("div, span, svg")
    n = nodes.count()
    best = None
    best_dx = float("inf")
    for i in range(n):
        h = nodes.nth(i).element_handle()
        if not h:
            continue
        box = h.bounding_box()
        if not box:
            continue
        if (box["y"] < top_y_max and min_w <= box["width"] <= max_w
                and min_w <= box["height"] <= max_w):
            dx = abs(box["x"] - x_target)
            if dx < best_dx:
                best_dx, best = dx, h
    if best is None or best_dx > tol:
        raise RuntimeError(f"Icon not found near x≈{x_target} (Δx={best_dx:.1f}). Adjust TOL/viewport.")
    return best

def _click_handle_by_center(page, handle):
    box = handle.bounding_box()
    if not box:
        raise RuntimeError("No bounding box for element.")
    try:
        handle.click(timeout=1500)
    except Exception:
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        page.mouse.click(cx, cy)

def fetch_RT_tide_dataframe() -> pd.DataFrame:
    """
    Opens the Greenstream dashboard, does:
      Filter -> Last 2 Days -> OK -> Download,
    and returns the file as a pandas DataFrame (no saving to disk).
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = browser.new_context(accept_downloads=True, viewport={"width": 1600, "height": 900})
        page = context.new_page()
        page.goto(GREENSTREAM_URL, wait_until="domcontentloaded")

        filter_handle = _find_icon_by_x(page, X_FILTER, TOL, TOP_Y_MAX, ICON_MIN_W, ICON_MAX_W)
        _click_handle_by_center(page, filter_handle)

        drawer = page.locator("div.drawer")
        drawer.wait_for(state="visible", timeout=5000)

        try:
            drawer.get_by_text("Last 2 Days", exact=True).click(timeout=2000)
        except PWTimeout:
            drawer.locator("div.radioButton").nth(1).click()  

        ok_clicked = False
        for loc in [
            drawer.get_by_role("button", name=re.compile(r"^OK$", re.I)),
            drawer.locator("button:has-text('OK')"),
            drawer.locator("[role=button]:has-text('OK')"),
            drawer.locator("[class*=Button]:has-text('OK')"),
            drawer.get_by_text(re.compile(r"^\s*OK\s*$", re.I)),
        ]:
            try:
                if loc.count():
                    el = loc.first
                    el.scroll_into_view_if_needed()
                    el.click(timeout=1500)
                    ok_clicked = True
                    break
            except Exception:
                pass
        if not ok_clicked:
            bb = drawer.bounding_box()
            if bb:
                page.mouse.click(bb["x"] + bb["width"] - 60, bb["y"] + bb["height"] - 40)

        # Settle
        try:
            drawer.wait_for(state="hidden", timeout=4000)
        except PWTimeout:
            pass
        try:
            page.wait_for_load_state("networkidle", timeout=6000)
        except PWTimeout:
            pass

        dl_el = page.locator("[title='Download']").first
        if dl_el.count():
            with page.expect_download(timeout=20000) as dl_info:
                try:
                    dl_el.click(timeout=1500)
                except Exception:
                    _click_handle_by_center(page, dl_el.element_handle())
            dl = dl_info.value
        else:
            download_handle = _find_icon_by_x(page, X_DOWNLOAD, TOL, TOP_Y_MAX, ICON_MIN_W, ICON_MAX_W)
            with page.expect_download(timeout=20000) as dl_info:
                _click_handle_by_center(page, download_handle)
            dl = dl_info.value

        tmp_path = dl.path()
        suggested = (dl.suggested_filename or "").lower()

        try:
            if suggested.endswith((".xlsx", ".xls")):
                df = pd.read_excel(tmp_path)
            else:
                df = pd.read_csv(tmp_path)
        except Exception:
            try:
                df = pd.read_csv(tmp_path)
            except Exception:
                df = pd.read_excel(tmp_path)

        browser.close()
        return df

def build_timestep_and_resample_15min(df_raw: pd.DataFrame,
                                      water_col: str = WATER_COL_LIVE,
                                      unit: str = "U.S. Customary",
                                      start_ts: Optional[pd.Timestamp] = None,
                                      navd88_to_sea_level_offset_ft: float = 0.0
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input: df_raw from Greenstream (~>=480 rows @ 6-min).
    Returns exactly 48h @ 15-min (192 bins), no padding/interpolation.
    """

    if water_col not in df_raw.columns:
        matches = [c for c in df_raw.columns if "water" in c.lower() and "level" in c.lower()]
        if not matches:
            raise ValueError(f"Water level column not found. Columns={list(df_raw.columns)}")
        water_col = matches[0]

    tide_df = df_raw[[water_col]].copy()
    n = len(tide_df)
    if n == 0:
        raise ValueError("Empty tide DataFrame (live).")

    REQUIRED_6MIN = 48 * 60 // 6  # 480
    if n < REQUIRED_6MIN:
        raise ValueError(f"Live dataset too short ({n} rows). Expected at least {REQUIRED_6MIN} rows for 48h.")
    if n > REQUIRED_6MIN:
        tide_df = tide_df.tail(REQUIRED_6MIN)
        n = REQUIRED_6MIN

    if start_ts is None:
        end6 = pd.Timestamp.now().floor("6min")
    else:
        end6 = (pd.Timestamp(start_ts).floor("6min") + pd.Timedelta(minutes=6*(n-1)))
    idx6 = pd.date_range(end=end6, periods=n, freq="6min")
    tide_df = tide_df.set_index(idx6)

    vals = tide_df[water_col].astype(float)

    offset = navd88_to_sea_level_offset_ft  
    if offset != 0.0:
        vals = vals - offset

    tide_df[water_col] = vals

    tide_15_series = tide_df.resample("15min").mean(numeric_only=True)[water_col]
    REQUIRED_15MIN = 48 * 60 // 15  # 192
    if len(tide_15_series) < REQUIRED_15MIN:
        raise ValueError(f"After resampling, got {len(tide_15_series)}×15-min bins; expected {REQUIRED_15MIN}.")
    tide_15_series = tide_15_series.tail(REQUIRED_15MIN)

    minutes_15 = np.arange(0, 48*60, 15, dtype=int) 
    tide_15 = tide_15_series.to_numpy()

    return minutes_15, tide_15

def get_tide_real_or_synthetic(moon_phase: str,
                               unit: str,
                               start_ts: Optional[pd.Timestamp] = None,
                               navd88_to_sea_level_offset_ft: float = 0
                               ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Try live tide (Greenstream). If it fails, return synthetic tide.
    Returns:
      minutes_15 (np.ndarray),
      tide_15    (np.ndarray) in selected 'unit' (ft or m), already shifted to MSL,
      used_live  (bool)
    """
    try:
        df_live = fetch_RT_tide_dataframe()
        m15, tide_15 = build_timestep_and_resample_15min(
            df_raw=df_live,
            water_col=WATER_COL_LIVE,
            unit=unit,
            start_ts=start_ts,
            navd88_to_sea_level_offset_ft=navd88_to_sea_level_offset_ft  
        )
        return m15, tide_15, True
    except Exception:
        m15, tide_15 = generate_tide_curve(moon_phase, "U.S. Customary")
        return m15, tide_15, False

def get_aligned_rainfall(
    total_inches: float,
    duration_minutes: int,
    moon_phase: str,
    unit: str,
    align: str = "peak",
    method: str = "Normal",
    start_ts: Optional[pd.Timestamp] = None,
    prominence: Optional[float] = None,
    navd88_to_sea_level_offset_ft: float = 0      
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, int]:
    """
    Returns (minutes_15, tide_15, rain_15, used_live, center_idx).
    """
    m15, tide_15, used_live = get_tide_real_or_synthetic(
        moon_phase, unit, start_ts, navd88_to_sea_level_offset_ft 
    )

    peaks, troughs = find_tide_extrema(tide_15, prominence=prominence)
    if align == "peak":
        cand = peaks
    else:
        cand = troughs

    if cand.size == 0:
        target_idx = len(tide_15) // 2
    else:
        target_idx = cand[np.argmin(np.abs(cand - len(tide_15)//2))]

    _, rain_15, center_idx = align_rainfall_to_tide(
        total_inches=total_inches,
        duration_minutes=duration_minutes,
        tide_curve_15min=tide_15,
        align=align,
        method=method,
        target_index=target_idx,
        prominence=prominence
    )
    return m15, tide_15, rain_15, used_live, center_idx
