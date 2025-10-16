# field_growth_streamlit.py
# -------------------------------------------------------------------
# Streamlit Cloud–friendly UI for open-field growth simulations
# - File uploader (no local Windows paths)
# - Deterministic RNG per (year, sim)
# - Chronological arrays (stateful soil moisture)
# - Gaussian effects peak at 1.0 (parity with original)
# - Optional Numba if available (no UI toggles needed)
# - Matplotlib optional: graceful fallback to Streamlit charts
# - Minimal UI: summary table, mean±std plot, daily bars + cumulative
# -------------------------------------------------------------------

import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

# ---- Plotting imports with graceful fallback ----
HAS_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    HAS_MPL = False

# --------------------------- Page config ---------------------------
st.set_page_config(page_title="Open-field Growth Simulation", layout="wide")
st.title("Open-field Growth Simulation — v.1.1.7")

# --------------------------- Try Numba (optional) ---------------------------
try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

# --------------------------- Constants ---------------------------
SOIL_TYPES = ["Sandy", "Loamy Sand", "Sandy Loam", "Silty Loam", "Silty Clay Loam", "Silty Clay", "Clay"]
SOIL_DRAINAGE = ["Well drained", "Moderately drained", "Poorly drained"]
OPTIMAL_TEMP_RANGE_GROWING = (15, 26)  # °C
WEATHER_SIGMA = 0.05                   # fixed stochastic variability (σ fraction)

# --------------------------- Session state init ---------------------------
ss = st.session_state
ss.setdefault("df", None)
ss.setdefault("loaded", False)
ss.setdefault("year_bounds", (2013, 2023))
ss.setdefault("results_df", None)
ss.setdefault("per_sim_yields", {})
ss.setdefault("per_sim_daily", {})      # {year: {sim: {"days":[...], "growth":[...]}}}
ss.setdefault("planting_windows", {})   # {year: (p_doy, h_doy)}
ss.setdefault("year_arrays_key", None)
ss.setdefault("year_arrays", {})

# --------------------------- Helpers ---------------------------
def _bytes_sha1(b: bytes) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()

@st.cache_data(show_spinner=False)
def load_weather_from_bytes(content: bytes) -> pd.DataFrame:
    """
    Load the Excel from uploaded bytes. Cloud-safe (no filesystem assumptions).
    """
    df = pd.read_excel(BytesIO(content), engine=None)  # let pandas pick engine
    if "Day" in df.columns:
        df["Day"] = df["Day"].fillna(0)
    if "g_rad (J/cm^2/day)" in df.columns:
        df["g_rad (J/cm^2/day)"] = pd.to_numeric(df["g_rad (J/cm^2/day)"], errors="coerce")
        df["g_rad (J/cm^2/day)"] = df["g_rad (J/cm^2/day)"].fillna(df["g_rad (J/cm^2/day)"].mean())
    if set(["Year","Month","Day"]).issubset(df.columns):
        df["Year"]  = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
        df["Day"]   = pd.to_numeric(df["Day"],   errors="coerce").astype("Int64")
        df["Date"]  = pd.to_datetime(df[["Year","Month","Day"]], errors="coerce")
        df["Day_of_Year"] = df["Date"].dt.dayofyear
        df["week_number"] = df["Date"].dt.isocalendar().week.astype("Int64")
        df["month"]       = df["Date"].dt.month.astype("Int64")
    return df

def year_range_in(df: pd.DataFrame):
    if "Year" not in df.columns:
        return (2013, 2023)
    yrs = pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int)
    return (int(yrs.min()), int(yrs.max())) if len(yrs) else (2013, 2023)

def soil_params_for_type(soil_type: str):
    mapping = {
        "Sandy":             (0.23, 1430, 57.29167),
        "Loamy Sand":        (0.27, 1430, 95.83333),
        "Sandy Loam":        (0.30, 1460, 110.4167),
        "Silty Loam":        (0.37, 1380, 187.5),
        "Silty Clay Loam":   (0.43, 1300, 158.3333),
        "Silty Clay":        (0.57, 1260, 133.3333),
        "Clay":              (0.20, 1330, 112.5),
    }
    return mapping.get(soil_type, (0.30, 1400, 120.0))

@st.cache_data(show_spinner=False)
def preprocess_year_arrays(df: pd.DataFrame):
    """
    Prepare dense float64 arrays per year for the kernel, in chronological order.
    """
    out = {}
    req = ["Year","Day_of_Year","Average Temp","PPFD (µmol/m²/s)","Average Humidity (%)","rain","pe","smd_wd","smd_md","smd_pd","Date"]
    if any(c not in df.columns for c in req):
        return out

    df2 = df.copy()
    for col in ["Average Temp","PPFD (µmol/m²/s)","Average Humidity (%)","rain","pe","smd_wd","smd_md","smd_pd","Day_of_Year"]:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")
    df2 = df2.dropna(subset=["Year","Day_of_Year","Date"])

    for y, b in df2.groupby("Year", dropna=True):
        b = b.sort_values("Date")  # chronological order matters
        days = b["Day_of_Year"].to_numpy(dtype=np.float64, copy=True)
        if days.size == 0:
            continue
        out[int(y)] = {
            "days":   days,
            "temp":   b["Average Temp"].to_numpy(dtype=np.float64, copy=True),
            "ppfd":   b["PPFD (µmol/m²/s)"].to_numpy(dtype=np.float64, copy=True),
            "hum":    b["Average Humidity (%)"].to_numpy(dtype=np.float64, copy=True),
            "rain":   b["rain"].to_numpy(dtype=np.float64, copy=True),
            "pe":     b["pe"].to_numpy(dtype=np.float64, copy=True),
            "smd_wd": b["smd_wd"].to_numpy(dtype=np.float64, copy=True),
            "smd_md": b["smd_md"].to_numpy(dtype=np.float64, copy=True),
            "smd_pd": b["smd_pd"].to_numpy(dtype=np.float64, copy=True),
        }
    return out

# --------------------------- Physics/biology pieces ---------------------------
def soil_step(current_sm, capacity, planting_depth, rain, evap, runoff_coeff, drainage_rate):
    total_fc = capacity * planting_depth
    runoff = rain if current_sm >= total_fc else runoff_coeff * rain
    drainage = drainage_rate * (current_sm / (total_fc + 1e-9))
    new_sm = current_sm + (rain - runoff - evap - drainage)
    return float(np.clip(new_sm, 0, total_fc))

def strawberry_growth_scalar(G_max, temperature, PPFD, humidity, soil_moisture_mm,
                             capacity_mm, day_of_year, p_doy, duration_days,
                             fertilizer_effect):
    # thresholds
    min_temp, max_temp = 7.0, 30.0
    min_hum,  max_hum  = 40.0, 80.0
    min_sm,   max_sm   = 0.0, capacity_mm
    if temperature < min_temp:
        return 0.0

    # light response
    light_effect = PPFD / (150.0 + PPFD + 1e-9)

    # gaussian-like effects (peak = 1.0 at optimum)
    def _scaled(x, mu, sigma=0.3):
        return np.exp(-0.5*((x - mu)/sigma)**2)

    Tn = (temperature - min_temp) / (max_temp - min_temp)
    Hn = (humidity   - min_hum)  / (max_hum  - min_hum)
    Mn = (soil_moisture_mm - min_sm) / (max_sm - min_sm + 1e-9)

    opt_T = np.mean(OPTIMAL_TEMP_RANGE_GROWING)
    mu_T = (opt_T - min_temp) / (max_temp - min_temp)
    mu_H = (70.0 - min_hum)   / (max_hum  - min_hum)
    mu_M = (4.643 - min_sm)   / (max_sm   - min_sm + 1e-9)

    temp_effect = _scaled(Tn, mu_T)
    hum_effect  = _scaled(Hn, mu_H)
    sm_effect   = _scaled(Mn, mu_M)

    # seasonal logistic derivative
    total_days = float(max(int(duration_days), 1))
    prog = np.clip((day_of_year - p_doy) / total_days, 0.0, 1.0)
    L, k, x0 = 1.0, 10.0, 0.5
    logistic = L / (1.0 + np.exp(-k*(prog - x0)))
    dlog = k * logistic * (1.0 - logistic / L)

    return G_max * temp_effect * light_effect * hum_effect * sm_effect * dlog * fertilizer_effect

# --------------------------- Optional Numba kernel ---------------------------
if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _simulate_one_year_kernel(
        days, temp, ppfd, hum, rain, pe, smd_wd, smd_md, smd_pd,
        p_doy, h_doy,
        capacity, planting_depth,
        runoff_coeff, drainage_rate,
        G_max, weather_sigma,
        planting_duration,
        seed
    ):
        np.random.seed(seed)
        total_fc = capacity * planting_depth
        current_sm = 0.5 * capacity
        total_growth_per_m2 = 0.0

        # constants
        min_temp, max_temp = 7.0, 30.0
        min_hum,  max_hum  = 40.0, 80.0
        min_sm,   max_sm   = 0.0, capacity
        L, k, x0 = 1.0, 10.0, 0.5
        sigma = 0.3
        mu_T = ((0.5*(15.0+26.0)) - 7.0) / (30.0 - 7.0)
        mu_H = (70.0 - 40.0) / (80.0 - 40.0)
        mu_M = (4.643 - 0.0) / (capacity - 0.0 + 1e-9)
        total_days = float(max(int(planting_duration), 1))

        n = days.shape[0]
        for i in range(n):
            doy = days[i]
            if doy < p_doy or doy > h_doy:
                continue

            w = weather_sigma
            t  = temp[i] * (1.0 + w * np.random.randn())
            l  = ppfd[i] * (1.0 + w * np.random.randn())
            h  = hum[i]  * (1.0 + w * np.random.randn())
            rn = rain[i] * (1.0 + w * np.random.randn())
            ev = pe[i]   * (1.0 + w * np.random.randn())

            runoff = rn if current_sm >= total_fc else runoff_coeff * rn
            drainage = drainage_rate * (current_sm / (total_fc + 1e-9))
            current_sm = current_sm + (rn - runoff - ev - drainage)
            if current_sm < 0.0: current_sm = 0.0
            if current_sm > total_fc: current_sm = total_fc

            if t < min_temp:
                continue

            Tn = (t - min_temp) / (max_temp - min_temp)
            Hn = (h - min_hum)  / (max_hum  - min_hum)
            Mn = (current_sm - min_sm) / (max_sm - min_sm + 1e-9)

            temp_effect = np.exp(-0.5*((Tn - mu_T)/sigma)**2)
            hum_effect  = np.exp(-0.5*((Hn - mu_H)/sigma)**2)
            sm_effect   = np.exp(-0.5*((Mn - mu_M)/sigma)**2)
            light_effect = l / (150.0 + l + 1e-9)

            prog = (doy - p_doy) / total_days
            if   prog < 0.0: prog = 0.0
            elif prog > 1.0: prog = 1.0
            logistic = L / (1.0 + np.exp(-k*(prog - x0)))
            dlog = k * logistic * (1.0 - logistic / L)

            fe = 1.0 + 0.2*np.random.rand()
            growth = G_max * temp_effect * light_effect * hum_effect * sm_effect * dlog * fe
            total_growth_per_m2 += growth

        return total_growth_per_m2

# --------------------------- Sidebar Inputs (Cloud-safe) ---------------------------
with st.sidebar:
    st.header("Inputs")

    uploaded = st.file_uploader("Upload weather Excel (.xlsx)", type=["xlsx"])
    if uploaded is not None:
        content = uploaded.read()
        sha = _bytes_sha1(content)
        df = load_weather_from_bytes(content)
        ss.df = df
        ss.loaded = True
        ss.year_bounds = year_range_in(df)
        ss.year_arrays = {}  # rebuild on next use
        ss.year_arrays_key = sha
        st.success(f"Weather loaded. Rows: {len(df):,}")
    else:
        st.info("Upload a weather Excel to begin.")

    field_length   = st.slider("Field length (m)", 1.0, 1000.0, 100.0, 1.0)
    field_width    = st.slider("Field width (m)",  1.0, 1000.0, 50.0, 1.0)
    plant_spacing  = st.slider("Distance between plants in rows (m)", 0.1, 10.0, 1.0, 0.1)
    row_spacing    = st.slider("Distance between rows (m)",           0.1, 10.0, 2.0, 0.1)

    soil_drainage  = st.selectbox("Soil drainage", SOIL_DRAINAGE, index=1)
    soil_type      = st.selectbox("Soil type", SOIL_TYPES, index=2)
    planting_depth = st.slider("Planting depth (m)", 0.05, 1.0, 0.3, 0.01)

    planting_date      = st.date_input("Planting date", datetime.date(2013, 4, 1))
    planting_duration  = st.slider("Planting duration (days)", 30, 240, 120, 1)
    expected_yield_per_plant = st.slider("Expected yield per plant (kg)", 0.01, 5.0, 0.5, 0.01)

    st.divider()
    random_seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.subheader("Daily series (optional)")
    store_daily = st.checkbox("Store per-simulation daily series", value=True)
    max_daily_store = st.slider("Max per-year stored series", 1, 50, 10, 1)

# --------------------------- Input summary ---------------------------
with st.expander("Input summary", expanded=False):
    field_area = field_length * field_width
    n_rows = field_width / row_spacing
    plants_per_row = field_length / plant_spacing
    total_plants = n_rows * plants_per_row
    density_m2 = total_plants / field_area
    density_ha = density_m2 * 10_000
    theoretical_yield_m2 = expected_yield_per_plant * density_m2

    st.write(f"**Field area:** {field_area:,.2f} m²")
    st.write(f"**Rows:** {n_rows:,.1f} | **Plants/row:** {plants_per_row:,.1f} | **Total plants:** {total_plants:,.0f}")
    st.write(f"**Planting density:** {density_m2:,.3f} /m²  ({density_ha:,.0f} /ha)")
    st.write(f"**Theoretical yield:** {theoretical_yield_m2:,.3f} kg/m²")

# --------------------------- Pre-cached arrays ---------------------------
def _get_year_arrays():
    if not ss.loaded or ss.df is None:
        return {}
    key = ss.year_arrays_key
    if ss.year_arrays_key != key or not ss.year_arrays:
        ss.year_arrays = preprocess_year_arrays(ss.df)
        ss.year_arrays_key = key
    return ss.year_arrays

# --------------------------- Main simulation ---------------------------
def run_simulations_fast(df, year_from, year_to,
                         field_length, field_width, soil_drainage, soil_type,
                         planting_depth, planting_date, planting_duration,
                         expected_yield_per_plant, num_simulations, random_seed,
                         store_daily, max_daily_store):
    # Seed host RNG (incidental); per-sim seeds derived below
    np.random.seed(int(random_seed))

    runoff_coeff, density, capacity = soil_params_for_type(soil_type)
    drainage_rate = 100 if soil_drainage == "Well drained" else 10 if soil_drainage == "Moderately drained" else 0.5
    planting_dt = datetime.datetime.combine(planting_date, datetime.time.min)

    year_arrays = _get_year_arrays()
    years = list(range(int(year_from), int(year_to) + 1))

    results = {}
    per_sim_yields = {}
    per_sim_daily = {}
    planting_windows = {}

    area = field_length * field_width
    nsims = int(num_simulations)
    base = int(random_seed)

    for year in years:
        arr = year_arrays.get(int(year))
        if not arr:
            results[year] = (0.0, 0.0)
            continue

        # Plant/harvest window
        p_date = planting_dt.replace(year=year)
        h_date = p_date + datetime.timedelta(days=int(planting_duration))
        p_doy = p_date.timetuple().tm_yday
        h_doy = h_date.timetuple().tm_yday
        planting_windows[year] = (p_doy, h_doy)

        G_max = expected_yield_per_plant / float(max(int(planting_duration), 1))

        sims = np.empty(nsims, dtype=np.float64)
        per_sim_daily_year = {} if (store_daily and max_daily_store > 0) else None

        for s in range(nsims):
            seed_s = np.int64(base) ^ np.int64(year * 1000003) ^ np.int64((s+1) * 97003)

            if HAS_NUMBA:
                total_per_m2 = _simulate_one_year_kernel(
                    arr["days"], arr["temp"], arr["ppfd"], arr["hum"], arr["rain"], arr["pe"],
                    arr["smd_wd"], arr["smd_md"], arr["smd_pd"],
                    float(p_doy), float(h_doy),
                    float(capacity), float(planting_depth),
                    float(runoff_coeff), float(drainage_rate),
                    float(G_max), float(WEATHER_SIGMA),
                    int(planting_duration),
                    int(seed_s)
                )
                sims[s] = total_per_m2 * area

                if per_sim_daily_year is not None and len(per_sim_daily_year) < int(max_daily_store):
                    rng = np.random.default_rng(int(seed_s) + 12345)
                    current_sm = 0.5 * capacity
                    daily = {}
                    for idx in range(arr["days"].shape[0]):
                        doy = int(arr["days"][idx])
                        if p_doy <= doy <= h_doy:
                            w = float(WEATHER_SIGMA)
                            t  = float(arr["temp"][idx]) * rng.normal(1.0, w)
                            L  = float(arr["ppfd"][idx]) * rng.normal(1.0, w)
                            h  = float(arr["hum"][idx])  * rng.normal(1.0, w)
                            rn = float(arr["rain"][idx]) * rng.normal(1.0, w)
                            ev = float(arr["pe"][idx])   * rng.normal(1.0, w)
                            current_sm = soil_step(current_sm, capacity, planting_depth, rn, ev, runoff_coeff, drainage_rate)
                            fe = rng.uniform(1.0, 1.2)
                            g = strawberry_growth_scalar(G_max, t, L, h, current_sm, capacity, doy, p_doy, planting_duration, fe)
                            daily[doy] = daily.get(doy, 0.0) + g
                    days = sorted(daily.keys())
                    per_sim_daily_year[len(per_sim_daily_year)+1] = {"days": days, "growth": [daily[d] for d in days]}

            else:
                rng = np.random.default_rng(int(seed_s))
                current_sm = 0.5 * capacity
                daily = {}
                for idx in range(arr["days"].shape[0]):
                    doy = int(arr["days"][idx])
                    if p_doy <= doy <= h_doy:
                        w = float(WEATHER_SIGMA)
                        t  = float(arr["temp"][idx]) * rng.normal(1.0, w)
                        L  = float(arr["ppfd"][idx]) * rng.normal(1.0, w)
                        h  = float(arr["hum"][idx])  * rng.normal(1.0, w)
                        rn = float(arr["rain"][idx]) * rng.normal(1.0, w)
                        ev = float(arr["pe"][idx])   * rng.normal(1.0, w)

                        current_sm = soil_step(current_sm, capacity, planting_depth, rn, ev, runoff_coeff, drainage_rate)
                        fe = rng.uniform(1.0, 1.2)
                        g = strawberry_growth_scalar(G_max, t, L, h, current_sm, capacity, doy, p_doy, planting_duration, fe)
                        daily[doy] = daily.get(doy, 0.0) + g

                sims[s] = float(np.sum(list(daily.values())) * area)

                if per_sim_daily_year is not None and len(per_sim_daily_year) < int(max_daily_store):
                    days = sorted(daily.keys())
                    per_sim_daily_year[len(per_sim_daily_year)+1] = {"days": days, "growth": [daily[d] for d in days]}

        results[year] = (float(sims.mean()), float(sims.std()))
        per_sim_yields[year] = sims.tolist()
        if per_sim_daily_year:
            per_sim_daily[year] = per_sim_daily_year

    years_sorted = sorted(results.keys())
    results_df = pd.DataFrame({
        "Year": years_sorted,
        "Average_Yield_kg": [results[y][0] for y in years_sorted],
        "StdDev_Yield_kg":  [results[y][1] for y in years_sorted],
    })
    results_df["Coeff_of_Variation_%"] = np.where(
        results_df["Average_Yield_kg"] > 0,
        results_df["StdDev_Yield_kg"]/results_df["Average_Yield_kg"]*100, np.nan
    )

    return results_df, per_sim_yields, per_sim_daily, planting_windows

# --------------------------- Run controls ---------------------------
yr_min, yr_max = ss.year_bounds
if ss.loaded:
    year_from, year_to = st.slider("Year range to simulate", min_value=yr_min, max_value=yr_max,
                                   value=(max(yr_min, 2013), min(yr_max, 2023)))
else:
    year_from, year_to = st.slider("Year range to simulate", 2013, 2023, (2013, 2023), disabled=True)

num_simulations = st.slider("Number of simulations per year", 1, 5000, 500, 1)
run_btn = st.button("▶ Run simulations")

# --------------------------- Run on click ---------------------------
if run_btn:
    if not ss.loaded or ss.df is None:
        st.warning("Upload weather data first.")
    else:
        with st.spinner("Simulating…"):
            results_df, per_sim_yields, per_sim_daily, planting_windows = run_simulations_fast(
                df=ss.df,
                year_from=year_from, year_to=year_to,
                field_length=field_length, field_width=field_width,
                soil_drainage=soil_drainage, soil_type=soil_type,
                planting_depth=planting_depth, planting_date=planting_date, planting_duration=planting_duration,
                expected_yield_per_plant=expected_yield_per_plant,
                num_simulations=num_simulations, random_seed=int(random_seed),
                store_daily=store_daily, max_daily_store=max_daily_store
            )
        ss.results_df = results_df
        ss.per_sim_yields = per_sim_yields
        ss.per_sim_daily = per_sim_daily
        ss.planting_windows = planting_windows
        st.success("Simulation complete.")

# --------------------------- Results ---------------------------
if ss.results_df is not None:
    st.subheader("Results summary")
    st.dataframe(ss.results_df, use_container_width=True)

    # Plot: mean ± std (fallback if matplotlib missing)
    if HAS_MPL:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.errorbar(ss.results_df["Year"], ss.results_df["Average_Yield_kg"],
                     yerr=ss.results_df["StdDev_Yield_kg"], marker="o", linestyle="-", capsize=4)
        ax1.set_xlabel("Year"); ax1.set_ylabel("Total yield (kg)")
        ax1.set_title("Average total yield per year (±1 std dev)")
        ax1.grid(axis="y", alpha=0.3)
        st.pyplot(fig1)
    else:
        st.info("Matplotlib not available — showing line chart without error bars.")
        st.line_chart(
            ss.results_df.set_index("Year")["Average_Yield_kg"],
            height=300
        )

    # Plot: Daily growth + Cumulative (always shown)
    st.markdown("### Daily growth (per-day) with cumulative overlay")
    if ss.per_sim_daily:
        years_with_series = sorted([y for y, sims in ss.per_sim_daily.items() if sims])
        if years_with_series:
            y_for_daily = st.selectbox("Year (with stored daily series)", options=years_with_series, index=0)
            sim_options = sorted(ss.per_sim_daily[y_for_daily].keys())
            sim_for_daily = st.selectbox("Simulation #", options=sim_options, index=0)

            series = ss.per_sim_daily[y_for_daily][sim_for_daily]
            days = series["days"]; growth_vals = series["growth"]

            if HAS_MPL:
                # Daily bars
                fig3, ax3 = plt.subplots(figsize=(12, 5))
                ax3.bar(days, growth_vals)
                ax3.set_xlabel("Day of Year"); ax3.set_ylabel("Growth (kg/m² per day)")
                ax3.set_title(f"Daily growth — Year {y_for_daily}, Simulation {sim_for_daily}")
                if y_for_daily in ss.planting_windows:
                    p_doy, h_doy = ss.planting_windows[y_for_daily]
                    ax3.axvline(x=p_doy, linestyle="--")
                    ax3.axvline(x=h_doy, linestyle="--")
                st.pyplot(fig3)

                # Cumulative line (always)
                cum = np.cumsum(growth_vals)
                fig4, ax4 = plt.subplots(figsize=(12, 4))
                ax4.plot(days, cum)
                ax4.set_xlabel("Day of Year"); ax4.set_ylabel("Cumulative growth (kg/m²)")
                ax4.set_title(f"Cumulative growth — Year {y_for_daily}, Simulation {sim_for_daily}")
                ax4.grid(axis="y", alpha=0.3)
                st.pyplot(fig4)
            else:
                st.info("Matplotlib not available — using Streamlit charts.")
                # Daily bars (approximate)
                daily_df = pd.DataFrame({"Growth": growth_vals}, index=pd.Index(days, name="Day_of_Year"))
                st.bar_chart(daily_df, height=300)
                # Cumulative
                cum = np.cumsum(growth_vals)
                cum_df = pd.DataFrame({"Cumulative": cum}, index=pd.Index(days, name="Day_of_Year"))
                st.line_chart(cum_df, height=250)
        else:
            st.info("No stored daily series. Enable storage and re-run.")
    else:
        st.info("Daily series storage disabled or none stored. Enable and re-run.")
