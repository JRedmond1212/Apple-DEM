# field_growth_streamlit.py
# -------------------------------------------------------------------
# Streamlit UI for open-field fruit growth simulations (PySimpleGUI -> Streamlit)
# - Session-state persistence for loaded data and results
# - Sidebar sliders/dropdowns for inputs
# - "Load weather data" loads/caches the dataframe
# - "Run simulations" (outside load gate) uses session_state & persists outputs
# - Plots:
#     (1) Average yield per year with error bars (±1 std dev)
#     (2) Simulation # vs Total yield for a selected year
#     (3) Per-day growth curve for a chosen (year, simulation) + planting/harvest markers
# - CSV/Excel exports for summary and per-simulation results
# - Optional storage of per-simulation daily series (bounded to save memory)
# -------------------------------------------------------------------

import datetime
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")  # ensure non-interactive backend for Streamlit
import matplotlib.pyplot as plt

import streamlit as st

# --------------------------- Page config ---------------------------
st.set_page_config(page_title="Open-field Growth Simulation (Streamlit)", layout="wide")
st.title("🍓 Open-field Growth Simulation — Streamlit UI")

# --------------------------- Defaults from your script ---------------------------
DEFAULT_BASE_DIR = Path(r"C:\(A) UNI\4th Year\MEng Final Year Project\Previous Years Project\Discrete-Event-Modelling-main\Discrete-Event-Modelling-main")
DEFAULT_XLSX = DEFAULT_BASE_DIR / "2013-present weather data 2.xlsx"

SOIL_TYPES = [
    "Sandy", "Loamy Sand", "Sandy Loam", "Silty Loam",
    "Silty Clay Loam", "Silty Clay", "Clay"
]
SOIL_DRAINAGE = ["Well drained", "Moderately drained", "Poorly drained"]

OPTIMAL_TEMP_RANGE_GROWING = (15, 26)  # °C
OPTIMAL_HUMIDITY = (65, 75)            # %

# --------------------------- Session state init ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "year_bounds" not in st.session_state:
    st.session_state.year_bounds = (2013, 2023)
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "per_sim_yields" not in st.session_state:
    st.session_state.per_sim_yields = {}
if "per_sim_daily" not in st.session_state:
    # Structure: {year: {sim_index: {"days": [...], "growth": [...]}}}
    st.session_state.per_sim_daily = {}
if "planting_windows" not in st.session_state:
    # Structure: {year: (p_doy, h_doy)}
    st.session_state.planting_windows = {}

# --------------------------- Helpers ---------------------------
@st.cache_data(show_spinner=False)
def load_weather(path: Path) -> pd.DataFrame:
    """Load daily weather DataFrame and perform basic cleaning."""
    df = pd.read_excel(path, engine="openpyxl")
    if "Day" in df.columns:
        df["Day"] = df["Day"].fillna(0)

    if "g_rad (J/cm^2/day)" in df.columns:
        df["g_rad (J/cm^2/day)"] = pd.to_numeric(df["g_rad (J/cm^2/day)"], errors="coerce")
        mean_g = df["g_rad (J/cm^2/day)"].mean()
        df["g_rad (J/cm^2/day)"] = df["g_rad (J/cm^2/day)"].fillna(mean_g)

    # Construct Date/time fields if Y/M/D exist
    if set(["Year", "Month", "Day"]).issubset(df.columns):
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
        df["Day"] = pd.to_numeric(df["Day"], errors="coerce").astype("Int64")
        df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
        df["Day_of_Year"] = df["Date"].dt.dayofyear
        df["week_number"] = df["Date"].dt.isocalendar().week.astype("Int64")
        df["month"] = df["Date"].dt.month.astype("Int64")
    return df

def year_range_in(df: pd.DataFrame):
    if "Year" not in df.columns:
        return (2013, 2023)
    yrs = pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int)
    if len(yrs) == 0:
        return (2013, 2023)
    return int(yrs.min()), int(yrs.max())

def soil_params_for_type(soil_type: str):
    """Return (runoff_coefficient, density, capacity_mm) by soil type (kept from your mapping)."""
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

def soil_moisture(
    planting_depth_m,
    capacity_mm,
    soil_drainage,
    precipitation_mm,
    evapotranspiration_mm,
    density,
    smd_wd,
    smd_md,
    smd_pd,
    current_soil_moisture_mm,
    total_field_capacity_mm,
    runoff_coeff,
    drainage_rate
):
    """Minimal water balance step as in your original model."""
    if soil_drainage == "Well drained":
        smd = smd_wd
    elif soil_drainage == "Moderately drained":
        smd = smd_md
    else:
        smd = smd_pd

    total_field_capacity_mm = capacity_mm * planting_depth_m

    if current_soil_moisture_mm >= total_field_capacity_mm:
        runoff = precipitation_mm
    else:
        runoff = runoff_coeff * precipitation_mm

    drainage = drainage_rate * (current_soil_moisture_mm / max(total_field_capacity_mm, 1e-6))
    delta_sm = precipitation_mm - runoff - evapotranspiration_mm - drainage
    new_moisture = np.clip(current_soil_moisture_mm + delta_sm, 0, total_field_capacity_mm)
    return new_moisture, runoff, evapotranspiration_mm, drainage, smd

def strawberry_growth(
    G_max,
    temperature,
    PPFD,
    humidity,
    soil_moisture_mm,
    capacity_mm,
    day_of_year,
    planting_day_of_year,
    planting_duration_days
):
    """Your growth response with normalized effects and seasonal logistic curve."""
    # Thresholds
    min_temp, max_temp = 7, 30
    min_hum, max_hum = 40, 80
    min_PPFD, max_PPFD = 100, 600
    min_sm, max_sm = 0, capacity_mm

    # Normalisations
    Tn = (temperature - min_temp) / (max_temp - min_temp)
    Hn = (humidity - min_hum) / (max_hum - min_hum)
    Mn = (soil_moisture_mm - min_sm) / max((max_sm - min_sm), 1e-6)

    # Optima
    opt_T = np.mean(OPTIMAL_TEMP_RANGE_GROWING)
    opt_Tn = (opt_T - min_temp) / (max_temp - min_temp)
    opt_H = 70
    opt_Hn = (opt_H - min_hum) / (max_hum - min_hum)
    opt_M = 4.643
    opt_Mn = (opt_M - min_sm) / max((max_sm - min_sm), 1e-6)

    sigma_T = sigma_H = sigma_M = 0.3
    mu_T, mu_H, mu_M = opt_Tn, opt_Hn, opt_Mn

    # Light response (Michaelis–Menten) — simplified from your original
    m, h = 1.0, 150.0
    light_effect = m * (PPFD / (h + PPFD + 1e-6))

    if temperature < min_temp:
        return 0, 0, 0, 0, 0, 0, 0

    # Effects (scaled PDFs)
    temp_effect = norm.pdf(Tn, mu_T, sigma_T) / max(norm.pdf(mu_T, mu_T, sigma_T), 1e-9)
    hum_effect  = norm.pdf(Hn, mu_H, sigma_H) / max(norm.pdf(mu_H, mu_H, sigma_H), 1e-9)
    sm_effect   = norm.pdf(Mn, mu_M, sigma_M) / max(norm.pdf(mu_M, mu_M, sigma_M), 1e-9)

    # Logistic growth over season
    total_days = max(int(planting_duration_days), 1)
    growth_day = day_of_year - planting_day_of_year
    growth_progress = np.clip(growth_day / total_days, 0, 1)

    L, k, x0 = 1.0, 10.0, 0.5
    logistic = L / (1 + np.exp(-k * (growth_progress - x0)))
    dlogistic = k * logistic * (1 - logistic / L)

    fertilizer_effect = random.uniform(1, 1.2)
    growth = G_max * temp_effect * light_effect * hum_effect * sm_effect * dlogistic * fertilizer_effect
    return growth, temp_effect, light_effect, hum_effect, sm_effect, dlogistic, fertilizer_effect

def run_simulations(
    df, year_from, year_to, field_length, field_width, plant_spacing, row_spacing,
    soil_drainage, soil_type, planting_depth, planting_date, planting_duration,
    expected_yield_per_plant, num_simulations, weather_variability, random_seed,
    store_daily=False, max_daily_store=20
):
    """Encapsulated simulation loop; returns (results_df, per_sim_yields, per_sim_daily, planting_windows)."""
    random.seed(int(random_seed))
    np.random.seed(int(random_seed))

    runoff_coeff, density, capacity = soil_params_for_type(soil_type)
    drainage_rate = 100 if soil_drainage == "Well drained" else 10 if soil_drainage == "Moderately drained" else 0.5

    planting_dt = datetime.datetime.combine(planting_date, datetime.time.min)
    yearly_results = {}
    per_sim_yields = {}
    per_sim_daily = {}       # {year: {sim_index: {"days":[...], "growth":[...]}}}
    planting_windows = {}    # {year: (p_doy, h_doy)}

    years = list(range(int(year_from), int(year_to) + 1))
    progress = st.progress(0)
    status = st.empty()

    for i, year in enumerate(years, start=1):
        status.write(f"Simulating {year} ({i}/{len(years)})…")
        year_data = df[df["Year"].astype("Int64") == int(year)].copy()
        if year_data.empty:
            yearly_results[year] = (0.0, 0.0)
            progress.progress(i / len(years))
            continue

        p_date = planting_dt.replace(year=year)
        h_date = p_date + datetime.timedelta(days=int(planting_duration))
        p_doy = p_date.timetuple().tm_yday
        h_doy = h_date.timetuple().tm_yday
        planting_windows[year] = (p_doy, h_doy)

        sim_yields = []
        daily_store_count = 0
        for sim_index in range(1, int(num_simulations) + 1):
            current_soil_moisture = 0.5 * capacity
            daily_growth = {}

            for _, row in year_data.iterrows():
                doy = row.get("Day_of_Year", np.nan)
                if pd.isna(doy):
                    continue
                day_of_year = int(doy)
                if p_doy <= day_of_year <= h_doy:
                    wv = float(weather_variability)
                    temperature   = float(row.get("Average Temp", np.nan)) * np.random.normal(1, wv)
                    PPFD          = float(row.get("PPFD (µmol/m²/s)", np.nan)) * np.random.normal(1, wv)
                    humidity      = float(row.get("Average Humidity (%)", np.nan)) * np.random.normal(1, wv)
                    precipitation = float(row.get("rain", 0.0)) * np.random.normal(1, wv)
                    evap          = float(row.get("pe", 0.0)) * np.random.normal(1, wv)

                    smd_wd = float(row.get("smd_wd", 0.0)) * np.random.normal(1, wv)
                    smd_md = float(row.get("smd_md", 0.0)) * np.random.normal(1, wv)
                    smd_pd = float(row.get("smd_pd", 0.0)) * np.random.normal(1, wv)

                    total_field_capacity = capacity * planting_depth
                    # per-day max growth per m²
                    G_max = expected_yield_per_plant / max(int(planting_duration), 1)

                    new_m, *_ = soil_moisture(
                        planting_depth, capacity, soil_drainage, precipitation, evap,
                        density, smd_wd, smd_md, smd_pd, current_soil_moisture,
                        total_field_capacity, runoff_coeff, drainage_rate
                    )
                    growth, *_ = strawberry_growth(
                        G_max, temperature, PPFD, humidity, new_m, capacity,
                        day_of_year, p_doy, planting_duration
                    )
                    current_soil_moisture = new_m
                    daily_growth[day_of_year] = daily_growth.get(day_of_year, 0.0) + float(growth)

            area = field_length * field_width
            total_yield = float(np.sum(list(daily_growth.values())) * area)
            sim_yields.append(total_yield)

            # Optionally store daily series (bounded)
            if store_daily and daily_store_count < int(max_daily_store):
                if year not in per_sim_daily:
                    per_sim_daily[year] = {}
                days = sorted(daily_growth.keys())
                growth_series = [daily_growth[d] for d in days]
                per_sim_daily[year][sim_index] = {"days": days, "growth": growth_series}
                daily_store_count += 1

        if sim_yields:
            yearly_results[year] = (float(np.mean(sim_yields)), float(np.std(sim_yields)))
            per_sim_yields[year] = sim_yields

        progress.progress(i / len(years))

    years_sorted = sorted(yearly_results.keys())
    results_df = pd.DataFrame({
        "Year": years_sorted,
        "Average_Yield_kg": [yearly_results[y][0] for y in years_sorted],
        "StdDev_Yield_kg":  [yearly_results[y][1] for y in years_sorted],
    })
    results_df["Coeff_of_Variation_%"] = np.where(
        results_df["Average_Yield_kg"] > 0,
        results_df["StdDev_Yield_kg"] / results_df["Average_Yield_kg"] * 100,
        np.nan
    )
    return results_df, per_sim_yields, per_sim_daily, planting_windows

# --------------------------- Sidebar Inputs ---------------------------
with st.sidebar:
    st.header("Inputs")

    # Weather file path (text box to point to your Excel)
    xlsx_path_str = st.text_input(
        "Excel weather file path",
        value=str(DEFAULT_XLSX),
        help="Absolute path to '2013-present weather data 2.xlsx'."
    )
    xlsx_path = Path(xlsx_path_str)

    # Geometry & planting
    field_length = st.slider("Field length (m)", 1.0, 1000.0, 100.0, 1.0)
    field_width  = st.slider("Field width (m)", 1.0, 1000.0, 50.0, 1.0)
    plant_spacing = st.slider("Distance between plants in rows (m)", 0.1, 10.0, 1.0, 0.1)
    row_spacing   = st.slider("Distance between rows (m)", 0.1, 10.0, 2.0, 0.1)

    soil_drainage = st.selectbox("Soil drainage", SOIL_DRAINAGE, index=1)
    soil_type     = st.selectbox("Soil type", SOIL_TYPES, index=2)
    planting_depth = st.slider("Planting depth (m)", 0.05, 1.0, 0.3, 0.01)

    planting_date = st.date_input("Planting date", datetime.date(2013, 4, 1))
    planting_duration = st.slider("Planting duration (days)", 30, 240, 120, 1)

    expected_yield_per_plant = st.slider("Expected yield per plant (kg)", 0.01, 5.0, 0.5, 0.01)

    # Sim & noise
    num_simulations = st.slider("Number of simulations per year", 1, 2000, 200, 1)
    weather_variability = st.slider("Weather variability (σ as fraction)", 0.0, 0.25, 0.05, 0.01)
    random_seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.subheader("Daily series storage (optional)")
    store_daily = st.checkbox("Store per-simulation daily series", value=True,
                              help="Keeps day-by-day growth for a subset of simulations so you can plot them later.")
    max_daily_store = st.slider("Max per-year series to store", 1, 100, 20, 1,
                                help="Limits memory usage. Only the first N simulations per year are stored.")

    # Explicit load button
    load_btn = st.button("Load weather data")

# --------------------------- Handle Load Button ---------------------------
if load_btn:
    if not xlsx_path.exists():
        st.error(f"File not found: {xlsx_path}")
    else:
        df = load_weather(xlsx_path)
        st.session_state.df = df
        st.session_state.loaded = True
        st.session_state.year_bounds = year_range_in(df)
        st.success(f"Weather data loaded. Rows: {len(df):,}")

# --------------------------- Input Summary Panel ---------------------------
with st.expander("Input summary", expanded=False):
    field_area = field_length * field_width
    num_rows = field_width / row_spacing
    num_plants_per_row = field_length / plant_spacing
    total_plants = num_rows * num_plants_per_row
    planting_density_m2 = total_plants / field_area
    planting_density_per_ha = planting_density_m2 * 10_000
    expected_yield_m2 = expected_yield_per_plant * planting_density_m2

    st.write(f"**Field area:** {field_area:,.2f} m²")
    st.write(f"**Number of rows:** {num_rows:,.1f}")
    st.write(f"**Plants per row:** {num_plants_per_row:,.1f}")
    st.write(f"**Total plants:** {total_plants:,.0f}")
    st.write(f"**Planting density:** {planting_density_m2:,.3f} plants/m²  ({planting_density_per_ha:,.0f} plants/ha)")
    st.write(f"**Theoretical yield:** {expected_yield_m2:,.3f} kg/m²")

# --------------------------- Year range & Run button (persist) ---------------------------
yr_min, yr_max = st.session_state.year_bounds
if st.session_state.loaded:
    year_from, year_to = st.slider(
        "Year range to simulate",
        min_value=yr_min, max_value=yr_max,
        value=(max(yr_min, 2013), min(yr_max, 2023))
    )
else:
    year_from, year_to = st.slider(
        "Year range to simulate",
        min_value=2013, max_value=2023, value=(2013, 2023),
        disabled=True
    )

run_btn = st.button("▶ Run simulations")

# --------------------------- Run simulations (using state) ---------------------------
if run_btn:
    if not st.session_state.loaded or st.session_state.df is None:
        st.warning("Load weather data first.")
    else:
        with st.spinner("Simulating…"):
            results_df, per_sim_yields, per_sim_daily, planting_windows = run_simulations(
                df=st.session_state.df,
                year_from=year_from,
                year_to=year_to,
                field_length=field_length,
                field_width=field_width,
                plant_spacing=plant_spacing,
                row_spacing=row_spacing,
                soil_drainage=soil_drainage,
                soil_type=soil_type,
                planting_depth=planting_depth,
                planting_date=planting_date,
                planting_duration=planting_duration,
                expected_yield_per_plant=expected_yield_per_plant,
                num_simulations=num_simulations,
                weather_variability=weather_variability,
                random_seed=int(random_seed),
                store_daily=store_daily,
                max_daily_store=max_daily_store
            )
        st.session_state.results_df = results_df
        st.session_state.per_sim_yields = per_sim_yields
        # Merge (append) per-day series & planting windows to persist across runs
        # (If you prefer overwrite semantics, assign instead of update)
        st.session_state.per_sim_daily = per_sim_daily
        st.session_state.planting_windows = planting_windows
        st.success("Simulation complete.")

# --------------------------- Results (persist across re-runs) ---------------------------
if st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    st.subheader("Results summary")
    st.dataframe(results_df, use_container_width=True)

    # Plot: Average yield per year with error bars
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.errorbar(
        results_df["Year"], results_df["Average_Yield_kg"],
        yerr=results_df["StdDev_Yield_kg"], marker="o", linestyle="-", capsize=4
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total yield (kg)")
    ax1.set_title("Average total yield per year (±1 std dev)")
    ax1.grid(axis="y", alpha=0.3)
    st.pyplot(fig1)

    # Plot: Simulation # vs Total yield for a selected year
    st.markdown("### Simulation # vs Total yield")
    year_choice = st.selectbox(
        "Select a year to view per-simulation yields",
        options=list(results_df["Year"]),
        index=0
    )
    if year_choice in st.session_state.per_sim_yields:
        yvals = st.session_state.per_sim_yields[year_choice]
        sim_ids = list(range(1, len(yvals) + 1))
        sim_df = pd.DataFrame({"Simulation": sim_ids, "Total_Yield_kg": yvals})

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(sim_df["Simulation"], sim_df["Total_Yield_kg"], marker="o", linewidth=1)
        ax2.set_xlabel("Simulation #")
        ax2.set_ylabel("Total yield (kg)")
        ax2.set_title(f"Per-simulation yields — {year_choice}")
        ax2.grid(axis="y", alpha=0.3)
        st.pyplot(fig2)

        # Downloads
        st.download_button(
            "Download per-simulation yields (CSV)",
            data=sim_df.to_csv(index=False).encode("utf-8"),
            file_name=f"per_simulation_yields_{year_choice}.csv",
            mime="text/csv"
        )

    # Export summary
    st.markdown("### Export summary")
    st.download_button(
        "Download summary (CSV)",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="yearly_yield_summary.csv",
        mime="text/csv"
    )
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Yearly Yields")
    st.download_button(
        "Download summary (Excel)",
        data=buf.getvalue(),
        file_name="yearly_yield_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # --------------------------- NEW: Per-day growth plot ---------------------------
    st.markdown("### Daily growth (per-day series)")
    if st.session_state.per_sim_daily:
        # Restrict selectable years to those that actually have stored daily series
        years_with_series = sorted([y for y, sims in st.session_state.per_sim_daily.items() if len(sims) > 0])
        if years_with_series:
            y_for_daily = st.selectbox("Year (with stored daily series)", options=years_with_series, index=0)
            sim_options = sorted(st.session_state.per_sim_daily[y_for_daily].keys())
            sim_for_daily = st.selectbox("Simulation # (stored)", options=sim_options, index=0)

            series = st.session_state.per_sim_daily[y_for_daily][sim_for_daily]
            days = series["days"]
            growth_vals = series["growth"]

            fig3, ax3 = plt.subplots(figsize=(12, 5))
            ax3.bar(days, growth_vals)
            ax3.set_xlabel("Day of Year")
            ax3.set_ylabel("Growth (kg/m² per day)")
            ax3.set_title(f"Daily growth — Year {y_for_daily}, Simulation {sim_for_daily}")

            # Planting/Harvest markers if available
            p_h = st.session_state.planting_windows.get(y_for_daily, None)
            if p_h:
                p_doy, h_doy = p_h
                ax3.axvline(x=p_doy, linestyle="--")
                ax3.axvline(x=h_doy, linestyle="--")
            st.pyplot(fig3)

            # Optional: cumulative curve
            show_cum = st.checkbox("Show cumulative growth overlay")
            if show_cum:
                cum = np.cumsum(growth_vals)
                fig4, ax4 = plt.subplots(figsize=(12, 4))
                ax4.plot(days, cum)
                ax4.set_xlabel("Day of Year")
                ax4.set_ylabel("Cumulative growth (kg/m²)")
                ax4.set_title(f"Cumulative growth — Year {y_for_daily}, Simulation {sim_for_daily}")
                ax4.grid(axis="y", alpha=0.3)
                st.pyplot(fig4)

            # CSV download for selected daily series
            daily_df = pd.DataFrame({"Day_of_Year": days, "Growth_kg_per_m2": growth_vals})
            st.download_button(
                "Download this daily series (CSV)",
                data=daily_df.to_csv(index=False).encode("utf-8"),
                file_name=f"daily_growth_year{y_for_daily}_sim{sim_for_daily}.csv",
                mime="text/csv"
            )
        else:
            st.info("No stored daily series yet. Enable 'Store per-simulation daily series' and re-run simulations.")
    else:
        st.info("Daily series storage is disabled or no series were stored. Tick the checkbox in the sidebar and re-run.")

# --------------------------- Notes ---------------------------
with st.expander("Notes", expanded=False):
    st.markdown(
        "- Buttons trigger full script re-runs; **session_state** keeps data/results visible.\n"
        "- The **Daily series** feature stores only the first *N* simulations per year (set in the sidebar) to manage memory.\n"
        "- If your Excel uses different column names, update the mappings where weather variables are read."
    )

