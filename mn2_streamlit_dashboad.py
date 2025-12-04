#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import os
from pathlib import Path
import re
from PIL import Image
import datetime
import h5py
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.interpolate import interp1d
import pandas as pd
import pickle

# external helper module (do not modify)
import mn2_dashboard_functions as mn2

st.set_page_config(layout="wide")

PROJECT_DIR = Path(__file__).resolve().parent
local_reference_psd_path = PROJECT_DIR / "reference_psds"
local_figure_path = PROJECT_DIR / "figures"
local_figure_path.mkdir(parents=True, exist_ok=True)
local_vibdetection_path = PROJECT_DIR / "output"
local_vibdetection_path.mkdir(parents=True, exist_ok=True)

# sensor/mirror mapping
i2m_dict = {
    1: "M3a", 2: "M3b", 3: "M2", 4: "empty",
    5: "M1+y", 6: "M1-x", 7: "M1-y", 8: "M1+x",
    9: "M4", 10: "M5", 11: "M6", 12: "M7",
}
sensor_idx_lab = [i for i, m in i2m_dict.items() if m != "empty"]

# upgrade reference date
MN2_UPGRADE_DATE = datetime.datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")

# peak detection controls
vib_det_window = 50
vib_det_thresh_factor = 3


# ---------------- sidebar controls ----------------

st.sidebar.markdown("**Input files (one per UT)**")

def _default_ut_path(ut: int) -> str:
    return str(PROJECT_DIR / "test_data" / f"ldlvib{ut}_raw_2023-01-13.hdf5")

file_ut = {}
for ut in (1, 2, 3, 4):
    file_ut[ut] = st.sidebar.text_input(
        f"UT{ut} HDF5 path",
        value=_default_ut_path(ut),
        key=f"ut{ut}_path",
    )

st.sidebar.markdown("---")

# pre/post-upgrade flag (applies to all UTs)
post_upgrade = st.sidebar.checkbox(
    f"Data recorded after MNII upgrade (≥ {MN2_UPGRADE_DATE.date()})",
    value=True,
)

# telescope state (applies to all UTs)
st.sidebar.markdown("**Telescope state (applies to all UTs)**")

focus_choice = st.sidebar.selectbox(
    "Focus",
    options=["Coude", "Nasmyth A", "Nasmyth B", "Cassegrain"],
    index=3,
)
open_enclosure = st.sidebar.checkbox("Enclosure open", value=False)
guiding = st.sidebar.checkbox("Guiding", value=False)

def build_tel_state_dict(focus_name: str, open_enclosure: bool, guiding: bool):
    focus_flags = {"Coude": False, "Nasmyth A": False, "Nasmyth B": False, "Cassegrain": False}
    if focus_name in focus_flags:
        focus_flags[focus_name] = True
    base_state = {"open_enclosure": open_enclosure, "guiding": guiding}
    base_state.update(focus_flags)
    return {ut: base_state.copy() for ut in (1, 2, 3, 4)}

tel_state_dict = build_tel_state_dict(focus_choice, open_enclosure, guiding)

st.sidebar.markdown("---")
st.sidebar.markdown("**Plot options**")
which_ts_plot = st.sidebar.selectbox("Time Series Plot Options", ("unfiltered", "filtered"), index=1)
usr_display_ref = st.sidebar.checkbox("display 10-90 percentile of the reference PSD", value=True)
usr_display_vib = st.sidebar.checkbox("display detected vibration peaks", value=False)


# ---------------- helpers ----------------

_TIME_KEY_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")

def list_time_keys(h5f) -> list[str]:
    try:
        keys = [k for k in h5f.keys() if _TIME_KEY_RE.match(k)]
        return sorted(keys)
    except Exception:
        return []

def file_mtime_str(path: Path) -> str:
    try:
        ts = datetime.datetime.fromtimestamp(path.stat().st_mtime)
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"

def color_status(val):
    return "color: red" if val == "NOK" else "color: green"

@st.cache_data(show_spinner=False)
def compute_psd_from_acc(acc_dict):
    current_psd_dict = {}
    for ut in acc_dict.keys():
        current_psd_dict[ut] = {}
        for s in acc_dict[ut].keys():
            f_acc, acc_psd = sig.welch(acc_dict[ut][s], fs=1e3, nperseg=2**11, axis=0)
            f_pos, pos_psd = mn2.double_integrate(f_acc, acc_psd)
            current_psd_dict[ut][s] = {"pos": (f_pos, pos_psd), "acc": (f_acc, acc_psd)}
    return current_psd_dict

@st.cache_data(show_spinner=False)
def fetch_reference_psds(psd_dict, tel_state_dict, ref_root: Path):
    ref_psd_dict = {}
    for ut in psd_dict.keys():
        ref_psd_dict[ut] = {}
        for s in psd_dict[ut].keys():
            ref_psd_dict[ut][s] = pd.DataFrame([])
            try:
                # mn2.get_psd_reference_file expects a string path
                ref_path_str = str(ref_root) + "/"
                path_list, no_ref = mn2.get_psd_reference_file(ref_path_str, ut, s, tel_state_dict)
                if not no_ref and path_list:
                    df = pd.read_csv(path_list[0], index_col=0)
                    if df is not None and not df.empty:
                        ref_psd_dict[ut][s] = df
                        continue
                # try again with focus forced to "all"
                tmp = {k: v.copy() for k, v in tel_state_dict.items()}
                tmp[ut]["Coude"] = False
                tmp[ut]["Nasmyth A"] = False
                tmp[ut]["Nasmyth B"] = False
                tmp[ut]["Cassegrain"] = False
                path_list, no_ref = mn2.get_psd_reference_file(ref_path_str, ut, s, tmp)
                if not no_ref and path_list:
                    df = pd.read_csv(path_list[0], index_col=0)
                    if df is not None and not df.empty:
                        ref_psd_dict[ut][s] = df
            except Exception:
                pass
    return ref_psd_dict

def classify_psds(psd_dict, ref_psd_dict):
    report_card = {}
    for ut in psd_dict.keys():
        report_card[ut] = {}
        for s in psd_dict[ut].keys():
            f_tmp, psd_tmp = psd_dict[ut][s]["pos"]
            ref_df = ref_psd_dict.get(ut, {}).get(s, pd.DataFrame([]))
            if ref_df is not None and not ref_df.empty:
                f_ref = np.array(list(ref_df.index))
                interp_fn = interp1d(f_tmp, psd_tmp, kind="linear", bounds_error=False, fill_value=np.nan)
                current_psd_tmp = interp_fn(f_ref)
                log_dist = np.sqrt(np.nanmean((np.log10(current_psd_tmp) - np.log10(1e-12 * ref_df["q50"])) ** 2))
                vib_detection_df = mn2.vibration_analysis(
                    (f_ref, current_psd_tmp),
                    detection_method="median",
                    window=vib_det_window,
                    thresh_factor=vib_det_thresh_factor,
                    plot_psd=False,
                    plot_peaks=False,
                )
                if "sensor" in s:
                    strong = np.array(vib_detection_df["rel_contr"]) > (1 / mn2.V2acc_gain * 150e-9) ** 2
                else:
                    strong = np.array(vib_detection_df["rel_contr"]) > (150e-9) ** 2
                report_card[ut][s] = {
                    "psd-data": (f_ref, current_psd_tmp),
                    "psd-log_dist": float(log_dist),
                    "psd-vib_detection_df": vib_detection_df,
                    "psd-strong_vibs_indx": strong,
                    "psd-strong_vibs": vib_detection_df["vib_freqs"][strong],
                    "psd-reference_psd_available": 1,
                }
            else:
                # fall back to native PSD grid if no reference; still store psd-data
                report_card[ut][s] = {
                    "psd-data": psd_dict[ut][s]["pos"],
                    "psd-reference_psd_available": 0,
                }
    return report_card

def build_status_tables(report_card):
    status_index = ["m1-3", "m4-7", "m1-7"] + [f"m{i}" for i in range(1, 8)] + [f"sensor_{i}" for i in sensor_idx_lab]
    status_df = pd.DataFrame({ut: ["OK"] * len(status_index) for ut in (1, 2, 3, 4)}, index=status_index)
    reasons = {ut: {row: [] for row in status_index} for ut in (1, 2, 3, 4)}
    for ut in (1, 2, 3, 4):
        if ut not in report_card:
            status_df[ut] = "NA"
            for s in status_index:
                reasons[ut][s].append(["no UT data"])
            continue
        for s in status_index:
            if s not in report_card[ut]:
                status_df.at[s, ut] = "NA"
                reasons[ut][s].append([f"no {s}"])
                continue
            r = report_card[ut][s]
            if r.get("psd-reference_psd_available", 0) == 0:
                continue
            if np.sum(r["psd-strong_vibs_indx"]) > 0:
                status_df.at[s, ut] = "NOK"
                reasons[ut][s].append([f"strong vibrations at {list(r['psd-strong_vibs'])} Hz"])
            if r.get("psd-log_dist", 0) > 3:
                status_df.at[s, ut] = "NOK"
                reasons[ut][s].append(["PSD log-distance > 3 dex"])
    return status_df, reasons


# ---------------- load files, select time keys, process ----------------

latest_files = {}
latest_data = {}
time_keys = {}
current_time_key = {}
file_mtime = {}

for ut in (1, 2, 3, 4):
    p = Path(file_ut[ut]).expanduser()
    latest_files[ut] = str(p)
    if not p.exists():
        latest_data[ut] = None
        time_keys[ut] = []
        current_time_key[ut] = None
        file_mtime[ut] = "missing"
        continue
    file_mtime[ut] = file_mtime_str(p)
    try:
        latest_data[ut] = h5py.File(p, "r")
    except Exception:
        latest_data[ut] = None
        time_keys[ut] = []
        current_time_key[ut] = None
        continue
    time_keys[ut] = list_time_keys(latest_data[ut])
    if not time_keys[ut]:
        current_time_key[ut] = None
    else:
        # selection widget per UT; remember selection by key
        default_idx = len(time_keys[ut]) - 1
        idx = st.sidebar.selectbox(
            f"UT{ut} sample time (HH:MM:SS)",
            options=list(range(len(time_keys[ut]))),
            format_func=lambda i: time_keys[ut][i],
            index=default_idx,
            key=f"ut{ut}_time_idx",
        )
        current_time_key[ut] = time_keys[ut][idx]

# build accelerometer dict per UT using mn2 helper
acc_dict = {}
for ut in (1, 2, 3, 4):
    if latest_data[ut] is None or current_time_key[ut] is None:
        continue
    acc_dict[ut] = mn2.process_single_mn2_sample(
        latest_data[ut],
        current_time_key[ut],
        post_upgrade=post_upgrade,
        user_defined_geometry=None,
        outlier_thresh=9,
        replace_value=0,
        ensure_1ms_sampling=False,
    )

# psds + references + classification
current_psd_dict = compute_psd_from_acc(acc_dict) if acc_dict else {}
current_ref_psds = fetch_reference_psds(current_psd_dict, tel_state_dict, local_reference_psd_path) if current_psd_dict else {}
report_card = classify_psds(current_psd_dict, current_ref_psds) if current_psd_dict else {}
mn2_status_df, status_justification_df = build_status_tables(report_card)


# ---------------- UI ----------------

st.header("PARANAL VIBRATION DASHBOARD")

tabs = st.tabs(["analyse data", "vibration source inventory"])

with tabs[0]:
    # show file timestamp provenance
    stamp_txt = []
    for ut in (1, 2, 3, 4):
        stamp_txt.append(f"UT{ut}: file mtime = {file_mtime[ut]}")
    st.caption("Titles below use file modification time (local filesystem): " + " | ".join(stamp_txt))

    # three even columns
    col1, col2, col3 = st.columns([1, 1, 1])

    # status table, state table, diagram
    col1.dataframe(mn2_status_df.style.applymap(color_status), height=420, use_container_width=True)

    tel_state_df = pd.DataFrame(tel_state_dict)
    tel_state_df.columns = ["UT1", "UT2", "UT3", "UT4"]
    col2.dataframe(tel_state_df, use_container_width=True, height=420)

    try:
        diag = Image.open(str(PROJECT_DIR / "figures" / "MNII_system.jpeg"))
        col3.image(diag, caption="The Manhattan network of accelerometers", use_column_width=True)
    except Exception:
        col3.write("MNII_system.jpeg not found")

    display_status_justification = st.checkbox("display_status_justification", value=False)
    if display_status_justification:
        st.dataframe(pd.DataFrame(status_justification_df), use_container_width=True)

    st.download_button(
        "Download Report Card",
        data=pickle.dumps(report_card),
        file_name=f"mn2_report_card_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
    )

    # -------- time series --------
    st.markdown('<p style="font-family:sans-serif; color:Green; font-size: 42px;">Most Recent Time Series</p>', unsafe_allow_html=True)

    fig_ts, ax_ts = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    for ut, ax in zip((1, 2, 3, 4), ax_ts.reshape(-1)):
        if latest_data[ut] is None or current_time_key[ut] is None:
            ax.set_title(f"UT{ut}\n(no data)")
            ax.axis("off")
            continue
        filen = Path(latest_files[ut]).name
        sat_check = {}
        if post_upgrade:
            sens = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
        else:
            sens = [1, 2, 3, 5, 6, 7, 8]
        for i, s in enumerate(sens):
            if which_ts_plot == "unfiltered":
                acc_data = np.array(latest_data[ut][current_time_key[ut]][f"sensor{s}"][:])
            else:
                acc_data = np.array(acc_dict[ut][f"sensor_{s}"])
            sat_check[s] = int(np.nansum(np.abs(acc_data) > 10))
            ax.plot(10 * i + acc_data, "k", alpha=0.9)
            ax.text(0, 10 * i + 2, i2m_dict[s], fontsize=12, color="red")
        ax.set_title(f"UT{ut}\n{filen} @ {current_time_key[ut]}UT  (file mtime {file_mtime[ut]})")
        ax.set_yticks([])
        ax.tick_params(labelsize=12)
        if any(v > 0 for v in sat_check.values()):
            ax.set_facecolor("pink")
            ax.text(100, 0, "amplifier saturation suspected", style="italic",
                    bbox={"facecolor": "white", "alpha": 0.9, "pad": 10})

    st.pyplot(fig_ts)
    if st.button("download timeseries figure"):
        fig_ts.savefig(local_figure_path / f"streamlit_ts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{which_ts_plot}.png", dpi=200)

    # -------- PSDs --------
    st.markdown('<p style="font-family:sans-serif; color:Green; font-size: 42px;">Most Recent Power Spectral Densities (PSDs)</p>', unsafe_allow_html=True)

    # build option list: combined geometries first, then sensors. default m1-7.
    combined_opts = ["m1-7", "m4-7", "m1-3"]
    sensor_opts = [f"sensor_{i}" for i in sensor_idx_lab]
    psd_plot_options = combined_opts + sensor_opts
    usr_which_psd_plot = st.selectbox("PSDs Plot Options", options=tuple(psd_plot_options), index=0, key="psd_opt")

    fig_psd, ax_psd = plt.subplots(2, 2, sharex=False, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for ut, ax in zip((1, 2, 3, 4), ax_psd.reshape(-1)):
        if ut not in report_card or usr_which_psd_plot not in report_card[ut]:
            ax.set_title(f"UT{ut}\n(no PSD for selection)")
            ax.axis("off")
            continue

        filen = Path(latest_files[ut]).name
        f, psd = report_card[ut][usr_which_psd_plot]["psd-data"]

        ax.loglog(f, 1e12 * psd, color="k", linestyle="-", label=usr_which_psd_plot)
        # reverse cumulative (approx) for visual context
        if len(f) > 1:
            df = np.diff(f)
            df = np.hstack([df[:1], df])  # pad to match length
            ax.loglog(f, 1e12 * np.cumsum(psd[::-1])[::-1] * df, color="grey", linestyle=":", alpha=0.7)

        # reference envelope
        if usr_display_ref:
            ref_df = current_ref_psds.get(ut, {}).get(usr_which_psd_plot, pd.DataFrame([]))
            if ref_df is not None and not ref_df.empty:
                f_ref = np.array(list(ref_df.index))
                q10 = ref_df["q10"].values
                q90 = ref_df["q90"].values
                ax.fill_between(f_ref, q10, q90, color="green", alpha=0.4, label="ref q10–q90")

        # vibration peaks
        if usr_display_vib and "psd-vib_detection_df" in report_card[ut][usr_which_psd_plot]:
            vib_df = report_card[ut][usr_which_psd_plot]["psd-vib_detection_df"]
            # ensure we index on the same frequency grid as plotted
            _, idx, _ = np.intersect1d(f, vib_df["vib_freqs"], return_indices=True)
            if len(idx) > 0:
                ax.loglog(f[idx], 1e12 * psd[idx], "x", color="r", label="vib peaks")
                med_floor = pd.Series(psd).rolling(vib_det_window, center=True).median()
                detect_level = vib_det_thresh_factor * med_floor
                ax.semilogy(f, 1e12 * med_floor, linestyle=":", color="k")
                ax.semilogy(f, 1e12 * detect_level, linestyle=":", color="orange", label="detection thresh.")
                ax.vlines(x=f[idx], ymin=1e12 * med_floor.values[idx], ymax=1e12 * psd[idx], color="k")

        ax.grid()
        ax.set_ylabel("OPL PSD [$\\mu m^2/Hz$]\nreverse cumulative [$\\mu m^2$]", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.set_ylim(1e-10, 1e4)
        ax.set_xlabel("frequency [Hz]", fontsize=12)
        ax.set_title(f"UT{ut}\n{filen} @ {current_time_key[ut]}UT")

        ax.legend(fontsize=11)

    st.pyplot(fig_psd)

    if st.button("download PSD figure"):
        fig_psd.savefig(
            local_figure_path / f"streamlit_psd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_vib-{usr_display_vib}_ref-{usr_display_ref}_{usr_which_psd_plot}.png",
            dpi=200,
        )

    # detected vibration tables (only for a concrete selection, not combined/all)
    if usr_which_psd_plot in psd_plot_options:
        show_tables = st.checkbox("show detected vibration tables", value=True)
        if show_tables:
            for ut in (1, 2, 3, 4):
                st.markdown(f"UT{ut}")
                if ut not in report_card or "psd-vib_detection_df" not in report_card[ut].get(usr_which_psd_plot, {}):
                    st.write("no detections")
                    continue
                df = pd.DataFrame(report_card[ut][usr_which_psd_plot]["psd-vib_detection_df"])[
                    ["vib_freqs", "fwhm", "rel_contr", "abs_contr"]
                ].copy()
                df[["rel_contr", "abs_contr"]] = (1e18 * df[["rel_contr", "abs_contr"]]) ** 0.5
                df.columns = ["frequency [Hz]", "FWHM [Hz]", "RMS cont-peak [nm]", "RMS absolute [nm]"]
                st.dataframe(df, use_container_width=True)
                if st.button(f"download vibration table UT{ut}", key=f"dl_vib_ut{ut}"):
                    df.to_csv(local_vibdetection_path / f"vibTable_UT{ut}_{usr_which_psd_plot}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

with tabs[1]:
    st.write("TO BE UPDATED:")
    for ut in (1, 2, 3, 4):
        st.markdown(f"UT{ut}")
        ex = pd.DataFrame(
            {
                "frequency (Hz)": [47, 200, 73],
                "origin": ["fans", "MACAO", ""],
                "related PR": ["-", "-", "https://wits.pl.eso.org/browse/PR-173294"],
            }
        )
        st.dataframe(ex, use_container_width=True)

# small css to make tabs headings large
st.markdown(
    """
<style>
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu May 11 12:11:26 2023

# @author: bcourtne

# For ESO/online tool version: 
# run in virtual environment 
# source venv/bin/activate
# see https://gitlab.eso.org/datalab/dlab_python_libraries/dlt/-/wikis/Jupyter-Playground-on-your-Laptop 


# """
# import streamlit as st
# #import mpld3
# import streamlit.components.v1 as components
# import numpy as np
# import os
# import subprocess
# import pandas as pd
# from pathlib import Path
# import glob
# from PIL import Image
# import datetime
# import h5py
# import matplotlib.pyplot as plt
# import scipy.signal as sig
# from scipy.interpolate import interp1d
# import pickle
# import plotly.express as px
# import re

# # for datalab functionality 
# try:
#     import dlt  # this will only work if we are in datalab virtual environment
# except:
#     print("cannot import dlt, probably not in correct virtual environment so use this as an offline tool!")
# # see https://gitlab.eso.org/datalab/dlab_python_libraries/dlt/-/wikis/Jupyter-Playground-on-your-Laptop 

# # then our own functions 
# import mn2_dashboard_functions as mn2

# # Folder where this script lives
# PROJECT_DIR = Path(__file__).resolve().parent

# # Static default telescope state (no longer used directly, kept for reference)
# STATIC_TEL_STATE = {
#     1: {
#         "open_enclosure": False,
#         "guiding": False,
#         "Coude": False,
#         "Nasmyth A": False,
#         "Nasmyth B": False,
#         "Cassegrain": True,
#     },
#     2: {
#         "open_enclosure": False,
#         "guiding": False,
#         "Coude": False,
#         "Nasmyth A": False,
#         "Nasmyth B": False,
#         "Cassegrain": True,
#     },
#     3: {
#         "open_enclosure": False,
#         "guiding": False,
#         "Coude": False,
#         "Nasmyth A": False,
#         "Nasmyth B": False,
#         "Cassegrain": True,
#     },
#     4: {
#         "open_enclosure": False,
#         "guiding": False,
#         "Coude": False,
#         "Nasmyth A": False,
#         "Nasmyth B": False,
#         "Cassegrain": True,
#     },
# }

# st.set_page_config(layout="wide")


# st.cache_resource
# def prepare_current_status(file_paths_tuple, time_key, post_upgrade):
#     """
#     file_paths_tuple: tuple of 4 strings, one per UT (in order UT1..UT4)
#     time_key: string 'HH:MM:SS' to select in each file
#     post_upgrade: bool
#     """
#     progress_text = "Collecting and classifying the selected data. This can take a little while. Please wait."
#     percent_complete = 0
#     my_bar = st.progress(percent_complete, text=progress_text)

#     latest_data = {}
#     latest_files = {}
#     acc_dict = {}

#     # unpack tuple into {1: path1, 2: path2, ...}
#     file_paths = {ut: file_paths_tuple[ut - 1] for ut in [1, 2, 3, 4]}

#     for ut in [1, 2, 3, 4]:
#         percent_complete += 25
#         my_bar.progress(percent_complete, text=progress_text)

#         latest_files[ut] = file_paths[ut]
#         latest_data[ut] = h5py.File(latest_files[ut], "r")

#         # assume same time keys for all UTs; use the user-selected one
#         current_time_key = time_key
#         if current_time_key not in latest_data[ut].keys():
#             # fall back to last available key if user-selected one is missing
#             all_keys = sorted(latest_data[ut].keys())
#             current_time_key = all_keys[-1]

#         acc_dict[ut] = mn2.process_single_mn2_sample(
#             latest_data[ut],
#             current_time_key,
#             post_upgrade=post_upgrade,
#             user_defined_geometry=None,
#             outlier_thresh=9,
#             replace_value=0,
#             ensure_1ms_sampling=False,
#         )

#     # telescope states – still using static state
#     tel_state_dict = {}
#     for ut in [1, 2, 3, 4]:
#         tel_state_dict[ut] = STATIC_TEL_STATE[ut]

#     current_psd_dict = prepare_psd_data(acc_dict)
#     current_ref_psds = get_reference_psds(current_psd_dict=current_psd_dict,
#                                           tel_state_dict=tel_state_dict)

#     return (
#         latest_files,
#         latest_data,
#         acc_dict,
#         tel_state_dict,
#         current_psd_dict,
#         current_ref_psds,
#         current_time_key,
#     )

# # @st.cache_resource
# # def prepare_current_status(data_path, time_key, post_upgrade):
# #     """Read the HDF5 data file, build latest_data, latest_files and processed acc_dict, then PSDs."""
# #     progress_text = "Collecting and classfying the most recent data. This can take around 2 minutes. Please wait."
# #     percent_complete = 0
# #     my_bar = st.progress(percent_complete, text=progress_text)

# #     latest_data = {}
# #     latest_files = {}
# #     acc_dict = {}

# #     for ut in [1, 2, 3, 4]:
# #         percent_complete += 25
# #         my_bar.progress(percent_complete, text=progress_text)

# #         latest_files[ut] = str(data_path)
# #         latest_data[ut] = h5py.File(latest_files[ut], "r")

# #         acc_dict[ut] = mn2.process_single_mn2_sample(
# #             latest_data[ut],
# #             time_key,
# #             post_upgrade=post_upgrade,
# #             user_defined_geometry=None,
# #             outlier_thresh=9,
# #             replace_value=0,
# #             ensure_1ms_sampling=False,
# #         )

# #     # process PSDs from current data
# #     current_psd_dict = prepare_psd_data(acc_dict)

# #     return latest_files, latest_data, acc_dict, current_psd_dict


# @st.cache_data
# def prepare_psd_data(acc_dict):
#     current_psd_dict = {}
#     for ut in acc_dict.keys():
#         current_psd_dict[ut] = {}
#         for s in acc_dict[ut].keys():
#             current_psd_dict[ut][s] = {}
#             f_acc, acc_psd = sig.welch(acc_dict[ut][s], fs=1e3, nperseg=2**11, axis=0)
#             f_pos, pos_psd = mn2.double_integrate(f_acc, acc_psd)
#             current_psd_dict[ut][s]["pos"] = (f_pos, pos_psd)
#             current_psd_dict[ut][s]["acc"] = (f_acc, acc_psd)
#     return current_psd_dict


# @st.cache_data
# def get_reference_psds(psd_dict, tel_state_dict):
#     ref_psd_dict = {}
#     for ut in psd_dict.keys():
#         ref_psd_dict[ut] = {}
#         for s in psd_dict[ut].keys():
#             ref_psd_dict[ut][s] = {}
#             no_ref_file_flag = 0

#             print(f"\n\nBEGIN searching for reference psd file for {s} on UT{ut}")

#             tmp_reference_psd_file, no_ref_file_flag = mn2.get_psd_reference_file(
#                 str(local_reference_psd_path) + "/", ut, s, tel_state_dict
#             )

#             if not no_ref_file_flag:
#                 ref_psd_features = pd.read_csv(tmp_reference_psd_file[0], index_col=0)
#             else:
#                 ref_psd_features = pd.DataFrame([])

#             if ref_psd_features.empty:
#                 print(
#                     f"\n!!!!!!\n\nPSD reference file {ref_psd_features} is empty or non-existant in path.\n\n  "
#                     "This can occur when there was insufficient PSD data to derive statistics (e.g. a M1 recoating or instrument change for the focus etc).\n\n   "
#                     "We will look at a new reference file not filtering for focus..."
#                 )

#                 tmp_tel_state_dict = tel_state_dict.copy()
#                 tmp_tel_state_dict[ut]["Coude"] = 0
#                 tmp_tel_state_dict[ut]["Nasmyth A"] = 0
#                 tmp_tel_state_dict[ut]["Nasmyth B"] = 0
#                 tmp_tel_state_dict[ut]["Cassegrain"] = 0

#                 tmp_reference_psd_file, no_ref_file_flag = mn2.get_psd_reference_file(
#                     str(local_reference_psd_path) + "/", ut, s, tmp_tel_state_dict
#                 )
#                 if not no_ref_file_flag:
#                     ref_psd_features = pd.read_csv(
#                         tmp_reference_psd_file[0], index_col=0
#                     )
#                 else:
#                     ref_psd_features = pd.DataFrame([])

#                 if ref_psd_features.empty:
#                     no_ref_file_flag = 1
#                     print(
#                         "\nEven considering all Foci we cannot find a suitable reference psd.\n"
#                         "Consider finding & uploading a new reference file for the given UT, focus, state, and sensor\n!!!!\n"
#                     )
#                 else:
#                     print("phewwwww... we got one\n\n!!\n")

#             ref_psd_dict[ut][s] = ref_psd_features

#     return ref_psd_dict


# def classify_psds(psd_dict, ref_psd_dict, report_card):
#     for ut in psd_dict.keys():
#         if ut not in report_card:
#             report_card[ut] = {}

#         for s in psd_dict[ut].keys():
#             if s not in report_card[ut]:
#                 report_card[ut][s] = {}

#             ref_psd = ref_psd_dict[ut][s]

#             if not ref_psd.empty:
#                 if "sensor" not in s:
#                     f_tmp, psd_tmp = psd_dict[ut][s]["pos"]
#                 else:
#                     f_tmp, psd_tmp = psd_dict[ut][s]["pos"]

#                 f_ref = np.array(list(ref_psd.index))
#                 interp_fn = interp1d(
#                     f_tmp, psd_tmp, kind="linear", bounds_error=False, fill_value=np.nan
#                 )
#                 current_psd_tmp = interp_fn(f_ref)

#                 report_card[ut][s]["psd-data"] = (f_ref, current_psd_tmp)

#                 log_dist = np.sqrt(
#                     np.mean(
#                         (
#                             np.log10(current_psd_tmp)
#                             - np.log10(1e-12 * ref_psd["q50"])
#                         )
#                         ** 2
#                     )
#                 )

#                 degraded_freqs = f_ref[
#                     current_psd_tmp > 1e-12 * ref_psd["q90"].values
#                 ]
#                 improved_freqs = f_ref[
#                     current_psd_tmp < 1e-12 * ref_psd["q10"].values
#                 ]

#                 vib_detection_df = mn2.vibration_analysis(
#                     (f_ref, current_psd_tmp),
#                     detection_method="median",
#                     window=vib_det_window,
#                     thresh_factor=vib_det_thresh_factor,
#                     plot_psd=False,
#                     plot_peaks=False,
#                 )

#                 if "sensor" in s:
#                     strong_vibs_indx = (
#                         np.array(vib_detection_df["rel_contr"])
#                         > (1 / mn2.V2acc_gain * 150e-9) ** 2
#                     )
#                 else:
#                     strong_vibs_indx = (
#                         np.array(vib_detection_df["rel_contr"]) > (150e-9) ** 2
#                     )

#                 report_card[ut][s]["psd-log_dist"] = log_dist
#                 report_card[ut][s]["psd-degraded_freqs"] = degraded_freqs
#                 report_card[ut][s]["psd-improved_freqs"] = improved_freqs
#                 report_card[ut][s]["psd-vib_detection_df"] = vib_detection_df
#                 report_card[ut][s]["psd-strong_vibs_indx"] = strong_vibs_indx
#                 report_card[ut][s]["psd-strong_vibs"] = vib_detection_df["vib_freqs"][
#                     strong_vibs_indx
#                 ]
#                 report_card[ut][s]["psd-reference_psd_available"] = 1
#             else:
#                 report_card[ut][s]["psd-reference_psd_available"] = 0

#     return report_card


# def classify_ts(acc_dict):
#     print("to do")


# def get_status_from_report_card(report_card):
#     status_index = (
#         ["m1-3", "m4-7", "m1-7"]
#         + [f"m{i}" for i in range(1, 8)]
#         + [f"sensor_{i}" for i in sensor_idx_lab]
#     )
#     mn2_status_df = pd.DataFrame(
#         {ut: ["OK" for _ in range(len(status_index))] for ut in [1, 2, 3, 4]},
#         index=status_index,
#     )
#     status_justification_df = {
#         ut: {s: [] for s in status_index} for ut in [1, 2, 3, 4]
#     }

#     for ut in mn2_status_df.columns:
#         print(ut)
#         if ut in report_card:
#             for s in mn2_status_df.index:
#                 if s in report_card[ut]:
#                     if sum(report_card[ut][s]["psd-strong_vibs_indx"]) > 0:
#                         mn2_status_df.at[s, ut] = "NOK"
#                         status_justification_df[ut][s].append(
#                             [
#                                 f"strong vibrations with relative contribution > 150nm RMS for {s} on UT{ut} at frequencies: {report_card[ut][s]['psd-strong_vibs']}"
#                             ]
#                         )

#                     if report_card[ut][s]["psd-log_dist"] > 3:
#                         mn2_status_df.at[s, ut] = "NOK"
#                         status_justification_df[ut][s].append(
#                             [
#                                 "psd rms distance is more than 3 orders of magnitude > reference mean (disconnected sensor?)"
#                             ]
#                         )

#                 else:
#                     mn2_status_df.at[s, ut] = "NA"
#                     status_justification_df[ut][s].append(
#                         [f"no {s} in report_card for UT{ut}"]
#                     )
#         else:
#             mn2_status_df[ut] = "NA"
#             for s in mn2_status_df.index:
#                 status_justification_df[ut][s].append(
#                     [f"no UT{ut} in report_card"]
#                 )

#     return mn2_status_df, status_justification_df


# def build_tel_state_dict(focus_name, open_enclosure, guiding):
#     """Build telescope state dict for all UTs based on user input."""
#     focus_flags = {
#         "Coude": False,
#         "Nasmyth A": False,
#         "Nasmyth B": False,
#         "Cassegrain": False,
#     }
#     if focus_name in focus_flags:
#         focus_flags[focus_name] = True

#     base_state = {
#         "open_enclosure": open_enclosure,
#         "guiding": guiding,
#     }
#     base_state.update(focus_flags)

#     tel_state = {ut: base_state.copy() for ut in [1, 2, 3, 4]}
#     return tel_state


# @st.cache_data
# def get_time_keys_from_file(data_path):
#     """Return sorted list of time-like keys (HH:MM:SS) from the HDF5 file."""
#     with h5py.File(data_path, "r") as h5:
#         all_keys = list(h5.keys())
#     time_keys = [k for k in all_keys if re.match(r"^\d{2}:\d{2}:\d{2}$", k)]
#     time_keys = sorted(time_keys)
#     return time_keys


# # # Global variables
# now = datetime.datetime.utcnow() - datetime.timedelta(days=1)

# # if (now.hour == 0) & (now.minute < 15):
# #     now = now - datetime.timedelta(minutes=25)

# local_reference_psd_path = PROJECT_DIR / "reference_psds"

# local_figure_path = PROJECT_DIR / "figures"
# local_figure_path.mkdir(parents=True, exist_ok=True)

# local_vibdetection_path = PROJECT_DIR / "output"
# local_vibdetection_path.mkdir(parents=True, exist_ok=True)

# # mapping from sensor indices to mirror position
# i2m_dict = {
#     1: "M3a",
#     2: "M3b",
#     3: "M2",
#     4: "empty",
#     5: "M1+y",
#     6: "M1-x",
#     7: "M1-y",
#     8: "M1+x",
#     9: "M4",
#     10: "M5",
#     11: "M6",
#     12: "M7",
# }
# m2i_dict = {v: k for k, v in i2m_dict.items()}
# mirror_lab = [m for m in m2i_dict if m != "empty"]
# sensor_idx_lab = [i for i in i2m_dict if i2m_dict[i] != "empty"]

# mn2_upgrade_date = datetime.datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")

# if now > mn2_upgrade_date:
#     post_upgrade = True
#     sensor_key_list = (
#         ["m1-3", "m4-7", "m1-7"]
#         + [f"m{i}" for i in range(1, 8)]
#         + [f"sensor_{i}" for i in sensor_idx_lab]
#     )
# else:
#     post_upgrade = False
#     sensor_key_list = (
#         ["m1-3"]
#         + [f"m{i}" for i in range(1, 4)]
#         + [f"sensor_{i}" for i in sensor_idx_lab[:7]]
#     )

# vib_det_window = 50
# vib_det_thresh_factor = 3

# # -------------------- User inputs (file, focus, state, time) --------------------

# st.sidebar.header("Input data")

# # default example path inside the repo
# default_path = PROJECT_DIR / "test_data" / "ldlvib1_raw_2023-01-13.hdf5"

# data_path_str = st.sidebar.text_input(
#     "Full path to MNII daily HDF5 file",
#     value=str(default_path),
# )

# data_path = Path(data_path_str).expanduser()

# if not data_path.is_file():
#     st.sidebar.error(f"File not found: {data_path}")
#     st.stop()

# # get available time keys from the file and let the user choose
# time_keys = get_time_keys_from_file(str(data_path))
# if not time_keys:
#     st.error("No time-like keys (HH:MM:SS) found in selected file.")
#     st.stop()

# time_index = st.sidebar.slider(
#     "Time index",
#     min_value=0,
#     max_value=len(time_keys) - 1,
#     value=len(time_keys) - 1,
# )
# current_time_key = time_keys[time_index]

# focus_choice = st.sidebar.selectbox(
#     "Focus",
#     options=["Coude", "Nasmyth A", "Nasmyth B", "Cassegrain"],
#     index=3,
# )

# open_enclosure = st.sidebar.checkbox("Enclosure open", value=False)
# guiding = st.sidebar.checkbox("Guiding", value=False)

# tel_state_dict = build_tel_state_dict(focus_choice, open_enclosure, guiding)

# # uploaded_file = st.sidebar.file_uploader(
# #     "Upload MNII daily HDF5 file", type=["hdf5", "h5"]
# # )

# # if uploaded_file is None:
# #     st.sidebar.info("Please upload an HDF5 file to start.")
# #     st.stop()

# # uploaded_dir = PROJECT_DIR / "uploaded"
# # uploaded_dir.mkdir(parents=True, exist_ok=True)
# # uploaded_path = uploaded_dir / uploaded_file.name

# # with open(uploaded_path, "wb") as f:
# #     f.write(uploaded_file.getbuffer())

# # time_keys = get_time_keys_from_file(str(uploaded_path))
# # if not time_keys:
# #     st.error("No time-like keys (HH:MM:SS) found in uploaded file.")
# #     st.stop()

# # time_index = st.sidebar.slider(
# #     "Time index",
# #     min_value=0,
# #     max_value=len(time_keys) - 1,
# #     value=len(time_keys) - 1,
# # )
# # current_time_key = time_keys[time_index]

# # focus_choice = st.sidebar.selectbox(
# #     "Focus",
# #     options=["Coude", "Nasmyth A", "Nasmyth B", "Cassegrain"],
# #     index=3,
# # )

# # open_enclosure = st.sidebar.checkbox("Enclosure open", value=False)
# # guiding = st.sidebar.checkbox("Guiding", value=False)

# # tel_state_dict = build_tel_state_dict(focus_choice, open_enclosure, guiding)

# # -------------------- Process data --------------------

# latest_files, latest_data, current_acc_dict, current_psd_dict = prepare_current_status(
#     str(data_path), current_time_key, post_upgrade
# )

# current_ref_psds = get_reference_psds(current_psd_dict, tel_state_dict)

# # latest_files, latest_data, current_acc_dict, current_psd_dict = prepare_current_status(
# #     str(uploaded_path), current_time_key, post_upgrade
# # )

# # current_ref_psds = get_reference_psds(current_psd_dict, tel_state_dict)

# tel_state_df = pd.DataFrame(tel_state_dict)
# tel_state_df.columns = ["UT1", "UT2", "UT3", "UT4"]

# report_card = {}
# report_card = classify_psds(current_psd_dict, current_ref_psds, report_card)

# mn2_status_df, status_justification_df = get_status_from_report_card(report_card)

# # -------------------- Begin Dashboard --------------------

# st.header("PARANAL VIBRATION DASHBOARD")

# tabs = st.tabs(["analyse data", "vibration source inventory"])

# with tabs[0]:
#     st.header(now)

#     col1, col2, col3 = st.columns(3)

#     tel_state_df = pd.DataFrame(tel_state_dict)
#     tel_state_df.columns = ["UT1", "UT2", "UT3", "UT4"]

#     mn2_diagram = Image.open(f"{local_figure_path}/MNII_system.jpeg")

#     def color_mn2_status_df(val):
#         color = "red" if val == "NOK" else "green"
#         return "color: %s" % color

#     col1.dataframe(mn2_status_df.style.applymap(color_mn2_status_df), height=420)
#     col3.image(mn2_diagram, caption="The Manhattan network of accelerometers")

#     display_status_justification = st.checkbox("display_status_justification")
#     if display_status_justification:
#         st.dataframe(pd.DataFrame(status_justification_df), width=2000)

#     st.download_button(
#         "Download Report Card",
#         data=pickle.dumps(report_card),
#         file_name=f"mn2_report_card_{now}.pkl",
#     )

#     # -------------------- Time series plot --------------------

#     latest_ts_title = (
#         '<p style="font-family:sans-serif; color:Green; font-size: 42px;">'
#         "Most Recent Time Series</p>"
#     )
#     st.markdown(latest_ts_title, unsafe_allow_html=True)

#     which_ts_plot = st.selectbox("Time Series Plot Options", ("unfiltered", "filtered"))

#     fig1, ax1 = plt.subplots(2, 2, figsize=(10, 8))
#     plt.subplots_adjust(hspace=0.5, wspace=0.2)
#     sat_check_dict = {}

#     if post_upgrade:
#         for ut, ax in zip([1, 2, 3, 4], ax1.reshape(-1)):
#             sat_check_dict[ut] = {}
#             filen = latest_files[ut].split("/")[-1]

#             for i, acc in enumerate([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]):
#                 if which_ts_plot == "unfiltered":
#                     acc_data = np.array(
#                         latest_data[ut][current_time_key][f"sensor{acc}"][:]
#                     )
#                 else:
#                     acc_data = np.array(current_acc_dict[ut][f"sensor_{acc}"])
#                 sat_check_dict[ut][acc] = np.nansum(abs(acc_data) > 10)
#                 ax.plot(10 * i + acc_data, "k", alpha=0.9)
#                 ax.text(0, 10 * i + 2, i2m_dict[acc], fontsize=12, color="red")

#             ax.set_title(f"UT{ut}\n{filen} @ {current_time_key}UT")
#             ax.set_yticks([])
#             ax.tick_params(labelsize=12)

#             if np.any([sat_check_dict[ut][a] > 0 for a in sat_check_dict[ut]]):
#                 ax.set_facecolor("pink")
#                 ax.text(
#                     100,
#                     0,
#                     "sensor(s) {} with some voltages at amplifier\nsaturation limit (do we have ADC Spikes?)".format(
#                         1
#                         + np.where(
#                             [sat_check_dict[ut][a] > 0 for a in sat_check_dict[ut]]
#                         )[0]
#                     ),
#                     style="italic",
#                     bbox={"facecolor": "white", "alpha": 0.9, "pad": 10},
#                 )
#     else:
#         for ut, ax in zip([1, 2, 3, 4], ax1.reshape(-1)):
#             sat_check_dict[ut] = {}
#             filen = latest_files[ut].split("/")[-1]
#             for i, acc in enumerate([1, 2, 3, 5, 6, 7, 8]):
#                 acc_data = latest_data[ut][current_time_key][f"sensor{acc}"][:]
#                 sat_check_dict[ut][acc] = np.nansum(abs(acc_data) > 10)

#                 ax.plot(10 * i + acc_data, "k", alpha=0.9)
#                 ax.text(0, 10 * i + 2, i2m_dict[acc], fontsize=12, color="red")

#             ax.set_title(f"UT{ut}\n{filen} @ {current_time_key}UT")
#             ax.set_yticks([])
#             ax.tick_params(labelsize=12)
#             if np.any([sat_check_dict[ut][a] > 0 for a in sat_check_dict[ut]]):
#                 ax.set_facecolor("pink")
#                 ax.text(
#                     100,
#                     0,
#                     "sensor(s) {} with some voltages at amplifier\nsaturation limit (do we have ADC Spikes?)".format(
#                         1
#                         + np.where(
#                             [sat_check_dict[ut][a] > 0 for a in sat_check_dict[ut]]
#                         )[0]
#                     ),
#                     style="italic",
#                     bbox={"facecolor": "white", "alpha": 0.9, "pad": 10},
#                 )

#     st.pyplot(fig1)

#     if st.button("download timeseries figure"):
#         fig1.savefig(
#             os.path.join(
#                 local_figure_path,
#                 f'streamlit_ts_{now.strftime("%d-%m-%YT%H:%M:%S")}_{which_ts_plot}_fig.png',
#             )
#         )

#     # -------------------- PSD plot --------------------

#     latest_ts_title = (
#         '<p style="font-family:sans-serif; color:Green; font-size: 42px;">'
#         "Most Recent Power Spectral Densities (PSDs)</p>"
#     )
#     st.markdown(latest_ts_title, unsafe_allow_html=True)

#     psd_available_sensors = [
#         s
#         for s in list(current_psd_dict[list(current_psd_dict.keys())[0]].keys())
#         if "outliers" not in s
#     ]

#     psd_plot_usr_options = psd_available_sensors[::-1] + [
#         "all mirrors",
#         "combined geometries",
#     ]

#     usr_which_psd_plot = st.selectbox("PSDs Plot Options", options=tuple(psd_plot_usr_options))

#     usr_display_ref = st.checkbox("display 10-90 percentile of the reference PSD")

#     usr_display_vib = st.checkbox("display detected vibration peaks")

#     fig2, ax2 = plt.subplots(2, 2, sharex=False, figsize=(10, 10))
#     plt.subplots_adjust(hspace=0.5, wspace=0.5)

#     for ut, ax in zip([1, 2, 3, 4], ax2.reshape(-1)):
#         if ut in report_card:
#             filen = latest_files[ut].split("/")[-1]

#             if post_upgrade:
#                 mirror_plot_list = [f"m{i}" for i in range(1, 8)]
#                 mirror_plot_colors = ["r", "b", "g", "orange", "purple", "grey", "cyan"]

#                 comb_g_plot_list = ["m1-3", "m4-7", "m1-7"]
#                 comb_g_plot_colors = ["r", "b", "g"]
#             else:
#                 mirror_plot_list = [f"m{i}" for i in range(1, 4)]
#                 mirror_plot_colors = ["r", "b", "g"]

#                 comb_g_plot_list = ["m1-3"]
#                 comb_g_plot_colors = ["r"]

#             if usr_which_psd_plot == "all mirrors":
#                 for s, col in zip(mirror_plot_list, mirror_plot_colors):
#                     f, psd = report_card[ut][s]["psd-data"]
#                     ax.loglog(f, 1e12 * psd, color=col, linestyle="-", label=s)
#                     ax.loglog(
#                         f,
#                         1e12 * np.cumsum(psd[::-1])[::-1] * np.diff(f)[1],
#                         color=col,
#                         linestyle=":",
#                     )

#             elif usr_which_psd_plot == "combined geometries":
#                 for s, col in zip(comb_g_plot_list, comb_g_plot_colors):
#                     f, psd = report_card[ut][s]["psd-data"]
#                     ax.loglog(f, 1e12 * psd, color=col, linestyle="-", label=s)
#                     ax.loglog(
#                         f,
#                         1e12 * np.cumsum(psd[::-1])[::-1] * np.diff(f)[1],
#                         label=s,
#                         color=col,
#                         linestyle=":",
#                     )

#             else:
#                 f, psd = report_card[ut][usr_which_psd_plot]["psd-data"]

#                 ax.loglog(
#                     f,
#                     1e12 * psd,
#                     color="k",
#                     linestyle="-",
#                     label=usr_which_psd_plot,
#                 )
#                 ax.loglog(
#                     f,
#                     1e12 * np.cumsum(psd[::-1])[::-1] * np.diff(f)[1],
#                     color="grey",
#                     linestyle=":",
#                     alpha=0.7,
#                 )

#                 if usr_display_ref:
#                     f_ref_tmp = np.array(
#                         list(current_ref_psds[ut][usr_which_psd_plot].index)
#                     )
#                     psd_ref_df_tmp = current_ref_psds[ut][usr_which_psd_plot]
#                     ax.fill_between(
#                         f_ref_tmp,
#                         psd_ref_df_tmp["q10"],
#                         psd_ref_df_tmp["q90"],
#                         color="green",
#                         alpha=0.5,
#                         label="ref q10-q90",
#                     )

#                 if usr_display_vib:
#                     med_floor = pd.Series(psd).rolling(
#                         vib_det_window, center=True
#                     ).median()
#                     detection_level = vib_det_thresh_factor * med_floor

#                     vib_df = report_card[ut][usr_which_psd_plot]["psd-vib_detection_df"]

#                     _, f_indx, _ = np.intersect1d(
#                         f, vib_df["vib_freqs"], return_indices=True
#                     )
#                     ax.loglog(
#                         f[f_indx],
#                         1e12 * psd[f_indx],
#                         "x",
#                         color="r",
#                         label="vib peaks",
#                     )
#                     ax.semilogy(f, 1e12 * med_floor, linestyle=":", color="k")
#                     ax.semilogy(
#                         f,
#                         1e12 * detection_level,
#                         linestyle=":",
#                         color="orange",
#                         label="detection thresh.",
#                     )
#                     ax.vlines(
#                         x=f[f_indx],
#                         ymin=1e12 * med_floor[f_indx],
#                         ymax=1e12 * psd[f_indx],
#                         color="k",
#                     )

#             ax.grid()
#             ax.set_ylabel(
#                 r"OPL PSD [$\mu m^2/Hz$]"
#                 + "\n"
#                 + r"reverse cumulative [$\mu m^2$]",
#                 fontsize=15,
#             )
#             ax.legend(fontsize=15)
#             ax.tick_params(labelsize=15)
#             ax.set_ylim(1e-10, 1e4)
#             ax.set_xlabel(r"frequency [Hz]", fontsize=15)
#             ax.set_title(f"UT{ut}\n{filen} @ {current_time_key}UT")
#         else:
#             print(f"UT{ut} not in current_psd_dict.. cannot plot PSDs")

#     st.pyplot(fig2)

#     if st.button("download PSD figure"):
#         fig2.savefig(
#             os.path.join(
#                 local_figure_path,
#                 f'streamlit_psd_{now.strftime("%d-%m-%YT%H:%M:%S")}_vibdet-{usr_display_vib}_withRef-{usr_display_ref}_{usr_which_psd_plot}_fig.png',
#             )
#         )

#     usr_display_vib_detection_table = st.checkbox("hide detected vibration tables")

#     @st.cache_data
#     def display_detected_vibrations_df(report_card, usr_which_psd_plot):
#         detect_df_2_display = {}
#         for ut in [1, 2, 3, 4]:
#             detect_df_2_display[ut] = pd.DataFrame(
#                 report_card[ut][usr_which_psd_plot]["psd-vib_detection_df"]
#             )[["vib_freqs", "fwhm", "rel_contr", "abs_contr"]]
#             detect_df_2_display[ut][["rel_contr", "abs_contr"]] *= 1e18
#             detect_df_2_display[ut][["rel_contr", "abs_contr"]] **= 0.5
#             detect_df_2_display[ut].columns = [
#                 "frequency [Hz]",
#                 "FWHM [Hz]",
#                 "RMS contiumn-peak continuum [nm]",
#                 "RMS absolute contribution [nm]",
#             ]
#         return detect_df_2_display

#     if (not usr_display_vib_detection_table) and (
#         usr_which_psd_plot not in ["all mirrors", "combined geometries"]
#     ):
#         latest_ts_title = (
#             '<p style="font-family:sans-serif; color:Green; font-size: 42px;">'
#             f"DETECTED VIBRATIONS FOR {usr_which_psd_plot}</p>"
#         )
#         st.markdown(latest_ts_title, unsafe_allow_html=True)

#         st.markdown(f"DETECTED VIBRATIONS FOR {usr_which_psd_plot}")
#         detect_df_2_display = display_detected_vibrations_df(
#             report_card, usr_which_psd_plot
#         )

#         for ut in [1, 2, 3, 4]:
#             st.markdown(f"UT{ut}")
#             st.dataframe(detect_df_2_display[ut])

#             if st.button(f"download vibration table for UT{ut}"):
#                 detect_df_2_display[ut].to_csv(
#                     os.path.join(
#                         local_vibdetection_path,
#                         f'streamlit_vibTable_UT{ut}_{usr_which_psd_plot}_{now.strftime("%d-%m-%YT%H:%M:%S")}.csv',
#                     )
#                 )

#     if st.button("download all figures and tables"):
#         for ut in [1, 2, 3, 4]:
#             if (not usr_display_vib_detection_table) and (
#                 usr_which_psd_plot not in ["all mirrors", "combined geometries"]
#             ):
#                 detect_df_2_display[ut].to_csv(
#                     os.path.join(
#                         local_vibdetection_path,
#                         f'streamlit_vibTable_UT{ut}_{usr_which_psd_plot}_{now.strftime("%d-%m-%YT%H:%M:%S")}.csv',
#                     )
#                 )

#         fig1.savefig(
#             os.path.join(
#                 local_figure_path,
#                 f'streamlit_ts_{now.strftime("%d-%m-%YT%H:%M:%S")}_{which_ts_plot}_fig.png',
#             )
#         )

#         fig2.savefig(
#             os.path.join(
#                 local_figure_path,
#                 f'streamlit_psd_{now.strftime("%d-%m-%YT%H:%M:%S")}_vibdet-{usr_display_vib}_withRef-{usr_display_ref}_{usr_which_psd_plot}_fig.png',
#             )
#         )

# with tabs[1]:
#     st.write("TO BE UPDATED:")
#     vib_inventory_UT1_dict = {
#         "frequency (Hz)": [47, 200, 73],
#         "origin": ["fans", "MACAO", ""],
#         "related PR": ["-", "-", "https://wits.pl.eso.org/browse/PR-173294"],
#     }
#     vib_inventory_UT2_dict = {
#         "frequency (Hz)": [47, 200, 73],
#         "origin": ["fans", "MACAO", ""],
#         "related PR": ["-", "-", "https://wits.pl.eso.org/browse/PR-173294"],
#     }

#     vib_inventory_UT3_dict = {
#         "frequency (Hz)": [47, 200, 73],
#         "origin": ["fans", "MACAO", ""],
#         "related PR": ["-", "-", "https://wits.pl.eso.org/browse/PR-173294"],
#     }

#     vib_inventory_UT4_dict = {
#         "frequency (Hz)": [47, 200, 73],
#         "origin": ["fans", "MACAO", ""],
#         "related PR": ["-", "-", "https://wits.pl.eso.org/browse/PR-173294"],
#     }

#     st.markdown("UT1")
#     st.dataframe(pd.DataFrame(vib_inventory_UT1_dict))

#     st.markdown("UT2")
#     st.dataframe(pd.DataFrame(vib_inventory_UT2_dict))

#     st.markdown("UT3")
#     st.dataframe(pd.DataFrame(vib_inventory_UT3_dict))

#     st.markdown("UT4")
#     st.dataframe(pd.DataFrame(vib_inventory_UT4_dict))

# css = """
# <style>
#     .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
#     font-size:2rem;
#     }
# </style>
# """

# st.markdown(css, unsafe_allow_html=True)

