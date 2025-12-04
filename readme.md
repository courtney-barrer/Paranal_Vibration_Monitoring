# MN2 Vibration Monitoring Dashboard

This repository contains a Streamlit-based dashboard for inspecting and classifying 10-second accelerometer samples from the MNII system at Paranal. The original dashboard relied on the ESO datalab and updated dynamically in near-real time. This version is fully **static and user-driven**: the user points the app at local HDF5 files, selects the relevant instrument state, and the dashboard produces analysis plots and classifications offline.

---

## Purpose

The dashboard provides a compact interface for analysing vibration behaviour recorded by the `wvbmet` workstation. Each sensor produces a 10-second sample (1 kHz sampling, typically one sample per minute). The tool compares a selected sample to historical **reference PSDs** and highlights departures from nominal vibration levels. It is designed for routine inspection of accelerometer health, vibration features, and telescope-level mechanical behaviour.

---

## Installation

```bash
# 1) Clone
git clone https://github.com/courtney-barrer/Paranal_Vibration_Monitoring
cd Paranal_Vibration_Monitoring

# 1b) if you want (optional) set up virtual environment to install in 
python -m venv .venv              
source .venv/bin/activate

# 2a) Recommended: install via pyproject
pip install .

# 2b) Or: install exact pinned deps
pip install -r requirements.txt
```

## Running the dashboard
# From the project root directory, run:
```bash
streamlit run mn2_streamlit_dashboad.py
```
# A local URL will appear in the terminal. Open it in your browser.

## How to use

1. Enter the full path to an MNII daily HDF5 file in the sidebar for each telescope (or the one you're interested in). These files can be several hundred megabytes and cannot be uploaded through the browser.
2. After loading, the dashboard scans the file for time keys in the format `HH:MM:SS`. A slider will appear to select the 10-second sample to inspect.
3. Select the telescope configuration:
   - Focus (Coudé, Nasmyth A, Nasmyth B, Cassegrain)
   - Enclosure state (open or closed)
   - Guiding state (on or off)

   These choices determine the state dictionary used to fetch the correct reference PSDs. Also plotting options (like showing reference vibration levels, peak detections etc) can be changed in the sidebar.

4. Once the time and state are selected, the dashboard computes and displays:
   - Raw accelerometer time series
   - Optional spike-filtered time series
   - PSDs for individual sensors and combined mirror geometries (M1–M7)
   - Optional overlay of 10–90 percentile reference bands
   - Automatically detected vibration peaks

5. Tables summarise the detected peaks and provide additional trace information.

All plots include buttons for saving figures to the local `figures/` directory.

---

## Expected data format

The dashboard assumes standard MNII raw data files:

- Filenames containing `raw_YYYY-MM-DD.hdf5`
- Top-level HDF5 keys matching `HH:MM:SS`
- Within each key: datasets named `sensor1`, `sensor2`, … containing 10-second, 1 kHz samples
- Reference PSDs located in `reference_psds/`, organised by UT and state

Older files (pre-2020) may not fully match this structure.

---

## Reference PSD files

Reference PSDs are stored under `reference_psds/UTX/`. Filenames encode UT, state, focus, and sensor using a human-readable pattern such as:

`UT1_2023-04_non-operations_allFoci_sensor1_psd_distribution_features.csv`

These CSV files contain percentile bands and peak metrics computed from a chosen reference month. They can be updated or replaced as needed, as long as the naming structure remains consistent.

---

## Features

- Time-domain inspection with optional spike filtering
- PSD calculation for sensors and combined mirrors
- Automatic peak detection with amplitude and width estimates
- Comparison against 10–90 percentile reference bands
- Sensor health classification using log-distance metrics
- Vibration source inventory for known frequencies
- Saving of figures and PSD output products

---

## Notes

- Large HDF5 files take a few seconds to read; caching is used to speed up repeated operations.
- Missing reference PSDs are flagged and will default to all-focus reference statistics.
- The application does not modify any input data files; all operations are read-only.