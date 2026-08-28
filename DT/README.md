# IMS Lightweight Digital Twin Pipeline

This folder contains the first DT-side pipeline for the bearing diagnosis QA system.

## Input data

The IMS raw dataset is not included in the public repository. Obtain it from
the dataset's official or authorized distribution and review its license
before use or redistribution. After obtaining it, place the archive locally
as `IMS.zip`. It includes:

- `IMS/1st_test.rar`
- `IMS/2nd_test.rar`
- `IMS/3rd_test.rar`
- `IMS/Readme Document for IMS Bearing Data.pdf`

The script expects these RAR files to be extracted so that the data root contains folders like:

```text
IMS/
  1st_test/
    2003.10.22.12.06.24
    ...
  2nd_test/
    2004.02.12.10.32.39
    ...
  3rd_test/
    2004.03.04.09.27.46
    ...
```

On macOS, install an extractor first if needed:

```bash
brew install unar
mkdir -p IMS
unzip IMS.zip -d .
unar IMS/1st_test.rar -o IMS
unar IMS/2nd_test.rar -o IMS
unar IMS/3rd_test.rar -o IMS
```

If you use conda instead:

```bash
conda install -c conda-forge unar
```

## Build DT features and database

Quick smoke test:

```bash
python DT/build_ims_dt_database.py \
  --data-root IMS \
  --output-dir DT/outputs \
  --db DT/outputs/ims_dt.db \
  --max-files-per-experiment 20
```

Full run:

```bash
python DT/build_ims_dt_database.py \
  --data-root IMS \
  --output-dir DT/outputs \
  --db DT/outputs/ims_dt.db
```

## Outputs

- `DT/outputs/ims_feature_records.csv`: per experiment / bearing / channel / timestamp features
- `DT/outputs/ims_bearing_snapshots.csv`: per experiment / bearing / timestamp aggregated DT state
- `DT/outputs/ims_latest_bearing_states.csv`: latest state per bearing
- `DT/outputs/ims_dt_summary.json`: summary metadata and latest states
- `DT/outputs/ims_dt.db`: SQLite database for Agent DT queries

## Feature strategy

The DT uses interpretable vibration features:

- time domain: RMS, peak, kurtosis, skewness, crest factor, impulse factor, clearance factor
- frequency domain: dominant frequency, spectral centroid, spectral entropy, low/mid/high band energy ratios
- health state: baseline-normalized weighted health index

Health states:

- `normal`
- `early_degradation`
- `degradation`
- `severe_degradation`
- `failure_near`

## DT time policy

The IMS dataset is an offline run-to-failure dataset, not a live sensor stream.
For DT questions, terms such as `current`, `latest`, `now`, and `at present`
are resolved as:

```text
latest_available_record = the latest available timestamp in the IMS DT database
for the requested experiment and bearing
```

For example, "the current health state of bearing_3 in 1st_test" means the
health state at the last stored snapshot of `bearing_3` in `1st_test`.

Trend questions use the full available sequence unless a future query interface
explicitly specifies an `as_of` timestamp or sequence index.
