# Data Explorer Agent

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

Analyzes Dermis tactile-sensor recordings (magnetic touch pods + IMU at 350 Hz) for **pandas-style tables over Dermis recordings: describe, crosstab, pivot and group-by, with the pandas expression behind each table**.

Focus: tables you can paste, group-by breakdowns, per-episode statistics.

Reads recordings from SpacetimeDB (local dataset fallback), computes contact force, slip vibration, sensor saturation and noise, judges them against this application's limits, charts the result and recommends what to change.

Keywords: table, tables, pivot, crosstab, groupby, group by, describe, stats, statistics, breakdown, columns, rows, pandas, dataframe, spreadsheet, explore

Example questions:
- Show the dataset overview
- Peak force by object and scenario
- Latency by safe grip
- Table stats for episode 4

## Limitations

Recordings are the synthetic dermis-sim-v1 set until hardware sessions land in SpacetimeDB; force is simulated ground truth or an uncalibrated field estimate.
