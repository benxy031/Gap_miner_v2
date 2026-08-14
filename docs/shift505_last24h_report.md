# Shift 505 Report - Last 24 Hours

Source: `gapchain.sqlite3`

Filter used:

- `shift = 505`
- `datetime(date || ' ' || time) >= datetime('now', '-24 hours')`

## Summary

- Blocks found: 155
- Time span: 2026-08-03 17:41:34 -> 2026-08-04 06:06:27
- Merit range: 19.897860173062 -> 26.679378786776
- Average merit: 21.245645423939
- Gap size range: 10,494 -> 14,056

## Prime-digit distribution

| Primedigits | Count | Avg merit | Min merit | Max merit |
|---|---:|---:|---:|---:|
| 229 | 93 | 21.320 | 19.903228367365 | 26.679378786776 |
| 230 | 62 | 21.134 | 19.897860173062 | 24.759127718093 |

## Highest-merit blocks

| Height | Timestamp | Merit | Gap size | Primedigits | Adder |
|---|---|---:|---:|---:|---:|
| 2507692 | 2026-08-03 18:30:38 | 26.679378786776 | 14056 | 229 | 3.63357190065702e+151 |
| 2508002 | 2026-08-04 05:51:38 | 24.759127718093 | 13056 | 230 | 3.48391252066412e+151 |
| 2507909 | 2026-08-04 01:23:00 | 24.544111795966 | 12942 | 230 | 7.86727244975794e+151 |
| 2507964 | 2026-08-04 04:08:38 | 24.533527170909 | 12936 | 229 | 2.41006002796868e+151 |
| 2507851 | 2026-08-03 23:13:28 | 24.472755598733 | 12904 | 229 | 3.45076152556995e+151 |
| 2507867 | 2026-08-03 23:45:47 | 24.459227228140 | 12886 | 229 | 7.05538535713435e+151 |
| 2507908 | 2026-08-04 01:21:40 | 24.264811939332 | 12790 | 229 | 4.39783301493225e+150 |
| 2507823 | 2026-08-03 22:28:01 | 24.047566934785 | 12680 | 229 | 8.33138434599107e+151 |
| 2507880 | 2026-08-04 00:04:38 | 23.928927322812 | 12614 | 229 | 3.47281516614885e+151 |
| 2507783 | 2026-08-03 21:16:52 | 23.791985517897 | 12548 | 230 | 2.4679920141166e+151 |

## Notes

- All 155 rows in this window are shift 505.
- The majority of blocks are in the 229-primedigit class.
- No extra classification marker exists in the database, so the report treats every matching shift 505 row in the last 24 hours as a block found by this miner record.

## Quick take

- Best block in the window: height 2507692, merit 26.679378786776, gap 14056.
- The run is stable: merits cluster around ~21-22 with a few strong outliers above 24.
