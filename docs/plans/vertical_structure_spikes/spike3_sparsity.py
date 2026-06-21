"""Spike 3: quantify single-level vs vertical-level structure in a real GFS file.

Pulls a GFS pgrb2 .idx (the GRIB index NODD publishes alongside each file) and
classifies every message by its level type, so we can see how dense/sparse a
(variable x level) grid would actually be.
"""

import re
import urllib.request
from collections import defaultdict

# A recent-ish GFS 0.25deg pgrb2 file index. pgrb2 carries the full pressure-level set.
URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20240101/00/atmos/gfs.t00z.pgrb2.0p25.f000.idx"

raw = urllib.request.urlopen(URL, timeout=30).read().decode()
lines = [ln for ln in raw.splitlines() if ln.strip()]
print(f"total GRIB messages in file: {len(lines)}")

# idx line: <n>:<offset>:d=<date>:<param>:<level>:<forecast>:...
pressure = defaultdict(set)  # param -> set of hPa levels
model_lvl = defaultdict(set)  # param -> set of hybrid levels
single = defaultdict(list)  # level-string -> [params]
level_types = defaultdict(int)

for line in lines:
    parts = line.split(":")
    param = parts[3]
    level = parts[4]
    m = re.match(r"^(\d+) mb$", level)
    mh = re.match(r"^(\d+) hybrid level$", level)
    if m:
        pressure[param].add(int(m.group(1)))
        level_types["pressure (mb)"] += 1
    elif mh:
        model_lvl[param].add(int(mh.group(1)))
        level_types["hybrid (model)"] += 1
    else:
        single[level].append(param)
        level_types["single/other"] += 1

print("\n=== message counts by level category ===")
for k, v in sorted(level_types.items(), key=lambda x: -x[1]):
    print(f"  {v:5d}  {k}")

print(f"\n=== PRESSURE-LEVEL variables: {len(pressure)} distinct params ===")
all_p_levels = sorted({lv for s in pressure.values() for lv in s}, reverse=True)
print(f"distinct pressure levels present: {len(all_p_levels)}")
print(f"levels: {all_p_levels}")
print(
    f"\n(variable x level) cross product if fully dense: "
    f"{len(pressure)} x {len(all_p_levels)} = {len(pressure) * len(all_p_levels)}"
)
actual = sum(len(s) for s in pressure.values())
print(f"actual pressure messages: {actual}")
print(f"=> pressure grid density: {actual / (len(pressure) * len(all_p_levels)):.1%}")
print("\nper-variable pressure-level counts (how many levels each var has):")
for p, s in sorted(pressure.items(), key=lambda x: -len(x[1])):
    print(f"  {len(s):3d} levels  {p}")

print(
    f"\n=== SINGLE-LEVEL / other: {sum(len(v) for v in single.values())} messages "
    f"across {len(single)} distinct level strings ==="
)
# how sparse is (variable x single-level) if we tried to grid it?
sl_pairs = [(p, lv) for lv, ps in single.items() for p in ps]
sl_vars = {p for p, _ in sl_pairs}
sl_levels = set(single.keys())
print(f"distinct single-level 'level' strings: {len(sl_levels)}")
print(f"distinct single-level params: {len(sl_vars)}")
print(
    f"(param x single-level) full grid: {len(sl_vars)} x {len(sl_levels)} = "
    f"{len(sl_vars) * len(sl_levels)}; actual messages: {len(sl_pairs)}"
)
print(
    f"=> single-level grid density: {len(sl_pairs) / (len(sl_vars) * len(sl_levels)):.1%}"
)
print("\nsample single-level level strings:")
for lv in list(single.keys())[:25]:
    print(f"  {lv!r}: {len(single[lv])} params")
