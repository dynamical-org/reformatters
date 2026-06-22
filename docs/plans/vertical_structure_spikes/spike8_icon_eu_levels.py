import re
import urllib.request
from collections import defaultdict

BASE = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00"


def get(u):
    return urllib.request.urlopen(u, timeout=30).read().decode()


# 1. variable subdirectories
top = get(f"{BASE}/")
vars_ = sorted(set(re.findall(r'href="([a-z0-9_]+)/"', top)))
print(f"ICON-EU /00/ variable dirs ({len(vars_)}): {vars_}")

# 2. which vars have pressure-level / model-level / single-level files
leveltype_by_var = defaultdict(set)
for v in vars_:
    try:
        listing = get(f"{BASE}/{v}/")
    except Exception as e:
        continue
    for lt in ("single-level", "pressure-level", "model-level"):
        if lt in listing:
            leveltype_by_var[lt].add(v)

for lt in ("single-level", "pressure-level", "model-level"):
    print(f"\n{lt}: {len(leveltype_by_var[lt])} vars -> {sorted(leveltype_by_var[lt])}")

# 3. enumerate the actual pressure levels and model levels present for temperature `t`
listing = get(f"{BASE}/t/")
files = re.findall(r'href="(icon-eu[^"]+\.grib2\.bz2)"', listing)
plev, mlev = set(), set()
samp = {}
for f in files:
    if "pressure-level" in f:
        m = re.search(r"pressure-level_\d+_\d+_(\d+)_T", f)
        if m:
            plev.add(int(m.group(1)))
            samp.setdefault("pl", f)
    elif "model-level" in f:
        m = re.search(r"model-level_\d+_\d+_(\d+)_T", f)
        if m:
            mlev.add(int(m.group(1)))
            samp.setdefault("ml", f)
print(f"\ntemperature `t`: {len(plev)} pressure levels {sorted(plev)}")
print(
    f"temperature `t`: {len(mlev)} model levels min={min(mlev) if mlev else '-'} max={max(mlev) if mlev else '-'}"
)
print("sample pressure file:", samp.get("pl"))
print("sample model file:   ", samp.get("ml"))
