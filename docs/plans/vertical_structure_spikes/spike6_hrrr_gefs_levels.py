# ruff: skip
import re, urllib.request
from collections import defaultdict


def fetch(url):
    try:
        return urllib.request.urlopen(url, timeout=30).read().decode()
    except Exception as e:
        return f"__ERR__ {e}"


def classify(url, label):
    raw = fetch(url)
    if raw.startswith("__ERR__"):
        print(f"\n== {label} ==  FETCH FAILED: {raw[:120]}\n   {url}")
        return
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    pres, hyb, depth, single = (
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
        defaultdict(list),
    )
    for line in lines:
        p = line.split(":")
        if len(p) < 5:
            continue
        param, level = p[3], p[4]
        if m := re.match(r"^(\d+) mb$", level):
            pres[param].add(int(m.group(1)))
        elif mh := re.match(r"^(\d+) hybrid level$", level):
            hyb[param].add(int(mh.group(1)))
        elif md := re.match(r"^(\d+) sigma", level):
            hyb[param].add(int(md.group(1)))
        else:
            single[level].append(param)
    plev = sorted({l for s in pres.values() for l in s})
    hlev = sorted({l for s in hyb.values() for l in s})
    print(f"\n== {label} ==  ({len(lines)} msgs)  {url.split('/')[-1]}")
    print(
        f"  pressure(mb):  {len(pres):2d} vars x {len(plev):3d} levels  range {plev[:1]}..{plev[-1:]} "
    )
    print(
        f"  hybrid/sigma:  {len(hyb):2d} vars x {len(hlev):3d} levels  range {hlev[:1]}..{hlev[-1:]}"
    )
    print(
        f"  single/other:  {sum(len(v) for v in single.values()):3d} msgs, {len(single)} level-strings"
    )


# HRRR: wrfprs (pressure) vs wrfnat (native/hybrid model levels)
hb = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240101/conus"
classify(f"{hb}/hrrr.t00z.wrfprsf00.grib2.idx", "HRRR wrfprs (pressure product)")
classify(f"{hb}/hrrr.t00z.wrfnatf00.grib2.idx", "HRRR wrfnat (native product)")

# GEFS 0.25 s-file and 0.5 a/b files; check pressure level density + resolution split
gb = "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240101/00/atmos"
classify(f"{gb}/pgrb2sp25/geavg.t00z.pgrb2s.0p25.f000.idx", "GEFS pgrb2s 0p25 (s-file)")
classify(f"{gb}/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f000.idx", "GEFS pgrb2a 0p50")
classify(f"{gb}/pgrb2bp5/geavg.t00z.pgrb2b.0p50.f000.idx", "GEFS pgrb2b 0p50")
