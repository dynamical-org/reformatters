import re, urllib.request
from collections import defaultdict


def classify(url, label):
    raw = urllib.request.urlopen(url, timeout=30).read().decode()
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    pres, hyb, single = defaultdict(set), defaultdict(set), defaultdict(list)
    for line in lines:
        p = line.split(":")
        param, level = p[3], p[4]
        if m := re.match(r"^(\d+) mb$", level):
            pres[param].add(int(m.group(1)))
        elif mh := re.match(r"^(\d+) hybrid level$", level):
            hyb[param].add(int(mh.group(1)))
        else:
            single[level].append(param)
    nhyb_levels = sorted({l for s in hyb.values() for l in s})
    print(f"\n== {label} ==  ({len(lines)} msgs)")
    print(
        f"  pressure: {len(pres)} vars x {len(sorted({l for s in pres.values() for l in s}))} levels"
    )
    print(f"  hybrid/model: {len(hyb)} vars across levels {nhyb_levels}")
    print(
        f"  single/other: {sum(len(v) for v in single.values())} msgs, {len(single)} level-strings"
    )


base = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20240101/00/atmos"
classify(f"{base}/gfs.t00z.pgrb2.0p25.f000.idx", "GFS pgrb2 0p25")
classify(f"{base}/gfs.t00z.pgrb2b.0p25.f000.idx", "GFS pgrb2b 0p25")
