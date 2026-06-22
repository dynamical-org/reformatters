import json
import re
import urllib.request
from collections import defaultdict

B = "https://ecmwf-forecasts.s3.amazonaws.com"
DATE = "20260621"


def get(u):
    return urllib.request.urlopen(u, timeout=30).read().decode()


def list_keys(prefix):
    y = get(f"{B}/?list-type=2&prefix={prefix}")
    return re.findall(r"<Key>([^<]+)</Key>", y)


def enumerate_index(index_key, label):
    raw = get(f"{B}/{index_key}")
    by = defaultdict(lambda: defaultdict(set))  # levtype -> param -> set(levels)
    n = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        n += 1
        lt = d.get("levtype", "?")
        param = d.get("param", "?")
        lev = d.get("levelist", "-")
        by[lt][param].add(str(lev))
    print(f"\n==== {label} ====  ({n} messages)  {index_key.split('/')[-1]}")
    for lt in sorted(by):
        params = by[lt]
        alllev = sorted(
            {l for s in params.values() for l in s if l != "-"},
            key=lambda v: (len(v), v),
        )
        print(f"  levtype={lt!r}: {len(params)} params, {len(alllev)} distinct levels")
        if lt in ("pl", "ml"):
            print(f"     levels: {alllev}")
            print(f"     params: {sorted(params)}")


for model, stream, lab in [
    ("ifs", "enfo", "IFS ENS (enfo)"),
    ("aifs-single", "oper", "AIFS Single (oper)"),
    ("aifs-ens", "enfo", "AIFS ENS (enfo)"),
]:
    pre = f"{DATE}/00z/{model}/0p25/{stream}/"
    keys = list_keys(pre)
    idx = [k for k in keys if k.endswith(".index")]
    # pick the step-0 (or earliest) index
    idx_sorted = sorted(idx)
    if not idx_sorted:
        print(
            f"\n==== {lab} ====  NO .index files under {pre}; sample keys: {keys[:4]}"
        )
        continue
    enumerate_index(idx_sorted[0], lab)
