# Enumerate MRMS CONUS products from the real public bucket and find the 3D
# (multi-height) products. The trailing _NN.NN token is the height (km) only for
# the 3D mosaics; for everything else it is a constant naming token.
import re
import urllib.request
from collections import defaultdict

B = "https://noaa-mrms-pds.s3.amazonaws.com"


def get(u):
    return urllib.request.urlopen(u, timeout=30).read().decode()


x = get(f"{B}/?list-type=2&delimiter=/&prefix=CONUS/")
prods = re.findall(r"<Prefix>CONUS/([^<]+)/</Prefix>", x)
heights = defaultdict(set)
for p in prods:
    m = re.match(r"^(.*?)_(\d\d\.\d\d)$", p)
    if m:
        heights[m.group(1)].add(m.group(2))
print(f"Total CONUS products: {len(prods)}")
print("Products with >1 height level (genuine 3D stacks):")
for base in sorted(heights):
    hs = sorted(heights[base], key=float)
    if len(hs) > 1:
        print(f"  {base}: {len(hs)} levels {hs[0]}..{hs[-1]} km")
