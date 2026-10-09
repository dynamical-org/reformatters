import json
from pathlib import Path
from typing import Any


def write_single_grib_chunk(
    store: Path, variable_name: str, metadata: dict[str, Any], content: bytes
) -> None:
    metadata = {
        **metadata,
        "shape": metadata["chunk_grid"]["configuration"]["chunk_shape"],
    }
    array_path = store / variable_name
    chunk_path = array_path / "c" / "/".join("0" for _ in metadata["shape"])
    chunk_path.parent.mkdir(parents=True)
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )
    (array_path / "zarr.json").write_text(json.dumps(metadata))
    chunk_path.write_bytes(content)
