from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    BaseInternalAttrs,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVar,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import (
    SPATIAL_REF_COORDS,  # noqa: F401
    TemplateConfig,
)
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)


class NasaSmapInternalAttrs(BaseInternalAttrs):
    """
    Variable specific attributes used internally to drive processing.
    Not written to the dataset.
    """

    h5_path: str


class NasaSmapDataVar(DataVar[NasaSmapInternalAttrs]):
    pass


class NasaSmapLevel336KmV9TemplateConfig(TemplateConfig[NasaSmapDataVar]):
    dims: tuple[Dim, ...] = ("time", "y", "x")
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("2015-04-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("1d")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="nasa-smap-level3-36km-v9",
            dataset_version="0.0.1",
            name="NASA SMAP L3 Radiometer Global Daily 36 km EASE-Grid Soil Moisture, Version 9",
            description="Soil moisture data from the NASA SMAP passive microwave radiometer.",
            attribution=(
                "O'Neill, P. E., Chan, S., Njoku, E. G., Jackson, T., Bindlish, R. & Chaubell, J. (2023). "
                "SMAP L3 Radiometer Global Daily 36 km EASE-Grid Soil Moisture. (SPL3SMP, Version 9). Boulder, Colorado USA. "
                "NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/4XXOGX0OOW1S."
            ),
            spatial_domain="Global",
            spatial_resolution="36 km",
            time_domain=f"Retrievals from {self.append_dim_start} to Present",
            time_resolution="Daily AM and PM observations",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        """Returns a dictionary of dimension names to coordinates for the dataset."""
        times = self.append_dim_coordinates(
            self.append_dim_start + self.append_dim_frequency
        )
        y = ...
        x = ...

        return {"time": times, "y": y, "x": x}

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """
        Return a dictionary of non-dimension coordinates for the dataset.
        Called whenever len(ds.append_dim) changes.
        """
        return {}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coords(self) -> Sequence[Coordinate]:
        """Define metadata and encoding for each coordinate."""
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        return [
            Coordinate(
                name=self.append_dim,
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=dim_coords[self.append_dim].min().isoformat(), max="Present"
                    ),
                ),
            ),
            Coordinate(
                name="y",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["y"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="??",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["y"].min()),
                        max=float(dim_coords["y"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="x",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["x"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="??",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["x"].min()),
                        max=float(dim_coords["x"].max()),
                    ),
                ),
            ),
            #     Coordinate(
            #         name="spatial_ref",
            #         encoding=Encoding(
            #             dtype="int64",
            #             fill_value=0,
            #             chunks=(),  # Scalar coordinate
            #             shards=None,
            #         ),
            #         attrs=CoordinateAttrs(
            #             units=None,
            #             statistics_approximate=None,
            #             # Deterived by running `ds.rio.write_crs("+proj=longlat +a=6371229 +b=6371229 +no_defs +type=crs")["spatial_ref"].attrs
            #             crs_wkt='GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371229,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],AXIS["Latitude",NORTH]]',
            #             semi_major_axis=6371229.0,
            #             semi_minor_axis=6371229.0,
            #             inverse_flattening=0.0,
            #             reference_ellipsoid_name="unknown",
            #             longitude_of_prime_meridian=0.0,
            #             prime_meridian_name="Greenwich",
            #             geographic_crs_name="unknown",
            #             horizontal_datum_name="unknown",
            #             grid_mapping_name="latitude_longitude",
            #             spatial_ref='GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371229,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],AXIS["Latitude",NORTH]]',
            #             comment="This coordinate reference system matches the source data which follows WMO conventions of assuming the earth is a perfect sphere with a radius of 6,371,229m. It is similar to EPSG:4326, but EPSG:4326 uses a more accurate representation of the earth's shape.",
            #         ),
            #     ),
            # ]
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[NasaSmapDataVar]:
        """Define metadata and encoding for each data variable."""
        # 27MB uncompressed, ~5MB compressed
        var_chunks: dict[Dim, int] = {
            "time": 360,
            "latitude": 136,
            "longitude": 138,
        }
        var_shards: dict[Dim, int] = {
            "time": var_chunks["time"],
            "latitude": var_chunks["latitude"] * 3,  # all chunks in one shard
            "longitude": var_chunks["longitude"] * 7,  # all chunks in one shard
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        default_keep_mantissa_bits = 7

        return [
            NasaSmapDataVar(
                name="soil_moisture_am",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="soil_moisture_am",
                    long_name="Soil Moisture (AM)",
                    units="m³/m³",
                    step_type="instant",
                ),
                internal_attrs=NasaSmapInternalAttrs(
                    h5_path="/Soil_Moisture_Retrieval_Data_AM/soil_moisture",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NasaSmapDataVar(
                name="soil_moisture_pm",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="soil_moisture_pm",
                    long_name="Soil Moisture (PM)",
                    units="m³/m³",
                    step_type="instant",
                ),
                internal_attrs=NasaSmapInternalAttrs(
                    h5_path="/Soil_Moisture_Retrieval_Data_PM/soil_moisture",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
        ]
