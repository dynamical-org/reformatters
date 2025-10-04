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
    SPATIAL_REF_COORDS,
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

        # From gdalinfo output:
        # Size is 964, 406 (x, y)
        # ulxmap - x-axis map coordinate of the outer edge of the upper-left pixel - 17367530.
        # ulymap - y-axis map coordinate of the outer edge of the upper-left pixel 7314540.
        # Grid cell size (x, y pixel dimensions) 36,032.22 m (x) 36,032.22 m (y)

        x_size = 964
        y_size = 406
        cell_size = 36032.22  # meters
        ulxmap = -17367530.0
        ulymap = 7314540.0

        # Calculate x coordinates (center of the cells)
        x = np.arange(ulxmap + cell_size / 2, ulxmap + x_size * cell_size, cell_size)
        # Calculate y coordinates (center of the cells), decreasing from top to bottom
        y = np.arange(ulymap - cell_size / 2, ulymap - y_size * cell_size, -cell_size)

        return {"time": times, "y": y, "x": x}

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """
        Return a dictionary of non-dimension coordinates for the dataset.
        Called whenever len(ds.append_dim) changes.
        """
        return {"spatial_ref": SPATIAL_REF_COORDS}

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
                    units="m",
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
                    units="m",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["x"].min()),
                        max=float(dim_coords["x"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="spatial_ref",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    chunks=(),  # Scalar coordinate
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units=None,
                    statistics_approximate=None,
                    crs_wkt='PROJCRS["EASE-Grid 2.0 Global",BASEGEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]],CONVERSION["unnamed",METHOD["Lambert Cylindrical Equal Area",ID["EPSG",9835]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["Scale factor at natural origin",1,SCALEUNIT["unity",1],ID["EPSG",8805]],PARAMETER["False easting",0,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["(Y)",north,ORDER[1],LENGTHUNIT["metre",1]],AXIS["(X)",east,ORDER[2],LENGTHUNIT["metre",1]],ID["EPSG",6933]]',
                    semi_major_axis=6378137.0,
                    semi_minor_axis=6356752.314245179,
                    inverse_flattening=298.257223563,
                    reference_ellipsoid_name="WGS 84",
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    geographic_crs_name="WGS 84",
                    horizontal_datum_name="World Geodetic System 1984",
                    grid_mapping_name="lambert_cylindrical_equal_area",
                    spatial_ref="+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs",
                    false_easting=0.0,
                    false_northing=0.0,
                    latitude_of_projection_origin=0.0,
                    longitude_of_central_meridian=0.0,
                    projected_crs_name="EASE-Grid 2.0 Global",
                    standard_parallel=(30.0, 30.0),
                    comment="This coordinate reference system describes the EASE-Grid 2.0 Global projection (EPSG:6933).",
                ),
            ),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[NasaSmapDataVar]:
        """Define metadata and encoding for each data variable."""
        var_chunks: dict[Dim, int] = {
            "time": 360,
            "y": 136,
            "x": 138,
        }
        var_shards: dict[Dim, int] = {
            "time": var_chunks["time"],
            "y": var_chunks["y"] * 3,
            "x": var_chunks["x"] * 6,
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
                    h5_path="//Soil_Moisture_Retrieval_Data_AM/soil_moisture_am",
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
                    h5_path="//Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
        ]
