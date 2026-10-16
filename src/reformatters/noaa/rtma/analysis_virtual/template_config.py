from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from gribberish.zarr import GribberishCodec
from pydantic import computed_field
from zarr.codecs import ScaleOffset

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.projection import latitude_longitude_grids
from reformatters.common.template_config import SPATIAL_REF_COORDS, TemplateConfig
from reformatters.common.types import AppendDim, Array1D, Dims, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.noaa.rtma.grib_header import (
    GRID_LA1,
    GRID_LO1,
    GRID_NX,
    GRID_NY,
    GRID_PROJ,
    GRID_SPACING,
    FieldIdentity,
)
from reformatters.noaa.rtma.rtma_config_models import RtmaDataVar, RtmaInternalAttrs


def spatial_coordinates() -> tuple[Array1D[np.float64], Array1D[np.float64]]:
    x0, y0 = pyproj.Proj(GRID_PROJ)(GRID_LO1, GRID_LA1)
    return (
        y0 + np.arange(GRID_NY - 1, -1, -1, dtype=np.float64) * GRID_SPACING,
        x0 + np.arange(GRID_NX, dtype=np.float64) * GRID_SPACING,
    )


class NoaaRtmaAnalysisVirtualTemplateConfig(TemplateConfig[RtmaDataVar]):
    dims: Dims = {ROOT: ("time", "y", "x")}
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("2017-05-02T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("1h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="noaa-rtma-analysis-virtual",
            dataset_version="0.1.0",
            name="NOAA RTMA analysis, virtual",
            description="Hourly Real-Time Mesoscale Analysis (RTMA) operated by NOAA NWS NCEP on the native CONUS grid.",
            attribution="NOAA NWS NCEP RTMA data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            license="CC-BY-4.0",
            spatial_domain="Continental United States",
            spatial_resolution="2.5 km",
            time_domain=f"{self.append_dim_start} UTC to Present",
            time_resolution="1 hour",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        y, x = spatial_coordinates()
        return {
            "time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "y": y,
            "x": x,
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        latitude, longitude = latitude_longitude_grids(
            GRID_PROJ, ds.x.values, ds.y.values
        )
        return {
            "latitude": (("y", "x"), latitude),
            "longitude": (("y", "x"), longitude),
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        y, x = spatial_coordinates()
        latitude, longitude = latitude_longitude_grids(GRID_PROJ, x, y)
        coords = [
            Coordinate(
                name="time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    chunks=self.append_dim_coordinate_chunk_size(),
                    shards=None,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                ),
                attrs=CoordinateAttrs(
                    long_name="Time",
                    standard_name="time",
                    axis="T",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(), max="Present"
                    ),
                ),
            ),
        ]
        for name, values, standard_name, units, axis in (
            ("y", y, "projection_y_coordinate", "m", "Y"),
            ("x", x, "projection_x_coordinate", "m", "X"),
            ("latitude", latitude, "latitude", "degree_north", None),
            ("longitude", longitude, "longitude", "degree_east", None),
        ):
            coords.append(
                Coordinate(
                    name=name,
                    encoding=Encoding(
                        dtype="float64" if values.ndim == 1 else "float32",
                        fill_value=np.nan,
                        chunks=values.shape,
                        shards=None,
                        compressors=[
                            BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
                            if values.ndim == 1
                            else BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE
                        ],
                    ),
                    attrs=CoordinateAttrs(
                        long_name=f"{axis} coordinate of projection"
                        if axis is not None
                        else name.capitalize(),
                        standard_name=standard_name,
                        units=units,
                        axis=axis,
                        statistics_approximate=StatisticsApproximate(
                            min=round(float(values.min()), 3)
                            if axis
                            else float(values.min()),
                            max=round(float(values.max()), 3)
                            if axis
                            else float(values.max()),
                        ),
                    ),
                )
            )
        crs = pyproj.CRS(GRID_PROJ)
        coords.append(
            Coordinate(
                name="spatial_ref",
                encoding=Encoding(dtype="int64", fill_value=0, chunks=(), shards=None),
                attrs=CoordinateAttrs(
                    units=None,
                    statistics_approximate=None,
                    spatial_ref=crs.to_wkt(),
                    **crs.to_cf(),
                ),
            )
        )
        return coords

    @computed_field
    @property
    def data_vars(self) -> Sequence[RtmaDataVar]:
        return [
            _var(
                "geopotential_height_surface",
                "HGT",
                (0, 3, 5, 1, 0),
                "gh",
                "Geopotential height",
                "m",
                "geopotential_height",
            ),
            _var(
                "pressure_surface",
                "PRES",
                (0, 3, 0, 1, 0),
                "sp",
                "Surface pressure",
                "Pa",
                "surface_air_pressure",
            ),
            _var(
                "temperature_2m",
                "TMP",
                (0, 0, 0, 103, 2),
                "2t",
                "2 metre temperature",
                "degree_Celsius",
                "air_temperature",
            ),
            _var(
                "dew_point_temperature_2m",
                "DPT",
                (0, 0, 6, 103, 2),
                "2d",
                "2 metre dewpoint temperature",
                "degree_Celsius",
                "dew_point_temperature",
            ),
            _var(
                "specific_humidity_2m",
                "SPFH",
                (0, 1, 0, 103, 2),
                "q",
                "Specific humidity",
                "1",
                "specific_humidity",
            ),
            _var(
                "wind_u_10m",
                "UGRD",
                (0, 2, 2, 103, 10),
                "10u",
                "10 metre U wind component",
                "m s-1",
                "x_wind",
                "Velocity along the model grid's x dimension, not eastward velocity.",
            ),
            _var(
                "wind_v_10m",
                "VGRD",
                (0, 2, 3, 103, 10),
                "10v",
                "10 metre V wind component",
                "m s-1",
                "y_wind",
                "Velocity along the model grid's y dimension, not northward velocity.",
            ),
            _var(
                "wind_speed_10m",
                "WIND",
                (0, 2, 1, 103, 10),
                "10si",
                "10 metre wind speed",
                "m s-1",
                "wind_speed",
            ),
            _var(
                "wind_direction_10m",
                "WDIR",
                (0, 2, 0, 103, 10),
                "10wdir",
                "10 metre wind direction",
                "degree",
                "wind_from_direction",
                "Direction the wind blows from, clockwise from true north.",
            ),
            _var(
                "wind_gust_10m",
                "GUST",
                (0, 2, 22, 103, 10),
                "gust",
                "Wind speed (gust)",
                "m s-1",
                "wind_speed_of_gust",
            ),
            _var(
                "visibility_surface",
                "VIS",
                (0, 19, 0, 1, 0),
                "vis",
                "Visibility",
                "m",
                "visibility_in_air",
            ),
            _var(
                "cloud_ceiling_height",
                "CEIL",
                (0, 6, 13, 215, 0),
                "ceil",
                "Ceiling",
                "m",
                None,
            ),
            _var(
                "total_cloud_cover_atmosphere",
                "TCDC",
                (0, 6, 1, 200, 0),
                "tcc",
                "Total cloud cover",
                "percent",
                "cloud_area_fraction",
            ),
        ]


def _var(
    name: str,
    element: str,
    identity: FieldIdentity,
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None,
    comment: str | None = None,
) -> RtmaDataVar:
    return RtmaDataVar(
        name=name,
        encoding=Encoding(
            dtype="float64",
            fill_value=np.nan,
            chunks=(1, GRID_NY, GRID_NX),
            shards=None,
            compressors=(),
            filters=(ScaleOffset(offset=-273.15, scale=1.0).to_dict(),)
            if units == "degree_Celsius"
            else (),
            serializer=GribberishCodec(
                var=element, north_up=True, adjust_longitude_range=True
            ).to_dict(),
        ),
        attrs=DataVarAttrs(
            short_name=short_name,
            long_name=long_name,
            units=units,
            standard_name=standard_name,
            step_type="instant",
            comment=comment,
        ),
        internal_attrs=RtmaInternalAttrs(
            grib_element=element,
            grib_identity=identity,
            keep_mantissa_bits=12
            if name == "wind_direction_10m"
            else 11
            if units == "Pa"
            else 6
            if name.startswith("wind_")
            else 7,
        ),
    )
