from collections.abc import Sequence
from typing import Any, Literal, assert_never

import numpy as np
import pandas as pd
import xarray as xr
from gribberish.zarr import GribberishCodec
from pydantic import computed_field
from zarr.codecs import ScaleOffset

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.pydantic import replace
from reformatters.common.types import Dims, Timestamp
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar, EcmwfInternalAttrs
from reformatters.ecmwf.ecmwf_template_config import (
    Ecmwf025DegreeForecastTemplateConfig,
)

type FastModel = Literal["ifs", "aifs"]


def lead_hours(model: FastModel, init_hour: int) -> tuple[int, ...]:
    match model:
        case "ifs":
            return (
                *range(91),
                *range(93, 145, 3),
                *(range(150, 361, 6) if init_hour in (0, 12) else ()),
            )
        case "aifs":
            return tuple(range(0, 361, 6))
        case unexpected:
            assert_never(unexpected)


class EcmwfFastTemplateConfig(Ecmwf025DegreeForecastTemplateConfig[EcmwfDataVar]):
    model: FastModel
    append_dim_start: Timestamp = pd.Timestamp("2026-11-01T00:00")
    dims: Dims = {
        ROOT: ("init_time", "lead_time", "ensemble_member", "latitude", "longitude")
    }

    def dimension_coordinates(self) -> dict[str, Any]:
        return super().dimension_coordinates() | {
            "lead_time": pd.to_timedelta(lead_hours(self.model, 0), unit="h").as_unit(
                "s"
            ),
            "ensemble_member": np.arange(51, dtype=np.int64),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        return super().derive_coordinates(ds) | {
            "expected_forecast_length": (
                (self.append_dim,),
                np.array(
                    [
                        lead_hours(self.model, init.hour)[-1]
                        for init in pd.to_datetime(ds.init_time.values)
                    ],
                    dtype="timedelta64[h]",
                ),
            )
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        coords = []
        match self.model:
            case "ifs":
                radius = 6367470.0
            case "aifs":
                radius = 6371229.0
            case unexpected:
                assert_never(unexpected)
        for base_coord in super().coords:
            coord = base_coord
            if coord.name == "spatial_ref":
                assert coord.attrs.crs_wkt is not None
                assert coord.attrs.spatial_ref is not None
                attrs = coord.attrs.model_dump()
                attrs.update(
                    semi_major_axis=radius,
                    semi_minor_axis=radius,
                    crs_wkt=coord.attrs.crs_wkt.replace("6371229", str(int(radius))),
                    spatial_ref=coord.attrs.spatial_ref.replace(
                        "6371229", str(int(radius))
                    ),
                    comment=f"The source assumes a spherical earth with a radius of {radius:,.0f}m.",
                )
                coord = replace(coord, attrs=CoordinateAttrs(**attrs))
            elif coord.name == "expected_forecast_length":
                match self.model:
                    case "ifs":
                        coord = replace(
                            coord,
                            attrs=replace(
                                coord.attrs,
                                statistics_approximate=StatisticsApproximate(
                                    min="6 days 00:00:00", max="15 days 00:00:00"
                                ),
                            ),
                        )
                    case "aifs":
                        pass
                    case unexpected:
                        assert_never(unexpected)
            coords.append(coord)
        return [
            *coords,
            Coordinate(
                name="ensemble_member",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    chunks=51,
                    shards=None,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=CoordinateAttrs(
                    long_name="Ensemble member",
                    standard_name="realization",
                    units="1",
                    statistics_approximate=StatisticsApproximate(min=0, max=50),
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[EcmwfDataVar]:
        quantities = [
            (
                "temperature_2m",
                "2t",
                "2 metre temperature",
                "degree_Celsius",
                "air_temperature",
            ),
            (
                "wind_u_10m",
                "10u",
                "10 metre U wind component",
                "m s-1",
                "eastward_wind",
            ),
            (
                "wind_v_10m",
                "10v",
                "10 metre V wind component",
                "m s-1",
                "northward_wind",
            ),
            (
                "wind_u_100m",
                "100u",
                "100 metre U wind component",
                "m s-1",
                "eastward_wind",
            ),
            (
                "wind_v_100m",
                "100v",
                "100 metre V wind component",
                "m s-1",
                "northward_wind",
            ),
            (
                "total_precipitation_run_total_surface",
                "tp",
                "Total precipitation",
                "kg m-2",
                "precipitation_amount",
            ),
            (
                "downward_short_wave_radiation_run_total_surface",
                "ssrd",
                "Surface short-wave (solar) radiation downwards",
                "W s m-2",
                "integral_wrt_time_of_surface_downwelling_shortwave_flux_in_air",
            ),
        ]
        result = []
        for name, param, long_name, units, standard_name in quantities:
            accumulated = param in ("tp", "ssrd")
            filters = []
            source_units = "J m-2" if param == "ssrd" else units
            if param == "2t":
                source_units = "K"
                filters = [ScaleOffset(offset=-273.15, scale=1.0).to_dict()]
            elif param == "tp":
                match self.model:
                    case "ifs":
                        source_units = "m"
                        filters = [ScaleOffset(offset=0.0, scale=0.001).to_dict()]
                    case "aifs":
                        filters = []
                    case unexpected:
                        assert_never(unexpected)
            result.append(
                EcmwfDataVar(
                    name=name,
                    encoding=Encoding(
                        dtype="float64",
                        fill_value=np.nan,
                        chunks=(1, 1, 1, 721, 1440),
                        shards=None,
                        compressors=(),
                        filters=filters,
                        serializer=GribberishCodec(
                            var=param, adjust_longitude_range=True, north_up=True
                        ).to_dict(),
                    ),
                    attrs=DataVarAttrs(
                        short_name=param,
                        long_name=long_name,
                        units=units,
                        standard_name=standard_name,
                        step_type="accum" if accumulated else "instant",
                        comment=(
                            "Accumulated from the forecast initialization time to this step."
                            if accumulated
                            else None
                        ),
                    ),
                    internal_attrs=EcmwfInternalAttrs(
                        grib_index_param=param,
                        grib_comment="",
                        grib_element="",
                        grib_description="",
                        source_units=source_units,
                        keep_mantissa_bits="no-rounding",
                    ),
                )
            )
        return result
