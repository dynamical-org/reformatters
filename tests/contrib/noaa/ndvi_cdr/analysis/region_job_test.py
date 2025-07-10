from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.types import ArrayFloat32, ArrayInt16
from reformatters.common.zarr import get_zarr_store
from reformatters.contrib.noaa.ndvi_cdr.ndvi_cdr.analysis.region_job import (
    NoaaNdviCdrAnalysisRegionJob,
    NoaaNdviCdrAnalysisSourceFileCoord,
)
from reformatters.contrib.noaa.ndvi_cdr.ndvi_cdr.analysis.template_config import (
    NoaaNdviCdrAnalysisTemplateConfig,
    NoaaNdviCdrDataVar,
)


def test_source_file_coord_get_url() -> None:
    coord = NoaaNdviCdrAnalysisSourceFileCoord(
        time=pd.Timestamp("2000-01-01"),
        url="https://noaa-cdr-ndvi-pds.s3.amazonaws.com/data/1990/AVHRR-Land_v005_AVH13C1_NOAA-11_19900107_c20170614232721.nc",
    )
    assert (
        coord.get_url()
        == "https://noaa-cdr-ndvi-pds.s3.amazonaws.com/data/1990/AVHRR-Land_v005_AVH13C1_NOAA-11_19900107_c20170614232721.nc"
    )


def test_source_file_coord_out_loc() -> None:
    """Test that out_loc returns the correct time coordinate mapping."""
    time = pd.Timestamp("2000-01-01")
    coord = NoaaNdviCdrAnalysisSourceFileCoord(
        time=time,
        url="https://noaa-cdr-ndvi-pds.s3.amazonaws.com/data/2000/test_file.nc",
    )

    out_loc = coord.out_loc()
    assert out_loc == {"time": time}


def test_region_job_generate_source_file_coords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the generation of source file coordinates with mocked S3 filesystem across multiple years."""
    # Mock the filesystem and its ls method
    mock_fs = Mock()
    # Mock fsspec.filesystem to return our mock filesystem
    monkeypatch.setattr("fsspec.filesystem", lambda *args, **kwargs: mock_fs)

    # Mock available files for multiple years
    def mock_ls(path: str) -> list[str]:
        if path == "noaa-cdr-ndvi-pds/data/1999":
            return [
                "noaa-cdr-ndvi-pds/data/1999/AVHRR-Land_v005_AVH13C1_NOAA-14_19991231_c20170614232721.nc",
            ]
        elif path == "noaa-cdr-ndvi-pds/data/2000":
            return [
                "noaa-cdr-ndvi-pds/data/2000/AVHRR-Land_v005_AVH13C1_NOAA-15_20000101_c20170614232721.nc",
                "noaa-cdr-ndvi-pds/data/2000/AVHRR-Land_v005_AVH13C1_NOAA-15_20000102_c20170614232721.nc",
            ]
        return []

    mock_fs.ls.side_effect = mock_ls

    template_config = NoaaNdviCdrAnalysisTemplateConfig()
    # Create a template dataset with time coordinates spanning multiple years
    template_ds = xr.Dataset(
        coords={
            "time": pd.date_range("1999-12-31", "2000-01-02", freq="D"),
            "latitude": np.linspace(89.999998472637188, -89.999998472637188, 3600),
            "longitude": np.linspace(-180.000006104363450, 179.999993895636550, 7200),
        }
    )

    region_job = NoaaNdviCdrAnalysisRegionJob.model_construct(
        final_store=get_zarr_store("fake-prod-path", "test-dataset", "test-version"),
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 10),
        reformat_job_name="test",
    )

    # Create a processing region dataset spanning multiple years
    processing_region_ds = xr.Dataset(
        coords={
            "time": pd.date_range("1999-12-31", "2000-01-02", freq="D"),
            "latitude": np.linspace(89.999998472637188, -89.999998472637188, 10),
            "longitude": np.linspace(-180.000006104363450, 179.999993895636550, 7200),
        }
    )

    source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars
    )

    # Verify we got the expected number of source file coordinates
    assert len(source_file_coords) == 3

    # Expected URLs based on the mocked file list spanning multiple years
    expected_urls = [
        "https://noaa-cdr-ndvi-pds.s3.amazonaws.com/data/1999/AVHRR-Land_v005_AVH13C1_NOAA-14_19991231_c20170614232721.nc",
        "https://noaa-cdr-ndvi-pds.s3.amazonaws.com/data/2000/AVHRR-Land_v005_AVH13C1_NOAA-15_20000101_c20170614232721.nc",
        "https://noaa-cdr-ndvi-pds.s3.amazonaws.com/data/2000/AVHRR-Land_v005_AVH13C1_NOAA-15_20000102_c20170614232721.nc",
    ]
    expected_times = [
        pd.Timestamp("1999-12-31"),
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("2000-01-02"),
    ]

    # Verify the coordinates have the correct structure and exact URLs
    for i, coord in enumerate(source_file_coords):
        assert isinstance(coord, NoaaNdviCdrAnalysisSourceFileCoord)
        assert coord.time == expected_times[i]
        assert coord.get_url() == expected_urls[i]

    # Verify filesystem was called twice (once for each year: 1999 and 2000)
    assert mock_fs.ls.call_count == 2


def test_region_job_generate_source_file_coords_file_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test error handling when no matching file is found for a given time."""
    # Mock the filesystem
    mock_fs = Mock()
    monkeypatch.setattr("fsspec.filesystem", lambda *args, **kwargs: mock_fs)

    # Mock available files that don't match our requested time
    mock_fs.ls.return_value = [
        "noaa-cdr-ndvi-pds/data/2000/AVHRR-Land_v005_AVH13C1_NOAA-15_20000105_c20170614232721.nc",
    ]

    template_config = NoaaNdviCdrAnalysisTemplateConfig()
    template_ds = xr.Dataset(
        coords={
            "time": pd.date_range("2000-01-01", "2000-01-01", freq="D"),
            "latitude": np.linspace(89.999998472637188, -89.999998472637188, 3600),
            "longitude": np.linspace(-180.000006104363450, 179.999993895636550, 7200),
        }
    )

    region_job = NoaaNdviCdrAnalysisRegionJob.model_construct(
        final_store=get_zarr_store("fake-prod-path", "test-dataset", "test-version"),
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 10),
        reformat_job_name="test",
    )

    processing_region_ds = xr.Dataset(
        coords={
            "time": pd.date_range("2000-01-01", "2000-01-01", freq="D"),
            "latitude": np.linspace(89.999998472637188, -89.999998472637188, 10),
            "longitude": np.linspace(-180.000006104363450, 179.999993895636550, 7200),
        }
    )

    source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars[:1]
    )
    assert source_file_coords[0].get_url() == "no-url"


def test_read_usable_ndvi_avhrr_era(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _read_usable_ndvi applies AVHRR masking correctly for 2013 data."""
    template_config = NoaaNdviCdrAnalysisTemplateConfig()

    region_job = NoaaNdviCdrAnalysisRegionJob.model_construct(
        final_store=get_zarr_store("fake-prod-path", "test-dataset", "test-version"),
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 10),
        reformat_job_name="test",
    )

    coord = NoaaNdviCdrAnalysisSourceFileCoord(
        time=pd.Timestamp("2013-12-31"),  # AVHRR era
        url="test_url",
    )

    # Same NDVI data for both tests
    ndvi_data = np.array(
        [[0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.3, 0.4]], dtype=np.float32
    )

    # QA values that mean different things in each system
    qa_data = np.array(
        [
            [64, 2, 72],  # AVHRR: night (preserved), cloudy (bad), water+night (bad)
            [0, 64, 2],  # AVHRR: good, night (preserved), cloudy (bad)
            [72, 0, 0],  # AVHRR: water+night (bad), good, good
        ],
        dtype=np.int16,
    )

    # Mock the netcdf data reading to return our test data
    def mock_read_netcdf_data(
        coord: NoaaNdviCdrAnalysisSourceFileCoord,
        data_var: NoaaNdviCdrDataVar,
    ) -> ArrayFloat32 | ArrayInt16:
        if data_var.internal_attrs.netcdf_var_name == "NDVI":
            return ndvi_data
        else:
            return qa_data

    monkeypatch.setattr(region_job, "_read_netcdf_data", mock_read_netcdf_data)

    result = region_job._read_usable_ndvi(
        coord, template_config.data_vars[1]
    )  # ndvi_usable

    # AVHRR behavior: 64=night (preserved), 2=cloudy (bad), 72=water+night (bad)
    assert result[0, 0] == 0.5  # night flag preserved
    assert np.isnan(result[0, 1])  # cloudy masked
    assert np.isnan(result[0, 2])  # water+night masked

    assert result[1, 0] == 0.8  # good pixel
    assert result[1, 1] == 0.9  # night flag preserved
    assert np.isnan(result[1, 2])  # cloudy masked

    assert np.isnan(result[2, 0])  # water+night masked
    assert result[2, 1] == 0.3  # good pixel
    assert result[2, 2] == 0.4  # good pixel


def test_read_usable_ndvi_viirs_era(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _read_usable_ndvi applies VIIRS masking correctly for 2014 data."""
    template_config = NoaaNdviCdrAnalysisTemplateConfig()

    region_job = NoaaNdviCdrAnalysisRegionJob.model_construct(
        final_store=get_zarr_store("fake-prod-path", "test-dataset", "test-version"),
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars,
        append_dim=template_config.append_dim,
        region=slice(0, 10),
        reformat_job_name="test",
    )

    coord = NoaaNdviCdrAnalysisSourceFileCoord(
        time=pd.Timestamp("2014-01-01"),  # VIIRS era
        url="test_url",
    )

    # Same NDVI data as AVHRR test
    ndvi_data = np.array(
        [[0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.3, 0.4]], dtype=np.float32
    )

    # Same QA values that mean different things in each system
    qa_data = np.array(
        [
            # aerosol_quality_ok (good), probably_cloudy (bad), land_no_desert+aerosol (good)
            [64, 2, 72],
            # no aerosol (bad), aerosol_quality_ok (good), probably_cloudy (bad)
            [0, 64, 2],
            # land_no_desert+aerosol (good), no aerosol (bad), no aerosol (bad)
            [72, 0, 0],
        ],
        dtype=np.int16,
    )

    # Mock the netcdf data reading to return our test data
    def mock_read_netcdf_data(
        coord: NoaaNdviCdrAnalysisSourceFileCoord,
        data_var: NoaaNdviCdrDataVar,
    ) -> ArrayFloat32 | ArrayInt16:
        if data_var.internal_attrs.netcdf_var_name == "NDVI":
            return ndvi_data
        else:
            return qa_data

    monkeypatch.setattr(region_job, "_read_netcdf_data", mock_read_netcdf_data)

    result = region_job._read_usable_ndvi(
        coord, template_config.data_vars[1]
    )  # ndvi_usable

    # VIIRS behavior: 64=aerosol_quality_ok (good), 2=probably_cloudy (bad), 72=land_no_desert+aerosol (good)
    assert result[0, 0] == 0.5  # aerosol quality preserved
    assert np.isnan(result[0, 1])  # cloudy masked
    assert result[0, 2] == 0.7  # land_no_desert+aerosol preserved

    assert np.isnan(result[1, 0])  # no aerosol quality masked
    assert result[1, 1] == 0.9  # aerosol quality preserved
    assert np.isnan(result[1, 2])  # cloudy masked

    assert result[2, 0] == 0.2  # land_no_desert+aerosol preserved
    assert np.isnan(result[2, 1])  # no aerosol quality masked
    assert np.isnan(result[2, 2])  # no aerosol quality masked
