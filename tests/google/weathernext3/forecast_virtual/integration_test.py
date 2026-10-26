from tests.google.weathernext3.forecast_virtual.datasets_test import DATASETS, job_for


def test_exact_source_paths() -> None:
    job = job_for(DATASETS[0], "2026-01-01T01:00")
    coord = job.source_file_coords()[0]
    assert (
        coord.get_url()
        == "gs://weathernext3_statistics_spatial/weathernext_3_0_0_statistics/zarr/2026_to_present/20260101_00hr_01_preds/predictions.zarr"
    )
    assert (
        coord.chunk_location(job.data_vars[0], "mean")
        == "https://wn.dynamical.org/chunks/weathernext_3_0_0_statistics/zarr/2026_to_present/20260101_00hr_01_preds/predictions.zarr/temperature_2m_mean/c/0/0/0"
    )
    (query,) = job._listing_queries(coord)
    assert (
        query.prefix
        == "weathernext_3_0_0_statistics/zarr/2026_to_present/20260101_00hr_01_preds/predictions.zarr/temperature_2m_"
    )
