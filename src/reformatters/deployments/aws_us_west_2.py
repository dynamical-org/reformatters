from typing import Any

from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.operational import OperationalResources
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.contrib.nasa.smap.level3_36km_v9 import NasaSmapLevel336KmV9Dataset
from reformatters.contrib.noaa.ndvi_cdr.analysis import (
    NoaaNdviCdrAnalysisDataset,
)
from reformatters.contrib.uarizona.swann.analysis import UarizonaSwannAnalysisDataset
from reformatters.deployments.storage import (
    DwdIconEuIcechunkAwsOpenDataDatasetStorageConfig,
    EcccHrdpsIcechunkAwsOpenDataDatasetStorageConfig,
    EcmwfAifsEnsIcechunkAwsOpenDataDatasetStorageConfig,
    EcmwfAifsSingleIcechunkAwsOpenDataDatasetStorageConfig,
    EcmwfIfsEnsIcechunkAwsOpenDataDatasetStorageConfig,
    NasaImergIcechunkAwsOpenDataDatasetStorageConfig,
    NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig,
    NoaaGfsIcechunkAwsOpenDataDatasetStorageConfig,
    NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig,
    NoaaMrmsIcechunkAwsOpenDataDatasetStorageConfig,
    NoaaRefsIcechunkAwsOpenDataDatasetStorageConfig,
    NoaaRrfsEnsIcechunkAwsOpenDataDatasetStorageConfig,
    NoaaRrfsIcechunkAwsOpenDataDatasetStorageConfig,
    NoaaRtmaIcechunkAwsOpenDataDatasetStorageConfig,
    NoaaUrmaIcechunkAwsOpenDataDatasetStorageConfig,
    SourceCoopZarrDatasetStorageConfig,
    UcsbChcChirpsIcechunkAwsOpenDataDatasetStorageConfig,
    UpstreamGriddedZarrsDatasetStorageConfig,
    Weathernext2IcechunkDatasetStorageConfig,
    Weathernext3IcechunkDatasetStorageConfig,
)
from reformatters.dwd.icon_eu.forecast_5_day import DwdIconEuForecast5DayDataset
from reformatters.eccc.hrdps.forecast import EcccHrdpsForecastDataset
from reformatters.ecmwf.aifs_ens.forecast import (
    EcmwfAifsEnsForecastDataset,
)
from reformatters.ecmwf.aifs_ens.forecast_fast import EcmwfAifsEnsForecastFastDataset
from reformatters.ecmwf.aifs_single.forecast import (
    EcmwfAifsSingleForecastDataset,
)
from reformatters.ecmwf.aifs_single.forecast_virtual import (
    EcmwfAifsSingleForecastVirtualDataset,
)
from reformatters.ecmwf.archive_gribs.forecast_46_day_archiver import (
    EcmwfIfsEns46DayGribArchiver,
)
from reformatters.ecmwf.ifs_ens.forecast_0_25_degree_fast import (
    EcmwfIfsEnsForecast025DegreeFastDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_6_day_0_1_degree_virtual.dynamical_dataset import (
    EcmwfIfsEnsForecast6Day01DegreeVirtualDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_6_day_0_25_degree_virtual.dynamical_dataset import (
    EcmwfIfsEnsForecast6Day025DegreeVirtualDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_1_degree_virtual.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day01DegreeVirtualDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.deprecated_zarr_v3 import (
    EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3Dataset,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree_virtual.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeVirtualDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_1_5_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast46Day15DegreeDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_6_hourly_1_5_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
)
from reformatters.google.weathernext2.forecast_historical_virtual import (
    GoogleWeathernext2ForecastHistoricalVirtualDataset,
)
from reformatters.google.weathernext2.forecast_operational_virtual import (
    GoogleWeathernext2ForecastOperationalVirtualDataset,
)
from reformatters.google.weathernext3.forecast_15_day_0_05_degree_virtual import (
    GoogleWeathernext3Forecast15Day005DegreeVirtualDataset,
)
from reformatters.google.weathernext3.forecast_15_day_0_1_degree_virtual import (
    GoogleWeathernext3Forecast15Day01DegreeVirtualDataset,
)
from reformatters.google.weathernext3.forecast_48_hour_0_05_degree_virtual import (
    GoogleWeathernext3Forecast48Hour005DegreeVirtualDataset,
)
from reformatters.google.weathernext3.forecast_48_hour_0_1_degree_virtual import (
    GoogleWeathernext3Forecast48Hour01DegreeVirtualDataset,
)
from reformatters.nasa.imerg.analysis_early import NasaImergAnalysisEarlyDataset
from reformatters.nasa.imerg.analysis_late import NasaImergAnalysisLateDataset
from reformatters.noaa.gefs.analysis.dynamical_dataset import GefsAnalysisDataset
from reformatters.noaa.gefs.analysis_0_25_degree_virtual.dynamical_dataset import (
    NoaaGefsAnalysis025DegreeVirtualDataset,
)
from reformatters.noaa.gefs.forecast_10_day_0_25_degree_virtual.dynamical_dataset import (
    NoaaGefsForecast10Day025DegreeVirtualDataset,
)
from reformatters.noaa.gefs.forecast_16_day_0_5_degree_virtual.dynamical_dataset import (
    NoaaGefsForecast16Day05DegreeVirtualDataset,
)
from reformatters.noaa.gefs.forecast_35_day.dynamical_dataset import (
    GefsForecast35DayDataset,
)
from reformatters.noaa.gefs.forecast_35_day_0_5_degree_virtual.dynamical_dataset import (
    NoaaGefsForecast35Day05DegreeVirtualDataset,
)
from reformatters.noaa.gfs.analysis import NoaaGfsAnalysisDataset
from reformatters.noaa.gfs.analysis_virtual import NoaaGfsAnalysisVirtualDataset
from reformatters.noaa.gfs.forecast import NoaaGfsForecastDataset
from reformatters.noaa.gfs.forecast_virtual import NoaaGfsForecastVirtualDataset
from reformatters.noaa.hrrr.analysis.dynamical_dataset import (
    NoaaHrrrAnalysisDataset,
)
from reformatters.noaa.hrrr.analysis_virtual.dynamical_dataset import (
    NoaaHrrrAnalysisVirtualDataset,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast18HourVirtualDataset,
)
from reformatters.noaa.hrrr.forecast_48_hour.dynamical_dataset import (
    NoaaHrrrForecast48HourDataset,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast48HourVirtualDataset,
)
from reformatters.noaa.hrrr.nomads_mirror import NoaaHrrrNomadsMirror
from reformatters.noaa.mrms.conus_analysis_hourly.dynamical_dataset import (
    NoaaMrmsConusAnalysisHourlyDataset,
)
from reformatters.noaa.refs.forecast_3_hourly_virtual.dynamical_dataset import (
    NoaaRefsForecast3HourlyVirtualDataset,
)
from reformatters.noaa.refs.forecast_hourly_virtual.dynamical_dataset import (
    NoaaRefsForecastHourlyVirtualDataset,
)
from reformatters.noaa.rrfs.forecast_18_hour_virtual.dynamical_dataset import (
    NoaaRrfsForecast18HourVirtualDataset,
)
from reformatters.noaa.rrfs.forecast_84_hour_virtual.dynamical_dataset import (
    NoaaRrfsForecast84HourVirtualDataset,
)
from reformatters.noaa.rrfs.forecast_sub_hourly_virtual.dynamical_dataset import (
    NoaaRrfsForecastSubHourlyVirtualDataset,
)
from reformatters.noaa.rrfs_ens.forecast_virtual.dynamical_dataset import (
    NoaaRrfsEnsForecastVirtualDataset,
)
from reformatters.noaa.rtma.analysis_virtual import NoaaRtmaAnalysisVirtualDataset
from reformatters.noaa.rtma_ru.analysis_virtual import NoaaRtmaRuAnalysisVirtualDataset
from reformatters.noaa.urma.analysis_virtual import NoaaUrmaAnalysisVirtualDataset
from reformatters.ucsb_chc.chirps.analysis_final import (
    UcsbChcChirpsAnalysisFinalDataset,
)
from reformatters.ucsb_chc.chirps.analysis_preliminary import (
    UcsbChcChirpsAnalysisPreliminaryDataset,
)

# Registry of all DynamicalDatasets.
CLUSTER_DYNAMICAL_DATASETS: list[DynamicalDataset[Any, Any]] = [
    GoogleWeathernext3Forecast48Hour01DegreeVirtualDataset(
        primary_storage_config=Weathernext3IcechunkDatasetStorageConfig()
    ),
    GoogleWeathernext3Forecast15Day01DegreeVirtualDataset(
        primary_storage_config=Weathernext3IcechunkDatasetStorageConfig()
    ),
    GoogleWeathernext3Forecast15Day005DegreeVirtualDataset(
        primary_storage_config=Weathernext3IcechunkDatasetStorageConfig()
    ),
    GoogleWeathernext3Forecast48Hour005DegreeVirtualDataset(
        primary_storage_config=Weathernext3IcechunkDatasetStorageConfig()
    ),
    # NOAA
    NoaaGfsForecastDataset(
        primary_storage_config=NoaaGfsIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    NoaaGfsAnalysisDataset(
        primary_storage_config=NoaaGfsIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    NoaaGfsAnalysisVirtualDataset(
        primary_storage_config=NoaaGfsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaGfsForecastVirtualDataset(
        primary_storage_config=NoaaGfsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    GefsAnalysisDataset(
        primary_storage_config=NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    GefsForecast35DayDataset(
        primary_storage_config=NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    NoaaGefsAnalysis025DegreeVirtualDataset(
        primary_storage_config=NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaGefsForecast10Day025DegreeVirtualDataset(
        primary_storage_config=NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaGefsForecast16Day05DegreeVirtualDataset(
        primary_storage_config=NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaGefsForecast35Day05DegreeVirtualDataset(
        primary_storage_config=NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaHrrrForecast48HourDataset(
        primary_storage_config=NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    NoaaHrrrAnalysisDataset(
        primary_storage_config=NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    NoaaHrrrAnalysisVirtualDataset(
        primary_storage_config=NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaRtmaAnalysisVirtualDataset(
        primary_storage_config=NoaaRtmaIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaRtmaRuAnalysisVirtualDataset(
        primary_storage_config=NoaaRtmaIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaUrmaAnalysisVirtualDataset(
        primary_storage_config=NoaaUrmaIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaHrrrForecast48HourVirtualDataset(
        primary_storage_config=NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaHrrrForecast18HourVirtualDataset(
        primary_storage_config=NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaRrfsForecast84HourVirtualDataset(
        primary_storage_config=NoaaRrfsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaRrfsForecast18HourVirtualDataset(
        primary_storage_config=NoaaRrfsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaRrfsForecastSubHourlyVirtualDataset(
        primary_storage_config=NoaaRrfsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaRrfsEnsForecastVirtualDataset(
        primary_storage_config=NoaaRrfsEnsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaRefsForecastHourlyVirtualDataset(
        primary_storage_config=NoaaRefsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaRefsForecast3HourlyVirtualDataset(
        primary_storage_config=NoaaRefsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaMrmsConusAnalysisHourlyDataset(
        primary_storage_config=NoaaMrmsIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    # ECMWF
    EcmwfIfsEnsForecast6Day025DegreeVirtualDataset(
        primary_storage_config=EcmwfIfsEnsIcechunkAwsOpenDataDatasetStorageConfig()
    ),
    EcmwfIfsEnsForecast6Day01DegreeVirtualDataset(
        primary_storage_config=EcmwfIfsEnsIcechunkAwsOpenDataDatasetStorageConfig()
    ),
    EcmwfIfsEnsForecast15Day025DegreeVirtualDataset(
        primary_storage_config=EcmwfIfsEnsIcechunkAwsOpenDataDatasetStorageConfig()
    ),
    EcmwfIfsEnsForecast15Day01DegreeVirtualDataset(
        primary_storage_config=EcmwfIfsEnsIcechunkAwsOpenDataDatasetStorageConfig()
    ),
    EcmwfIfsEnsForecast15Day025DegreeDataset(
        primary_storage_config=EcmwfIfsEnsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3Dataset(
        primary_storage_config=SourceCoopZarrDatasetStorageConfig(),
    ),
    EcmwfIfsEnsForecast46Day15DegreeDataset(
        primary_storage_config=EcmwfIfsEnsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset(
        primary_storage_config=EcmwfIfsEnsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    EcmwfAifsSingleForecastDataset(
        primary_storage_config=EcmwfAifsSingleIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    EcmwfAifsSingleForecastVirtualDataset(
        primary_storage_config=EcmwfAifsSingleIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    EcmwfAifsEnsForecastDataset(
        primary_storage_config=EcmwfAifsEnsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    EcmwfIfsEnsForecast025DegreeFastDataset(
        primary_storage_config=StorageConfig(
            base_path="s3://dynamical-ecmwf-ifs-ens",
            k8s_secret_name="dynamical-ecmwf-output-r2-storage-options-key",  # noqa: S106
            format=DatasetFormat.ICECHUNK,
        ),
    ),
    EcmwfAifsEnsForecastFastDataset(
        primary_storage_config=StorageConfig(
            base_path="s3://dynamical-ecmwf-aifs-ens",
            k8s_secret_name="dynamical-ecmwf-output-r2-storage-options-key",  # noqa: S106
            format=DatasetFormat.ICECHUNK,
        ),
    ),
    # DWD
    DwdIconEuForecast5DayDataset(
        primary_storage_config=DwdIconEuIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    # ECCC
    EcccHrdpsForecastDataset(
        primary_storage_config=EcccHrdpsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    # Google
    GoogleWeathernext2ForecastHistoricalVirtualDataset(
        primary_storage_config=Weathernext2IcechunkDatasetStorageConfig(),
    ),
    GoogleWeathernext2ForecastOperationalVirtualDataset(
        primary_storage_config=Weathernext2IcechunkDatasetStorageConfig(),
    ),
    # NASA
    NasaImergAnalysisEarlyDataset(
        primary_storage_config=NasaImergIcechunkAwsOpenDataDatasetStorageConfig()
    ),
    NasaImergAnalysisLateDataset(
        primary_storage_config=NasaImergIcechunkAwsOpenDataDatasetStorageConfig()
    ),
    # UCSB CHC
    UcsbChcChirpsAnalysisFinalDataset(
        primary_storage_config=UcsbChcChirpsIcechunkAwsOpenDataDatasetStorageConfig()
    ),
    UcsbChcChirpsAnalysisPreliminaryDataset(
        primary_storage_config=UcsbChcChirpsIcechunkAwsOpenDataDatasetStorageConfig()
    ),
    # Contrib
    UarizonaSwannAnalysisDataset(
        primary_storage_config=UpstreamGriddedZarrsDatasetStorageConfig()
    ),
    NoaaNdviCdrAnalysisDataset(
        primary_storage_config=UpstreamGriddedZarrsDatasetStorageConfig()
    ),
    NasaSmapLevel336KmV9Dataset(
        primary_storage_config=UpstreamGriddedZarrsDatasetStorageConfig()
    ),
]


# Source archives that feed a dataset but have no store of their own. They deploy
# their own cronjobs and are not datasets, so they carry no update/validate/backfill.
CLUSTER_OPERATIONAL_ARCHIVERS: list[OperationalResources] = [
    EcmwfIfsEns46DayGribArchiver(),
    NoaaHrrrNomadsMirror(),
]
