from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from reformatters.common.config_models import (
    ROOT,
    BaseInternalAttrs,
    DatasetAttributes,
    DataVar,
    DataVarAttrs,
    Encoding,
    Group,
    codecs_to_dicts,
    var_path,
)


class TestGroupAndVarPath:
    def test_root_var_path_is_bare_name(self) -> None:
        assert var_path(ROOT, "temperature_2m") == "temperature_2m"

    def test_vertical_group_var_path_is_group_slash_name(self) -> None:
        assert var_path("pressure_level", "temperature") == "pressure_level/temperature"

    def _data_var(self, group: Group = ROOT) -> DataVar:  # type: ignore[type-arg]
        attrs = DataVarAttrs(
            long_name="Temperature", short_name="t", units="K", step_type="instant"
        )
        encoding = Encoding(dtype="float32", chunks=(1,), shards=None, fill_value=0.0)
        internal_attrs = BaseInternalAttrs(keep_mantissa_bits="no-rounding")
        return DataVar(
            name="temperature",
            group=group,
            encoding=encoding,
            attrs=attrs,
            internal_attrs=internal_attrs,
        )

    def test_datavar_defaults_to_root(self) -> None:
        var = self._data_var()
        assert var.group is ROOT
        assert var.path == "temperature"

    def test_datavar_in_vertical_group(self) -> None:
        var = self._data_var(group="pressure_level")
        assert var.path == "pressure_level/temperature"

    def test_datavar_rejects_unknown_group(self) -> None:
        with pytest.raises(ValidationError):
            self._data_var(group="height_above_ground")  # ty: ignore[invalid-argument-type]


class TestCodecsToDicts:
    def test_none_input(self) -> None:
        assert codecs_to_dicts(None) is None

    def test_empty_list(self) -> None:
        assert codecs_to_dicts([]) == []

    def test_codec_with_to_dict(self) -> None:
        codec = MagicMock()
        codec.to_dict.return_value = {"name": "zstd", "level": 3}
        result = codecs_to_dicts([codec])
        assert result == [{"name": "zstd", "level": 3}]

    def test_codec_without_to_dict(self) -> None:
        codec = MagicMock(spec=[])
        codec.__dict__ = {"name": "delta", "dtype": "int32"}
        result = codecs_to_dicts([codec])
        assert result == [{"name": "delta", "dtype": "int32"}]


class TestEncodingValidation:
    def test_valid_encoding(self) -> None:
        enc = Encoding(
            dtype="float32",
            chunks=(10, 10),
            shards=(20, 20),
            fill_value=0.0,
        )
        assert enc.dtype == "float32"

    def test_shards_must_be_multiple_of_chunks(self) -> None:
        with pytest.raises(ValidationError, match="multiple"):
            Encoding(
                dtype="float32",
                chunks=(10, 10),
                shards=(15, 20),
                fill_value=0.0,
            )

    def test_shards_none_is_valid(self) -> None:
        enc = Encoding(
            dtype="float32",
            chunks=(10,),
            shards=None,
            fill_value=0.0,
        )
        assert enc.shards is None

    def test_int_chunks_and_shards(self) -> None:
        enc = Encoding(
            dtype="int16",
            chunks=5,
            shards=10,
            fill_value=-1,
        )
        assert enc.chunks == 5
        assert enc.shards == 10

    def test_int_shards_not_multiple_raises(self) -> None:
        with pytest.raises(ValidationError, match="multiple"):
            Encoding(
                dtype="int16",
                chunks=5,
                shards=7,
                fill_value=-1,
            )

    def test_serializer_defaults_to_none(self) -> None:
        enc = Encoding(dtype="float64", chunks=(1, 1), shards=None, fill_value=0.0)
        assert enc.serializer is None

    def test_serializer_dict_is_retained(self) -> None:
        serializer = {"name": "gribberish", "configuration": {"var": "TMP"}}
        enc = Encoding(
            dtype="float64",
            chunks=(1, 1),
            shards=None,
            fill_value=0.0,
            serializer=serializer,
        )
        assert enc.serializer == serializer


class TestDatasetAttributes:
    def test_valid_attributes(self) -> None:
        attrs = DatasetAttributes(
            dataset_id="test-dataset",
            dataset_version="1.0",
            name="Test Dataset",
            description="A test dataset for unit tests.",
            attribution="Test attribution source.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain="2020-01-01 to present",
            time_resolution="1 hour",
        )
        assert attrs.dataset_id == "test-dataset"

    def test_invalid_dataset_id(self) -> None:
        with pytest.raises(ValidationError):
            DatasetAttributes(
                dataset_id="invalid id!",
                dataset_version="1.0",
                name="Test Dataset",
                description="A test dataset.",
                attribution="Test attribution.",
                license="CC-BY-4.0",
                spatial_domain="Global",
                spatial_resolution="0.25 degrees (~20km)",
                time_domain="2020-01-01 to present",
                time_resolution="1 hour",
            )


class TestDataVarAttrs:
    def test_valid_data_var_attrs(self) -> None:
        attrs = DataVarAttrs(
            long_name="Temperature",
            short_name="t",
            units="K",
            step_type="instant",
        )
        assert attrs.long_name == "Temperature"
        assert attrs.standard_name is None

    def test_empty_long_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            DataVarAttrs(
                long_name="",
                short_name="t",
                units="K",
                step_type="instant",
            )

    def test_invalid_step_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            DataVarAttrs(
                long_name="Temperature",
                short_name="t",
                units="K",
                step_type="invalid",  # ty: ignore[invalid-argument-type]
            )
