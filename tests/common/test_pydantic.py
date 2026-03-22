import pydantic
import pytest

from reformatters.common.pydantic import FrozenBaseModel, replace


class SampleModel(pydantic.BaseModel):
    name: str
    value: int


class FrozenModel(FrozenBaseModel):
    name: str
    value: int


class TestReplace:
    def test_replaces_field(self) -> None:
        original = SampleModel(name="a", value=1)
        updated = replace(original, value=2)
        assert updated.value == 2
        assert updated.name == "a"

    def test_returns_same_type(self) -> None:
        original = SampleModel(name="a", value=1)
        updated = replace(original, name="b")
        assert type(updated) is SampleModel

    def test_original_unchanged(self) -> None:
        original = SampleModel(name="a", value=1)
        replace(original, value=99)
        assert original.value == 1

    def test_validates_replacement_values(self) -> None:
        original = SampleModel(name="a", value=1)
        with pytest.raises(pydantic.ValidationError):
            replace(original, value="not_an_int")

    def test_replace_multiple_fields(self) -> None:
        original = SampleModel(name="a", value=1)
        updated = replace(original, name="b", value=2)
        assert updated.name == "b"
        assert updated.value == 2

    def test_works_with_frozen_model(self) -> None:
        original = FrozenModel(name="a", value=1)
        updated = replace(original, value=5)
        assert updated.value == 5
        assert type(updated) is FrozenModel


class TestFrozenBaseModel:
    def test_frozen(self) -> None:
        model = FrozenModel(name="a", value=1)
        with pytest.raises(pydantic.ValidationError):
            model.name = "b"

    def test_strict_mode_rejects_coercion(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            FrozenModel(name="a", value="1")  # ty: ignore[invalid-argument-type]

    def test_valid_construction(self) -> None:
        model = FrozenModel(name="test", value=42)
        assert model.name == "test"
        assert model.value == 42
