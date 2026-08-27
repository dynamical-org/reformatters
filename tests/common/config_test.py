from reformatters.common.config import DynamicalConfig, Env


class TestEnv:
    def test_values(self) -> None:
        assert Env.dev == "dev"
        assert Env.prod == "prod"
        assert Env.test == "test"

    def test_from_string(self) -> None:
        assert Env("dev") is Env.dev
        assert Env("prod") is Env.prod
        assert Env("test") is Env.test


class TestDynamicalConfig:
    def test_is_test(self) -> None:
        config = DynamicalConfig(env=Env.test)
        assert config.is_test is True
        assert config.is_dev is False
        assert config.is_prod is False

    def test_is_dev(self) -> None:
        config = DynamicalConfig(env=Env.dev)
        assert config.is_dev is True
        assert config.is_test is False
        assert config.is_prod is False

    def test_is_prod(self) -> None:
        config = DynamicalConfig(env=Env.prod)
        assert config.is_prod is True
        assert config.is_dev is False
        assert config.is_test is False

    def test_sentry_enabled_when_dsn_set(self) -> None:
        config = DynamicalConfig(env=Env.dev, sentry_dsn="https://example.com/dsn")
        assert config.is_sentry_enabled is True

    def test_sentry_disabled_when_dsn_none(self) -> None:
        config = DynamicalConfig(env=Env.dev, sentry_dsn=None)
        assert config.is_sentry_enabled is False
