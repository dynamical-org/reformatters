from reformatters.ecmwf.fast.template_config import EcmwfFastTemplateConfig, FastModel


class EcmwfAifsEnsForecastFastTemplateConfig(EcmwfFastTemplateConfig):
    model: FastModel = "aifs"
