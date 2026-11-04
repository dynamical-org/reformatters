from reformatters.ecmwf.fast.template_config import EcmwfFastTemplateConfig, FastModel


class EcmwfIfsEnsForecast025DegreeFastTemplateConfig(EcmwfFastTemplateConfig):
    model: FastModel = "ifs"
