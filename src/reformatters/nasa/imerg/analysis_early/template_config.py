from reformatters.nasa.imerg.imerg_config_models import ImergRun
from reformatters.nasa.imerg.template_config import NasaImergAnalysisTemplateConfig


class NasaImergAnalysisEarlyTemplateConfig(NasaImergAnalysisTemplateConfig):
    run: ImergRun = "early"
