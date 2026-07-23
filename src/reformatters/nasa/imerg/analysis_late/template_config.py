from reformatters.nasa.imerg.imerg_config_models import ImergRun
from reformatters.nasa.imerg.template_config import NasaImergAnalysisTemplateConfig


class NasaImergAnalysisLateTemplateConfig(NasaImergAnalysisTemplateConfig):
    run: ImergRun = "late"
