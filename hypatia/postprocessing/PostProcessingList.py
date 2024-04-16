from hypatia.postprocessing.DefaultPostProcessing import DefaultPostProcessing
from hypatia.postprocessing.AggregatedPostProcessing import AggregatedPostProcessing

POSTPROCESSING_MODULES = {
    "default": DefaultPostProcessing,
    "aggregated": AggregatedPostProcessing,
}
