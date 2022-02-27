import imp
from typing import Dict
import functools

from rawrelight import LightMap
from image_dataset.dataset_loaders.CEL import CELImage, CELDatasetLoader, cel_filters

INPUT_EXPOSURES = [0.1]
OUTPUT_EXPOSURES = [10]

TRAIN_JSON: str = "./dataset/train.JSON"


FILTER_SCENARIOS =  [2001,2002,2003,2004,2005]


def Run():
    inputExposuresFilter = functools.partial(cel_filters.FilterExactInList, INPUT_EXPOSURES)
    truthExposuresFilter = functools.partial(cel_filters.FilterExactInList, OUTPUT_EXPOSURES)

    filterScenarios = functools.partial(cel_filters.FilterExactScenarios,FILTER_SCENARIOS)

    inputFilters = functools.partial(cel_filters.Chain,[inputExposuresFilter,filterScenarios])
    truthFilters = functools.partial(cel_filters.Chain,[truthExposuresFilter,filterScenarios])

    datasetLoader = CELDatasetLoader(TRAIN_JSON,inputFilters,truthFilters)
    sets = datasetLoader.GetSet()
    pass



if __name__ == "__main__":
    Run()