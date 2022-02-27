import torch

from typing import Dict
import functools

from lightmap import LightMap
from util.common import RawHandleBlackLevels, BayerUnpack
from image_dataset.dataset_loaders.CEL import RawCELDatasetLoader, cel_filters

INPUT_EXPOSURES = [0.1]
OUTPUT_EXPOSURES = [10]

TRAIN_JSON: str = "./dataset/train.JSON"

TRAIN_JSON: str = "/media/mikel/New040Volume/WORK/dataset/train.JSON"


MAX_RELIGHT_LEVEL = 5000


FILTER_SCENARIOS =  [2001,2002,2003,2004,2005]


def Run():
    inputExposuresFilter = functools.partial(cel_filters.FilterExactInList, INPUT_EXPOSURES)
    truthExposuresFilter = functools.partial(cel_filters.FilterExactInList, OUTPUT_EXPOSURES)

    filterScenarios = functools.partial(cel_filters.FilterExactScenarios,FILTER_SCENARIOS)

    inputFilters = functools.partial(cel_filters.Chain,[inputExposuresFilter,filterScenarios])
    truthFilters = functools.partial(cel_filters.Chain,[truthExposuresFilter,filterScenarios])

    datasetLoader = RawCELDatasetLoader(TRAIN_JSON,inputFilters,truthFilters)
    sets = datasetLoader.GetSet()

    lightmap = LightMap(MAX_RELIGHT_LEVEL,4,-1)

    for set in sets:
        input, truth = set.GetPair()

        input = input.Load()
        truth=truth.Load()

        input = BayerUnpack(RawHandleBlackLevels(input)).transpose((2,0,1))
        truth = BayerUnpack(RawHandleBlackLevels(truth)).transpose((2,0,1))

        input = torch.tensor(input.astype("int32"))
        truth = torch.tensor(truth.astype("int32"))


        lightmap.SampleImage(input,truth)

    lightmap.Save("./local/lightmap.bin")





if __name__ == "__main__":
    Run()