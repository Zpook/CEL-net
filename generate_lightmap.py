import torch

from typing import Dict
import functools

from lightmap import LightMap
from util.common import RawHandleBlackLevels, BayerUnpack, RAW_WHITE_LEVEL
from image_dataset.dataset_loaders.CEL import RawCELDatasetLoader, cel_filters

INPUT_EXPOSURES = [0.1]
OUTPUT_EXPOSURES = [1]

TRAIN_JSON: str = "./dataset/train.JSON"

MAX_RELIGHT_LEVEL = RAW_WHITE_LEVEL



def Run():
    inputExposuresFilter = functools.partial(cel_filters.FilterExactInList, INPUT_EXPOSURES)
    truthExposuresFilter = functools.partial(cel_filters.FilterExactInList, OUTPUT_EXPOSURES)

    datasetLoader = RawCELDatasetLoader(TRAIN_JSON,inputExposuresFilter,truthExposuresFilter)
    sets = datasetLoader.GetSet()

    lightmap = LightMap(MAX_RELIGHT_LEVEL,4,1)

    for set in sets:
        input, truth = set.GetPair()

        input = input.Load()
        truth=truth.Load()

        input = BayerUnpack(RawHandleBlackLevels(input)).transpose((2,0,1))
        truth = BayerUnpack(RawHandleBlackLevels(truth)).transpose((2,0,1))

        input = torch.tensor(input.astype("int32"))
        truth = torch.tensor(truth.astype("int32"))

        print("Processing " + set.trainList[0].path)

        lightmap.SampleImage(input,truth)

    lightmap.Save("./local/lightmap_0.1x1.bin")





if __name__ == "__main__":
    Run()