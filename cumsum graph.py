import torch
import numpy as np
from matplotlib import pyplot as plt

from typing import Dict
import functools

from lightmap import LightMap
from util.common import RawHandleBlackLevels, BayerUnpack, RAW_WHITE_LEVEL
from image_dataset.dataset_loaders.CEL import RawCELDatasetLoader, cel_filters

INPUT_EXPOSURES = [0.1]
OUTPUT_EXPOSURES = [1]

TRAIN_JSON: str = "./dataset/train.JSON"

TRAIN_JSON: str = "/mnt/5488CA8688CA6658/WORK/dataset/train.JSON"

MAX_RELIGHT_LEVEL = RAW_WHITE_LEVEL



def Run():
    inputExposuresFilter = functools.partial(cel_filters.Exposures_Whitelist, INPUT_EXPOSURES)
    truthExposuresFilter = functools.partial(cel_filters.Exposures_Whitelist, OUTPUT_EXPOSURES)

    datasetLoader = RawCELDatasetLoader(TRAIN_JSON,inputExposuresFilter,truthExposuresFilter)
    sets = datasetLoader.GetSet()

    avgHist = np.zeros(RAW_WHITE_LEVEL-1)
    imageIndex = 0
    valueRanges = range(RAW_WHITE_LEVEL)

    for set in sets:
        input, truth = set.GetPair()

        imageIndex += 1

        # meta = {}
        # scenario = input.scenario


        # meta["scenario"] = scenario
        # meta["in_exposure"] = input.exposure
        # meta["out_exposure"] = truth.exposure
        # meta["location"] = input.location
        # meta["in_iso"] = input.iso
        # meta["out_iso"] = input.iso


        input = input.Load()
        # truth=truth.Load()

        input = RawHandleBlackLevels(input)
        # truth = BayerUnpack(RawHandleBlackLevels(truth)).transpose((2,0,1))

        # input = torch.tensor(input.astype("int32"))
        imHist = np.histogram(input,bins=valueRanges)[0]
        avgHist = (avgHist * (imageIndex-1)/imageIndex) + imHist / imageIndex
        # truth = torch.tensor(truth.astype("int32"))

        print("Processing " + set.trainList[0].path)

    cumsum = np.cumsum(avgHist / (input.shape[0] * input.shape[1]))
    plt.plot(cumsum)
    plt.show()







if __name__ == "__main__":
    Run()