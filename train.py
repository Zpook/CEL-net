import logging
import sys
import os
from typing import Union, Tuple, List

import torch
import torch.nn
import torch.utils.data
from torch import optim
import functools
from torchvision import transforms

import util.common as common
from util.model_wrapper import ModelWrapper
from networks import CELNet
from image_dataset.dataset_loaders.CEL import CELDataloaderFactory, cel_filters

# --- General Settings ---

IMAGE_BPS: int = 16
# can be a 2D tuple, make sure BOTH values are divisible by 16
PATCH_SIZE: Union[Tuple[int], int] = 512

MODEL_DEVICE: str = "cuda:0"

# Consider moving relight to GPU or increasing worker count on larger patches
# There is a considerable slow-down in performance if both are applied on smaller patches
RELIGHT_DEVICE: str = "cpu"
RELIGHT_WORKER_COUNT: int = 1

# fiddle with these if training seems oddly slow
# TODO Worker count does nothing, either remove this or fix the cuda thread bug
DATASET_WORKER_COUNT: int = 0
BATCH_COUNT = 2

# Maximum RAM allowed to be used in megabytes. Approx 80-60 gigabytes is optimal
IMAGE_CACHE_SIZE_MAX = 10000

TRAIN_JSON: str = "./dataset/train.JSON"
TEST_JSON: str = "./dataset/test.JSON"

# ! REMOVE !
TRAIN_JSON: str = "/media/mikel/New040Volume/WORK/dataset/train.JSON"
TEST_JSON: str = "/media/mikel/New040Volume/WORK/dataset/test.JSON"


WEIGHTS_DIRECTORY: str = "./local/model.pt"

EPOCHS_TRAIN = {
    1: 400,
    2: 800,
    3: 1000,
}

LR_TRAIN = {1: 1e-4, 2: 0.5 * 1e-4, 3: 0.25 * 1e-4}

EPOCHS_TUNE = {
    1: 1400,
    2: 1700,
}

LR_TUNE = {1: 1e-4, 2: 0.25 * 1e-4}

# --- Dataset Filtering ---

# train inputs
TRAIN_INPUT_EXPOSURE: List[float] = [0.1]
TRAIN_TRUTH_EXPOSURE: List[float] = [1]

# tune inputs
TUNE_INPUT_EXPOSURE: List[float] = [0.1]
TUNE_TRUTH_EXPOSURE: List[float] = [10]

# whitelisting scenarios will use ONLY selected scenarios, useful for overfitting
WHITELIST_SCENARIOS = []
BLACKLIST_SCENARIOS = []


def Run():

    # construct image transformations

    lightmapDict = common.GetLightmaps(RELIGHT_DEVICE, RELIGHT_WORKER_COUNT)
    exposureNormTransform = common.NormByRelight_Local(lightmapDict, IMAGE_BPS)

    trainTransforms = transforms.Compose(
        [
            common.GetTrainTransforms(
                IMAGE_BPS, PATCH_SIZE, normalize=False, device=MODEL_DEVICE
            ),
            exposureNormTransform,
        ]
    )

    # construct filters to sort database

    train_input_filter = functools.partial(
        cel_filters.Exposures_Whitelist, TRAIN_INPUT_EXPOSURE
    )
    train_truth_filter = functools.partial(
        cel_filters.Exposures_Whitelist, TRAIN_TRUTH_EXPOSURE
    )

    tune_input_filter = functools.partial(
        cel_filters.Exposures_Whitelist, TUNE_INPUT_EXPOSURE
    )
    tune_truth_filter = functools.partial(
        cel_filters.Exposures_Whitelist, TUNE_TRUTH_EXPOSURE
    )

    if WHITELIST_SCENARIOS.__len__()!= 0:
        whitelist_filter = functools.partial(cel_filters.Scenario_Whitelist, WHITELIST_SCENARIOS)

        train_input_filter = functools.partial(cel_filters.Chain, [train_input_filter,whitelist_filter])
        train_truth_filter = functools.partial(cel_filters.Chain, [train_truth_filter,whitelist_filter])
        
        tune_input_filter = functools.partial(cel_filters.Chain, [tune_input_filter,whitelist_filter])
        tune_truth_filter = functools.partial(cel_filters.Chain, [tune_truth_filter,whitelist_filter])

    if BLACKLIST_SCENARIOS.__len__()!= 0:
        blacklist_filter = functools.partial(cel_filters.Scenario_Whitelist, BLACKLIST_SCENARIOS)

        train_input_filter = functools.partial(cel_filters.Chain, [train_input_filter,blacklist_filter])
        train_truth_filter = functools.partial(cel_filters.Chain, [train_truth_filter,blacklist_filter])
        
        tune_input_filter = functools.partial(cel_filters.Chain, [tune_input_filter,blacklist_filter])
        tune_truth_filter = functools.partial(cel_filters.Chain, [tune_truth_filter,blacklist_filter])


    dataloaderFactory = CELDataloaderFactory(
        TRAIN_JSON,
        TEST_JSON,
        patchSize=PATCH_SIZE,
        datasetWorkers=DATASET_WORKER_COUNT,
        batch=BATCH_COUNT,
        cacheLimit=IMAGE_CACHE_SIZE_MAX,
    )

    network = CELNet(adaptive=False)
    optimiser = optim.Adam(network.parameters(), lr=LR_TRAIN[0])
    wrapper = ModelWrapper(network, optimiser, torch.nn.L1Loss(), MODEL_DEVICE)

    if not os.path.exists(WEIGHTS_DIRECTORY):
        network._initialize_weights()
        wrapper.metaDict["model_tune_state"] = False
        wrapper.Save(WEIGHTS_DIRECTORY)

    checkpoint = torch.load(WEIGHTS_DIRECTORY)
    model_tune_flag = checkpoint["META"]["model_tune_state"]
    del checkpoint

    if not model_tune_flag:

        wrapper.OnTrainEpoch += lambda *args: wrapper.Save(WEIGHTS_DIRECTORY)

        wrapper.LoadWeights(WEIGHTS_DIRECTORY, strictWeightLoad=True)

        trainDataloader = dataloaderFactory.GetTrain(
            trainTransforms, train_input_filter, train_truth_filter
        )

        wrapper.Train(
            trainDataloader, trainToEpoch=EPOCHS_TRAIN[0], learningRate=LR_TRAIN[0]
        )
        wrapper.Train(
            trainDataloader, trainToEpoch=EPOCHS_TRAIN[1], learningRate=LR_TRAIN[1]
        )
        wrapper.Train(
            trainDataloader, trainToEpoch=EPOCHS_TRAIN[2], learningRate=LR_TRAIN[2]
        )

        # free up memory
        del trainDataloader

        wrapper.metaDict["model_tune_state"] = True
        wrapper.Save(WEIGHTS_DIRECTORY)

    # tuning starts here, rebuild everything
    tuneDataloader = dataloaderFactory.GetTrain(
        trainTransforms, tune_input_filter, tune_truth_filter
    )

    network = CELNet(adaptive=True)
    optimParams = network.TuningMode()

    optimiser = optim.Adam(optimParams, lr=LR_TUNE[0])
    wrapper = ModelWrapper(network, optimiser, torch.nn.L1Loss(), MODEL_DEVICE)
    wrapper.LoadWeights(WEIGHTS_DIRECTORY, loadOptimiser=False, strictWeightLoad=True)

    wrapper.OnTrainEpoch += lambda *args: wrapper.Save(WEIGHTS_DIRECTORY)

    wrapper.Train(tuneDataloader, trainToEpoch=EPOCHS_TUNE[0], learningRate=LR_TUNE[0])
    wrapper.Train(tuneDataloader, trainToEpoch=EPOCHS_TUNE[1], learningRate=LR_TUNE[1])


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    torch.multiprocessing.set_start_method("spawn")
    Run()
