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
import imageio
import numpy as np
import metric_handlers

import util.common as common
from util.model_wrapper import ModelWrapper
from networks import CELNet
from image_dataset.dataset_loaders.CEL import (
    CELDataloaderFactory,
    cel_filters,
    CELImage,
)

# --- General Settings ---

TRUTH_IMAGE_BPS: int = 16
# can be a 2D tuple, make sure both values are divisible by 16
PATCH_SIZE: Union[Tuple[int], int] = 512

MODEL_DEVICE: str = "cuda:0"

# Consider moving relight to GPU or increasing worker count on larger patches
# There is a considerable slow-down in performance if both are applied on smaller patches
RELIGHT_DEVICE: str = "cpu"
RELIGHT_WORKER_COUNT: int = 1

# fiddle with these if training seems oddly slow
# ! Worker count is currently bugged !
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

VALIDATION_ENALBED: bool = True
VALIDATION_OUTPUT_DIRECTORY: str = "./output/validation/"
VALIDATION_RATE: int = 50
VALIDATION_DEVICE = "cpu"
VALIDATION_PATCH_SIZE: Union[Tuple[int], int] = (2000,3008)

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


class ValidationHandler:
    def __init__(
        self,wrapper: ModelWrapper, dataloader, validationRate: int, savedir
    ) -> None:
        self.wrapper = wrapper
        self.dataloader = dataloader
        self.validationRate = validationRate
        self.outdir = savedir

        self._currentDir = None

        self.metricPSNR = metric_handlers.PSNR(name="PSNR", dataRange=255)
        self.metricSSIM = metric_handlers.SSIM(multichannel=True, name="SSIM", dataRange=255)
        self.imageNumberMetric = metric_handlers.Metric[int](name="Image")


    def __call__(self,epochIndex: int):
        if (epochIndex % self.validationRate) != 0:
            return

        self._currentDir = self.outdir + epochIndex.__str__() + "/"

        self.metricPSNR.Flush()
        self.metricSSIM.Flush()
        self.imageNumberMetric.Flush()

        csvFileDir: str = self._currentDir + "data.csv"

        metricsToCsv = metric_handlers.MetricsToCsv(
            csvFileDir, [self.imageNumberMetric, self.metricPSNR, self.metricSSIM]
        )

        self.wrapper.OnTestIter += self.OnIterCallback
        self.wrapper.Test(self.dataloader)
        self.wrapper.OnTestIter -= self.OnIterCallback

        metricsToCsv.Write()


    def OnIterCallback(
        self,
        inputImage: torch.Tensor,
        gTruthImage: torch.Tensor,
        unetOutput: torch.Tensor,
        inputMeta: CELImage,
        gtruthMeta: CELImage,
        loss: float,
    ):

        gtruthProcessed = gTruthImage[0].permute(1, 2, 0).cpu().data.numpy()
        unetOutputProcessed = unetOutput[0].permute(1, 2, 0).cpu().data.numpy()

        unetOutputProcessed = np.minimum(np.maximum(unetOutputProcessed, 0), 1)

        self.imageNumberMetric.Call(inputMeta[0].scenario)
        self.metricPSNRPSNR.Call(
            (gtruthProcessed * 255).astype("uint8"),
            (unetOutputProcessed * 255).astype("uint8"),
        )
        self.metricPSNRSSIM.Call(
            (gtruthProcessed * 255).astype("uint8"),
            (unetOutputProcessed * 255).astype("uint8"),
        )

        imname = +"scenario_" + inputMeta[0].scenario.__str__()
        imdir = self._currentDir + imname + ".jpg"

        unetOutputProcessed *= 255
        unetOutputProcessed = unetOutputProcessed.astype(np.uint8)

        imageio.imwrite(imdir, unetOutputProcessed, "jpg")


def Run():

    # construct image transformations

    lightmapDict = common.GetLightmaps(RELIGHT_DEVICE, RELIGHT_WORKER_COUNT)
    exposureNormTransform = common.NormByRelight_Local(lightmapDict, TRUTH_IMAGE_BPS)

    trainTransforms = transforms.Compose(
        [
            common.GetTrainTransforms(
                TRUTH_IMAGE_BPS, PATCH_SIZE, normalize=False, device=MODEL_DEVICE
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

    if WHITELIST_SCENARIOS.__len__() != 0:
        whitelist_filter = functools.partial(
            cel_filters.Scenario_Whitelist, WHITELIST_SCENARIOS
        )

        train_input_filter = functools.partial(
            cel_filters.Chain, [train_input_filter, whitelist_filter]
        )
        train_truth_filter = functools.partial(
            cel_filters.Chain, [train_truth_filter, whitelist_filter]
        )

        tune_input_filter = functools.partial(
            cel_filters.Chain, [tune_input_filter, whitelist_filter]
        )
        tune_truth_filter = functools.partial(
            cel_filters.Chain, [tune_truth_filter, whitelist_filter]
        )

    if BLACKLIST_SCENARIOS.__len__() != 0:
        blacklist_filter = functools.partial(
            cel_filters.Scenario_Whitelist, BLACKLIST_SCENARIOS
        )

        train_input_filter = functools.partial(
            cel_filters.Chain, [train_input_filter, blacklist_filter]
        )
        train_truth_filter = functools.partial(
            cel_filters.Chain, [train_truth_filter, blacklist_filter]
        )

        tune_input_filter = functools.partial(
            cel_filters.Chain, [tune_input_filter, blacklist_filter]
        )
        tune_truth_filter = functools.partial(
            cel_filters.Chain, [tune_truth_filter, blacklist_filter]
        )

    dataloaderFactory = CELDataloaderFactory(
        TRAIN_JSON,
        TEST_JSON,
        patchSize=PATCH_SIZE,
        datasetWorkers=DATASET_WORKER_COUNT,
        batch=BATCH_COUNT,
        cacheLimit=IMAGE_CACHE_SIZE_MAX,
    )

    network = CELNet(adaptive=False)
    optimiser = optim.Adam(network.parameters(), lr=LR_TRAIN[1])
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
        if VALIDATION_ENALBED:
            validationWrapper = ModelWrapper(network, optimiser, torch.nn.L1Loss(), VALIDATION_DEVICE)
            validator = ValidationHandler(validationWrapper,dataloader,VALIDATION_RATE,VALIDATION_OUTPUT_DIRECTORY)
            wrapper.OnTrainEpoch += validator

        wrapper.LoadWeights(WEIGHTS_DIRECTORY, strictWeightLoad=True)

        dataloader = dataloaderFactory.GetTrain(
            trainTransforms, train_input_filter, train_truth_filter
        )

        wrapper.Train(
            dataloader, trainToEpoch=EPOCHS_TRAIN[1], learningRate=LR_TRAIN[1]
        )
        wrapper.Train(
            dataloader, trainToEpoch=EPOCHS_TRAIN[2], learningRate=LR_TRAIN[2]
        )
        wrapper.Train(
            dataloader, trainToEpoch=EPOCHS_TRAIN[3], learningRate=LR_TRAIN[3]
        )

        wrapper.metaDict["model_tune_state"] = True
        wrapper.Save(WEIGHTS_DIRECTORY)

    # tuning starts here, rebuild everything
    # TODO: this is blatant copy-past code
    dataloader = dataloaderFactory.GetTrain(
        trainTransforms, tune_input_filter, tune_truth_filter
    )

    network = CELNet(adaptive=True)
    optimParams = network.TuningMode()

    optimiser = optim.Adam(optimParams, lr=LR_TUNE[0])
    wrapper = ModelWrapper(network, optimiser, torch.nn.L1Loss(), MODEL_DEVICE)
    wrapper.LoadWeights(WEIGHTS_DIRECTORY, loadOptimiser=False, strictWeightLoad=True)

    wrapper.OnTrainEpoch += lambda *args: wrapper.Save(WEIGHTS_DIRECTORY)
    if VALIDATION_ENALBED:
        validationWrapper = ModelWrapper(network, optimiser, torch.nn.L1Loss(), VALIDATION_DEVICE)
        validator = ValidationHandler(validationWrapper,dataloader,VALIDATION_RATE,VALIDATION_OUTPUT_DIRECTORY)
        wrapper.OnTrainEpoch += validator

    wrapper.Train(dataloader, trainToEpoch=EPOCHS_TUNE[1], learningRate=LR_TUNE[1])
    wrapper.Train(dataloader, trainToEpoch=EPOCHS_TUNE[2], learningRate=LR_TUNE[2])


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    torch.multiprocessing.set_start_method("spawn")
    Run()
