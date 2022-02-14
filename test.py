import os
import logging
import sys
from typing import Union, Tuple, List

import torch
from torchvision import transforms
import torch.nn
import torch.utils.data
from torch import optim
import functools
import numpy as np
import imageio
import exif

from util.model_wrapper import ModelWrapper
from image_dataset.dataset_loaders.CEL import (
    CELDataloaderFactory,
    cel_filters,
    CELImage,
)
import metric_handlers
from networks import CELNet
import util.common as common

# --- General Settings ---

IMAGE_BPS: int = 16
# can be a 2D tuple, make sure BOTH values are divisible by 16
# PATCH_SIZE = (2000,3008) # the maximum for our dataset. You'll probably need to use CPU for this, and around 40+ GB of RAM
PATCH_SIZE: Union[Tuple[int], int] = 512
# if GPU has insufficient memory (will result in crashes), switch DEVICE to "cpu"
#  make sure you have enoguh RAM available if you do, especially if you cache images
DEVICE: str = "cuda:0"

# Maximum memory allowed to be used in megabytes. Approx 80-60 gigabytes is ideal
# If you are running just one test (decided by number of items in TUNE_FACTORS), set this to 0.
IMAGE_CACHE_SIZE_MAX = 0

TRAIN_JSON: str = "./dataset/train.JSON"
TEST_JSON: str = "./dataset/test.JSON"
WEIGHTS_DIRECTORY: str = "./local/model.pt"
OUTPUT_DIRECTORY: str = "./output/"

TUNE_FACTORS = [0.5]

# Write an output image every SAVE_IMAGE_RATE input images
SAVE_IMAGE_RATE = 10

# --- Dataset Filtering ---
TEST_INPUT_EXPOSURE: List[float] = [0.1]
TEST_TRUTH_EXPOSURE: List[float] = [5]


def GetTestCallbacks(
    wrapper: ModelWrapper,
    PSNR: metric_handlers.PSNR,
    SSIM: metric_handlers.SSIM,
    imageNumberMetric: metric_handlers.Metric[int],
):
    def SaveMetrics(
        inputImage: torch.Tensor,
        gTruthImage: torch.Tensor,
        unetOutput: torch.Tensor,
        inputMeta: CELImage,
        gtruthMeta: CELImage,
        loss: float,
    ):

        imageNumber = imageNumberMetric.data.__len__()
        imageNumberMetric.Call(imageNumber)

        gtruthProcessed = gTruthImage[0].permute(1, 2, 0).cpu().data.numpy()
        unetOutputProcessed = unetOutput[0].permute(1, 2, 0).cpu().data.numpy()

        unetOutputProcessed = np.minimum(np.maximum(unetOutputProcessed, 0), 1)

        PSNR.Call(
            (gtruthProcessed * 255).astype("uint8"),
            (unetOutputProcessed * 255).astype("uint8"),
        )
        SSIM.Call(
            (gtruthProcessed * 255).astype("uint8"),
            (unetOutputProcessed * 255).astype("uint8"),
        )

    return SaveMetrics


def GetSaveImagesCallback(
    wrapper: ModelWrapper,
    directory: str,
    rate: int,
    prefix: str,
):

    imageIndex = [0]

    def Callback(
        inputImage: torch.Tensor,
        gTruthImage: torch.Tensor,
        unetOutput: torch.Tensor,
        inputMeta: List[CELImage],
        gtruthMeta: List[CELImage],
        loss: float,
    ):
        if (imageIndex[0] % rate) == 0:

            imname = (
                prefix
                + "_scenario_"
                + inputMeta[0].scenario.__str__()
                + "_CSVindex_"
                + imageIndex[0].__str__()
            )
            imdir = directory + "/" + imname + ".jpg"

            convertedImage = unetOutput[0].permute(1, 2, 0).cpu().data.numpy()
            convertedImage = np.minimum(np.maximum(convertedImage, 0), 1)

            convertedImage *= 255
            convertedImage = convertedImage.astype(np.uint8)

            focal_length = (gtruthMeta[0].focalLength,)
            f_number = gtruthMeta[0].f_number
            location = gtruthMeta[0].location

            imageio.imwrite(imdir, convertedImage, "jpg")

            # now read the file via exif and store EXIF data
            # doing this directly via imageio or cv2 proved to be troublesome
            image = exif.Image(imdir)

            image.focal_length = focal_length
            image.f_number = f_number
            image.user_comment = location

            # write out file
            file = open(imdir, mode="wb")
            file.write(image.get_file())

        imageIndex[0] += 1

    return Callback


def Run():

    exposureNormTransform = common.NormByExposureTime(IMAGE_BPS)

    testTransforms = transforms.Compose(
        [
            common.GetEvalTransforms(PATCH_SIZE, device=DEVICE),
            exposureNormTransform,
        ]
    )

    testInputFilter = functools.partial(
        cel_filters.FilterExactInList, TEST_INPUT_EXPOSURE
    )
    testTruthFilter = functools.partial(
        cel_filters.FilterExactInList, TEST_TRUTH_EXPOSURE
    )

    dataloaderFactory = CELDataloaderFactory(
        TRAIN_JSON,
        TEST_JSON,
        batch=1,
        cacheLimit=IMAGE_CACHE_SIZE_MAX,
    )

    testDataloader = dataloaderFactory.GetTest(
        testTransforms, testInputFilter, testTruthFilter
    )

    network = CELNet(adaptive=True)
    optimiser = optim.Adam(network.parameters(), lr=1e-4)
    wrapper = ModelWrapper(network, optimiser, torch.nn.L1Loss(), DEVICE)

    metricPSNR = metric_handlers.PSNR(name="PSNR", dataRange=255)
    metricSSIM = metric_handlers.SSIM(multichannel=True, name="SSIM", dataRange=255)
    tuneFactorMetric = metric_handlers.Metric[float](name="Tune factor")
    imageNumberMetric = metric_handlers.Metric[int](name="Image")

    csvFileDir: str = OUTPUT_DIRECTORY + "data.csv"

    metricsToCsv = metric_handlers.MetricsToCsv(
        csvFileDir, [imageNumberMetric, tuneFactorMetric, metricPSNR, metricSSIM]
    )

    wrapper.OnTestIter += GetTestCallbacks(
        wrapper, metricPSNR, metricSSIM, imageNumberMetric
    )

    if not os.path.exists(WEIGHTS_DIRECTORY):
        raise IOError("File " + WEIGHTS_DIRECTORY + " not found")
    else:
        wrapper.LoadWeights(
            WEIGHTS_DIRECTORY, strictWeightLoad=True, loadOptimiser=False
        )

    for factor in TUNE_FACTORS:

        tuneFactorLambda = lambda *args: tuneFactorMetric.Call(factor)
        saveImageCallback = GetSaveImagesCallback(
            wrapper, OUTPUT_DIRECTORY, SAVE_IMAGE_RATE, "factor_" + factor.__str__()
        )

        wrapper.OnTestIter += tuneFactorLambda
        wrapper.OnTestIter += saveImageCallback

        network.InterpolateAndLoadWeights(factor)
        wrapper.Test(testDataloader)

        wrapper.OnTestIter -= tuneFactorLambda

    metricsToCsv.Write()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    Run()
