import torch
import os
import numpy as np
import rawpy
import imageio
from rawpy._rawpy import RawPy

from typing import Dict
import functools

from lightmap import LightMap
import metric_handlers
from util.common import (
    RAW_BLACK_LEVEL,
    RawHandleBlackLevels,
    BayerUnpack,
    RAW_WHITE_LEVEL,
)
from image_dataset.dataset_loaders.CEL import RawCELDatasetLoader, cel_filters
from skimage import metrics


INPUT_EXPOSURES = [0.1]
OUTPUT_EXPOSURES = [1]

TRAIN_JSON: str = "./dataset/train.JSON"

TRAIN_JSON: str = "/mnt/5488CA8688CA6658/WORK/dataset/test.JSON"

def scale_range_plot(image, maxwhite):
    image = (image).astype("float64") / maxwhite
    image *= 255
    return image.astype("uint8")

def PSNR_SSIM_RAW(input, truth):
    psnr = metrics.peak_signal_noise_ratio(truth, input, data_range=RAW_WHITE_LEVEL)
    ssim = metrics.structural_similarity(
        truth,
        input,
        data_range=RAW_WHITE_LEVEL,
        multichannel=False,
    )

    return [psnr, ssim]


def PSNR_SSIM_RGB(input, truth):
    psnr = metrics.peak_signal_noise_ratio(truth, input, data_range=255)
    ssim = metrics.structural_similarity(
        truth, input, data_range=255, multichannel=True
    )

    return [psnr, ssim]


def GetRGB_FromPackedRaw(rawpyImage, rawData):
    rawpyImage.raw_image_visible[:] = BayerPack(rawData) + RAW_BLACK_LEVEL
    return rawpyImage.postprocess(
        no_auto_bright=True,
        output_bps=8,
        four_color_rgb=True,
        use_camera_wb=True,
    )


def BayerPack(image):
    # Pack bayer to 1 layer
    shape = image.shape
    h = shape[1]
    w = shape[2]

    out = np.zeros((h * 2, w * 2))

    out[0 : h * 2 : 2, 0 : w * 2 : 2] = image[0, :, :]

    out[0 : h * 2 : 2, 1 : w * 2 : 2] = image[1, :, :]

    out[1 : h * 2 : 2, 1 : w * 2 : 2] = image[2, :, :]

    out[1 : h * 2 : 2, 0 : w * 2 : 2] = image[3, :, :]

    return out


def Run():
    inputExposuresFilter = functools.partial(
        cel_filters.Exposures_Whitelist, INPUT_EXPOSURES
    )
    truthExposuresFilter = functools.partial(
        cel_filters.Exposures_Whitelist, OUTPUT_EXPOSURES
    )

    datasetLoader = RawCELDatasetLoader(
        TRAIN_JSON, inputExposuresFilter, truthExposuresFilter
    )
    sets = datasetLoader.GetSet()

    lightmap = LightMap.Load("./local/lightmap_test_0.1x1.pt")
    RUN_PREFIX = "0.1"

    outputdir = "./output/" + RUN_PREFIX + "/"
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    stupidMultiplication = OUTPUT_EXPOSURES[0] / INPUT_EXPOSURES[0]

    outputdir_raw = outputdir + "RAW/"
    if not os.path.isdir(outputdir_raw):
        os.makedirs(outputdir_raw)

    outputdir_rgb = outputdir + "RGB/"
    if not os.path.isdir(outputdir_rgb):
        os.makedirs(outputdir_rgb)


    metric_scenario = metric_handlers.Metric[int](name="Image Scenario")

    metric_global_raw_psnr = metric_handlers.Metric[float](name="Global RAW PSNR")
    metric_global_raw_ssim = metric_handlers.Metric[float](name="Global RAW SSIM")

    metric_local_raw_psnr = metric_handlers.Metric[float](name="Local RAW PSNR")
    metric_local_raw_ssim = metric_handlers.Metric[float](name="Local RAW SSIM")

    metric_stupid_raw_psnr = metric_handlers.Metric[float](name="Stupid RAW PSNR")
    metric_stupid_raw_ssim = metric_handlers.Metric[float](name="Stupid RAW SSIM")

    metric_global_rgb_psnr = metric_handlers.Metric[float](name="Global RGB PSNR")
    metric_global_rgb_ssim = metric_handlers.Metric[float](name="Global RGB SSIM")

    metric_local_rgb_psnr = metric_handlers.Metric[float](name="Local RGB PSNR")
    metric_local_rgb_ssim = metric_handlers.Metric[float](name="Local RGB SSIM")

    metric_stupid_rgb_psnr = metric_handlers.Metric[float](name="Stupid RGB PSNR")
    metric_stupid_rgb_ssim = metric_handlers.Metric[float](name="Stupid RGB SSIM")

    csvFileDir: str = outputdir + "data_RAW.csv"

    metricWriter = metric_handlers.MetricsToCsv(
        csvFileDir,
        [
            metric_scenario,
            metric_global_raw_psnr,
            metric_global_raw_ssim,
            metric_local_raw_psnr,
            metric_local_raw_ssim,
            metric_stupid_raw_psnr,
            metric_stupid_raw_ssim,
            metric_global_rgb_psnr,
            metric_global_rgb_ssim,
            metric_local_rgb_psnr,
            metric_local_rgb_ssim,
            metric_stupid_rgb_psnr,
            metric_stupid_rgb_ssim,
        ],
    )

    for set in sets:

        celInput, celTruth = set.GetPair()

        scenario = celInput.scenario

        if not scenario in lightmap._samplesMetadata:
            continue

        raw_Input = celInput.Load()
        raw_Truth = celTruth.Load()

        rawpyImage: RawPy = rawpy.imread(celInput.path)

        raw_Input = BayerUnpack(RawHandleBlackLevels(raw_Input)).transpose((2, 0, 1))
        raw_Truth = BayerUnpack(RawHandleBlackLevels(raw_Truth)).transpose((2, 0, 1))

        raw_Input = torch.tensor(raw_Input.astype("int32"))
        raw_Truth = torch.tensor(raw_Truth.astype("int32"))

        print("Processing " + set.trainList[0].path)

        raw_stupid = raw_Input * stupidMultiplication
        raw_stupid[raw_stupid > RAW_WHITE_LEVEL] = RAW_WHITE_LEVEL

        raw_globalRelight = lightmap.Relight(raw_Input.to(torch.float))

        sampleArrayIndex = lightmap._samplesMetadata[scenario]["samples_array_index"]
        localMap = lightmap._samples[sampleArrayIndex]
        raw_localRelight = lightmap.Relight(raw_Input.to(torch.float), localMap)

        rgb_truth = GetRGB_FromPackedRaw(rawpyImage, raw_Truth)
        rgb_stupid = GetRGB_FromPackedRaw(rawpyImage, raw_stupid)
        rgb_localRelight = GetRGB_FromPackedRaw(rawpyImage, raw_localRelight)
        rgb_globalRelight = GetRGB_FromPackedRaw(rawpyImage, raw_globalRelight)

        raw_globalRelight = BayerPack(raw_globalRelight)
        raw_localRelight = BayerPack(raw_localRelight)
        raw_stupid = BayerPack(raw_stupid)
        raw_Truth = BayerPack(raw_Truth)

        metric_scenario.Call(scenario)

        raw_global_psnr, raw_global_ssim = PSNR_SSIM_RAW(raw_globalRelight, raw_Truth)
        rgb_global_psnr, rgb_global_ssim = PSNR_SSIM_RGB(rgb_globalRelight, rgb_truth)

        metric_global_raw_psnr.Call(raw_global_psnr)
        metric_global_raw_ssim.Call(raw_global_ssim)

        metric_global_rgb_psnr.Call(rgb_global_psnr)
        metric_global_rgb_ssim.Call(rgb_global_ssim)

        # local

        raw_local_psnr, raw_local_ssim = PSNR_SSIM_RAW(raw_localRelight, raw_Truth)
        rgb_local_psnr, rgb_local_ssim = PSNR_SSIM_RGB(rgb_localRelight, rgb_truth)

        metric_local_raw_psnr.Call(raw_local_psnr)
        metric_local_raw_ssim.Call(raw_local_ssim)

        metric_local_rgb_psnr.Call(rgb_local_psnr)
        metric_local_rgb_ssim.Call(rgb_local_ssim)

        # stupid

        raw_stupid_psnr, raw_stupid_ssim = PSNR_SSIM_RAW(raw_stupid, raw_Truth)
        rgb_stupid_psnr, rgb_stupid_ssim = PSNR_SSIM_RGB(rgb_stupid, rgb_truth)

        metric_stupid_raw_psnr.Call(raw_stupid_psnr)
        metric_stupid_raw_ssim.Call(raw_stupid_ssim)

        metric_stupid_rgb_psnr.Call(rgb_stupid_psnr)
        metric_stupid_rgb_ssim.Call(rgb_stupid_ssim)

        rawdir = outputdir_raw + scenario.__str__()
        rgbdir = outputdir_rgb + scenario.__str__()

        imageio.imwrite(rawdir + "_global", scale_range_plot(raw_globalRelight,RAW_WHITE_LEVEL - RAW_BLACK_LEVEL), "jpg")
        imageio.imwrite(rawdir + "_local", scale_range_plot(raw_localRelight,RAW_WHITE_LEVEL - RAW_BLACK_LEVEL), "jpg")
        imageio.imwrite(rawdir + "_stupid", scale_range_plot(raw_stupid,RAW_WHITE_LEVEL - RAW_BLACK_LEVEL), "jpg")
        imageio.imwrite(rawdir + "_truth", scale_range_plot(raw_Truth,RAW_WHITE_LEVEL - RAW_BLACK_LEVEL), "jpg")

        imageio.imwrite(rgbdir + "_global", rgb_globalRelight, "jpg")
        imageio.imwrite(rgbdir + "_local", rgb_localRelight, "jpg")
        imageio.imwrite(rgbdir + "_stupid", rgb_stupid, "jpg")
        imageio.imwrite(rgbdir + "_truth", rgb_truth, "jpg")

    metricWriter.Write()


if __name__ == "__main__":
    Run()
