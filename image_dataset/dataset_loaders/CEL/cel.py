import json
import rawpy
import os
import numpy as np
from types import MethodType

from image_dataset.dataset_loaders import (
    BaseDatasetLoader,
    BaseDatasetPair,
    BaseImage,
)

from typing import Dict, List, Callable, Union

IMAGE_BPS = 16


class CELImage(BaseImage):
    def __init__(
        self,
        imagePath: str,
        scenario: int,
        location: str,
        exposure: float,
        focalLength: float,
        f_number: float,
        iso: int,
    ):

        basename = os.path.basename(imagePath)
        formatSplit = basename.rsplit(".", 1)
        format = formatSplit[1]

        BaseImage.__init__(self, imagePath, format)

        self.scenario = scenario
        self.location = location
        self.exposure = exposure
        self.focalLength = focalLength
        self.f_number = f_number
        self.iso = iso

    @classmethod
    def FromJSON(cls, relativePath: str, jsonDict: Dict):

        outputArr: List[CELImage] = []

        for scenarioKey, scenarioDict in jsonDict.items():
            for imageKey, imageData in scenarioDict.items():
                imagePath = relativePath + imageData["location"]
                scenario = imageData["scenario"]
                exposure = imageData["exposure"]
                focal_length = imageData["focal_length"]
                # fnumber is aperture in our meta JSON files
                f_number = imageData["aperture"]
                location = scenarioKey.split("_")[0]
                iso = imageData["iso"]

                # this is a quick fix to the fact that we have indoor and outdoor images with the same scenario index
                if location == "indoor":
                    scenario += 1000
                else:
                    scenario += 2000

                outputArr.append(
                    cls(
                        imagePath, scenario, location, exposure, focal_length, f_number, iso
                    )
                )

        return outputArr


class CELPair(BaseDatasetPair):
    def __init__(
        self,
        trainingImage: Union[CELImage, List[CELImage]],
        truthImage: Union[CELImage, List[CELImage]],
    ) -> None:

        self.trainList: List[CELImage] = self._LoadListOrImage(trainingImage)
        self.truthList: List[CELImage] = self._LoadListOrImage(truthImage)

    def GetPair(self):
        train = self._GetRandomImage(self.trainList)
        truth = self._GetRandomImage(self.truthList)

        return [train, truth]

    def _LoadListOrImage(self, listOrImage: Union[CELImage, List[CELImage]]):
        if isinstance(listOrImage, list):
            return listOrImage
        else:
            return [listOrImage]

    def _GetRandomImage(self, arr):
        randint = np.random.randint(0, arr.__len__())
        return arr[randint]


def RAWImageLoadHook(self: BaseImage):
    rawImage = rawpy.imread(self.path)
    postProcImage = rawImage.postprocess(
        no_auto_bright=True,
        output_bps=IMAGE_BPS,
        four_color_rgb=True,
        use_camera_wb=True,
    )

    return postProcImage


DatasetFilterCallbackType = Callable[[List[CELImage]], List[CELImage]]
nopFilter: Callable[[List[CELImage]], List[CELImage]] = lambda images: images


class CELDatasetLoader(BaseDatasetLoader):
    def __init__(
        self,
        path: str,
        trainFilter: DatasetFilterCallbackType = nopFilter,
        truthFilter: DatasetFilterCallbackType = nopFilter,
    ) -> None:
        self._metaFilePath = path
        self._trainFilter = trainFilter
        self._truthFilter = truthFilter

    def _GenerateScenarioDict(self, imageList: List[CELImage]):
        newDict: Dict[int, List[CELImage]] = {}

        for image in imageList:
            if image.scenario in newDict.keys():
                newDict[image.scenario].append(image)
            else:
                newDict[image.scenario] = [image]

        return newDict

    def _GeneratePairs(
        self,
        trainList: List[CELImage],
        truthList: List[CELImage],
    ):

        truthDict: Dict[int, List[CELImage]] = {}
        trainDict: Dict[int, List[CELImage]] = {}
        pairs: List[CELPair] = []

        truthDict = self._GenerateScenarioDict(truthList)
        trainDict = self._GenerateScenarioDict(trainList)

        trainKeysToRemove = []
        for scenario in trainDict.keys():
            if scenario not in truthDict.keys():
                trainKeysToRemove.append(scenario)

        for key in trainKeysToRemove:
            del trainDict[key]

        for trainKey in trainDict:
            trainInd = trainDict[trainKey]
            truthInd = truthDict[trainKey]

            for index, image in enumerate(truthInd):
                image.LoadHook = MethodType(RAWImageLoadHook, image)
                truthInd[index] = image

            newPair = CELPair(trainInd, truthInd)
            pairs.append(newPair)

        return pairs

    def GetSet(self):

        with open(self._metaFilePath, "r") as file:
            imageDict = json.load(file)

        metaFileDirectory = os.path.split(self._metaFilePath)[0] + "/"

        trainImages = CELImage.FromJSON(metaFileDirectory, imageDict)
        truthImages = CELImage.FromJSON(metaFileDirectory, imageDict)

        trainImages = self._trainFilter(trainImages)
        truthImages = self._truthFilter(truthImages)

        # associate truth and train with their respective filtered scenarios
        pairs = self._GeneratePairs(trainImages, truthImages)

        return pairs


class RawCELDatasetLoader(CELDatasetLoader):
    def __init__(
        self,
        path: str,
        trainFilter: DatasetFilterCallbackType = nopFilter,
        truthFilter: DatasetFilterCallbackType = nopFilter,
    ) -> None:
        CELDatasetLoader.__init__(self,path,trainFilter,truthFilter)

    def _GeneratePairs(
        self,
        trainList: List[CELImage],
        truthList: List[CELImage],
    ):

        truthDict: Dict[int, List[CELImage]] = {}
        trainDict: Dict[int, List[CELImage]] = {}
        pairs: List[CELPair] = []

        truthDict = self._GenerateScenarioDict(truthList)
        trainDict = self._GenerateScenarioDict(trainList)

        trainKeysToRemove = []
        for scenario in trainDict.keys():
            if scenario not in truthDict.keys():
                trainKeysToRemove.append(scenario)

        for key in trainKeysToRemove:
            del trainDict[key]

        for trainKey in trainDict:
            trainInd = trainDict[trainKey]
            truthInd = truthDict[trainKey]

            for index, image in enumerate(truthInd):
                truthInd[index] = image

            newPair = CELPair(trainInd, truthInd)
            pairs.append(newPair)

        return pairs