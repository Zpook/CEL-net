from typing import Union, Tuple, Dict

from torchvision import transforms
import numpy as np

from image_dataset import dataset_transforms
from image_dataset.dataset_loaders.CEL import CELImage
from lightmap import LightMap

RAW_BLACK_LEVEL = 512
RAW_WHITE_LEVEL = 16383


def GetLightmaps(device, workerCount: int = 1):

    lightmapDict: Dict[float, LightMap] = {}

    lightmapDict[10] = LightMap.Load("./local/lightmap_full_0.1x10.pt", device=device)
    lightmapDict[5] = LightMap.Load("./local/lightmap_full_0.1x5.pt", device=device)
    lightmapDict[1] = LightMap.Load("./local/lightmap_full_0.1x1.pt", device=device)

    lightmapDict[10]._processCount = workerCount
    lightmapDict[5]._processCount = workerCount
    lightmapDict[1]._processCount = workerCount

    return lightmapDict


def BayerUnpack(image):
    # pack Bayer image to 4 channels
    image = np.expand_dims(image, axis=2)
    img_shape = image.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            image[0:H:2, 0:W:2, :],
            image[0:H:2, 1:W:2, :],
            image[1:H:2, 1:W:2, :],
            image[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )
    return out


def RawHandleBlackLevels(image):
    image[image < RAW_BLACK_LEVEL] = RAW_BLACK_LEVEL
    image = image - RAW_BLACK_LEVEL
    return image


class DummyNorm(dataset_transforms._PairMetaTransform):
    def __init__(self, truthImageBps: int):
        self.truthImageBps: int = truthImageBps

    def _Apply(
        self,
        trainImage: np.ndarray,
        truthImage: np.ndarray,
        trainingData: CELImage,
        truthData: CELImage,
    ):
        normRatio = 1 / (RAW_WHITE_LEVEL - RAW_BLACK_LEVEL)
        truthImage = truthImage / float(2 ** self.truthImageBps - 1)
        # to float and subtract black level
        trainImage = trainImage - RAW_BLACK_LEVEL
        trainImage *= normRatio
        trainImage = trainImage.clamp(0, 1)

        return [trainImage, truthImage]


class NormByExposureTime(dataset_transforms._PairMetaTransform):
    def __init__(self, truthImageBps: int):
        self.truthImageBps: int = truthImageBps

    def _Apply(
        self,
        trainImage: np.ndarray,
        truthImage: np.ndarray,
        trainingData: CELImage,
        truthData: CELImage,
    ):
        exposureRatio = (float(truthData.exposure) / trainingData.exposure) / (
            RAW_WHITE_LEVEL - RAW_BLACK_LEVEL
        )
        truthImage = truthImage / float(2 ** self.truthImageBps - 1)
        # to float and subtract black level
        trainImage = trainImage - RAW_BLACK_LEVEL
        trainImage *= exposureRatio
        trainImage = trainImage.clamp(0, 1)

        return [trainImage, truthImage]


class NormByRelight(dataset_transforms._PairMetaTransform):
    def __init__(self, lightmaps: Dict[float, LightMap], truthImageBps: int) -> None:
        self._lightmaps = lightmaps
        self._truthImageBps: int = truthImageBps
        super().__init__()

    def _Apply(
        self,
        trainImage: np.ndarray,
        truthImage: np.ndarray,
        trainingData: CELImage,
        truthData: CELImage,
    ):

        truthExp = truthData.exposure
        lightmap = self._lightmaps[truthExp]

        truthImage = truthImage / float(2 ** self._truthImageBps - 1)
        trainImage = RawHandleBlackLevels(trainImage)

        trainImage = lightmap.Relight(trainImage)
        trainImage /= RAW_WHITE_LEVEL - RAW_BLACK_LEVEL
        trainImage = trainImage.clamp(0, 1)

        return [trainImage, truthImage]


class NormByRelight_Local(dataset_transforms._PairMetaTransform):
    def __init__(
        self,
        lightmaps: Dict[float, LightMap],
        truthImageBps: int,
        normalize: bool = False,
        normValue: int = 160
    ) -> None:
        self._lightmaps = lightmaps
        self._truthImageBps: int = truthImageBps
        self._normalize = normalize
        self._normValue = normValue
        super().__init__()

    def _Apply(
        self,
        trainImage: np.ndarray,
        truthImage: np.ndarray,
        trainingData: CELImage,
        truthData: CELImage,
    ):

        truthExp = truthData.exposure
        lightmap = self._lightmaps[truthExp]

        truthImage = truthImage / float(2 ** self._truthImageBps - 1)
        trainImage = RawHandleBlackLevels(trainImage)

        scenario = trainingData.scenario
        sampleArrayIndex = lightmap._samplesMetadata[scenario]["samples_array_index"]
        localMap = lightmap._samples[sampleArrayIndex]
        trainImage = lightmap.Relight(trainImage, localMap)

        if self._normalize:
            maxValue = localMap[self._normValue]
            
        else:
            trainImage /= RAW_WHITE_LEVEL - RAW_BLACK_LEVEL
            trainImage = trainImage.clamp(0, 1)

        return [trainImage, truthImage]


def GetTrainTransforms(
    rgbBps: float, patchSize: Union[Tuple[int], int], normalize: bool, device: str
):

    transform = transforms.Compose(
        [
            dataset_transforms.ToTensor(),
            dataset_transforms.BayerUnpack(applyTrain=True, applyTruth=False),
            dataset_transforms.RandomCropRAWandRGB(patchSize),
            dataset_transforms.RandomFlip(),
            dataset_transforms.Permute(2, 0, 1),
            dataset_transforms.ToDevice(device),
        ]
    )

    if normalize:

        normTransforms = transforms.Compose(
            [
                dataset_transforms.Normalize(
                    0, 2 ** rgbBps - 1, applyTrain=False, applyTruth=True
                ),
                dataset_transforms.Normalize(
                    RAW_BLACK_LEVEL,
                    RAW_WHITE_LEVEL - RAW_BLACK_LEVEL,
                    applyTrain=True,
                    applyTruth=False,
                ),
            ]
        )

        transform = transforms.Compose([normTransforms, transform])

    return transform


def GetEvalTransforms(
    rgbBps: float, patchSize: Union[int, Tuple[int]], normalize: bool, device: str
):

    transform = transforms.Compose(
        [
            dataset_transforms.ToTensor(),
            dataset_transforms.BayerUnpack(applyTrain=True, applyTruth=False),
            dataset_transforms.CenterCropRAWandRGB(patchSize),
            dataset_transforms.Permute(2, 0, 1),
            dataset_transforms.ToDevice(device),
        ]
    )

    if normalize:

        normTransforms = transforms.Compose(
            [
                dataset_transforms.Normalize(
                    0, 2 ** rgbBps - 1, applyTrain=False, applyTruth=True
                ),
                dataset_transforms.Normalize(
                    RAW_BLACK_LEVEL,
                    RAW_WHITE_LEVEL - RAW_BLACK_LEVEL,
                    applyTrain=True,
                    applyTruth=False,
                ),
            ]
        )

        transform = transforms.Compose([normTransforms, transform])

    return transform
