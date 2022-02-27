from typing import Union, Tuple

from torchvision import transforms
import numpy as np

from image_dataset import dataset_transforms
from image_dataset.dataset_loaders.CEL import CELImage
from util.rawrelight import RawRelight

RAW_BLACK_LEVEL = 512
RAW_WHITE_LEVEL = 16383


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
    image[image<512] = 512
    image = image - 512
    return image

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
    def __init__(self, truthImageBps: int) -> None:
        self.truthImageBps: int = truthImageBps
        super().__init__()

    def _Apply(
        self,
        trainImage: np.ndarray,
        truthImage: np.ndarray,
        trainingData: CELImage,
        truthData: CELImage,
    ):
        rawTrain = trainingData.Load()
        rawTrain = RawHandleBlackLevels(rawTrain)
        rawTrain = BayerUnpack(rawTrain)

        rawTruth = truthData._LoadDefault()
        rawTruth = RawHandleBlackLevels(rawTruth)
        rawTruth = BayerUnpack(rawTruth)

        numChannels = rawTrain.shape[-1]

        for channel in range(numChannels):
            map = RawRelight.GetLightmap(rawTrain[:,:,channel],rawTruth[:,:,channel],RAW_WHITE_LEVEL)
            trainImage[channel,:,:] = RawRelight.Relight(trainImage[channel,:,:], map, RAW_WHITE_LEVEL)

        trainImage -= RAW_BLACK_LEVEL
        truthImage = truthImage / float(2 ** self.truthImageBps - 1)
        trainImage = RawRelight.Relight(trainImage, map, RAW_WHITE_LEVEL)
        trainImage /= (RAW_WHITE_LEVEL-RAW_BLACK_LEVEL)

        return [trainImage, truthImage]

def GetTrainTransforms(
    rgbBps: float, patchSize: Union[Tuple[int], int], normalize: bool, device: str
):

    transform = transforms.Compose(
        [
            dataset_transforms.BayerUnpack(applyTrain=True, applyTruth=False),
            dataset_transforms.RandomCropRAWandRGB(patchSize),
            dataset_transforms.RandomFlip(),
            dataset_transforms.ToTensor(),
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
            dataset_transforms.BayerUnpack(applyTrain=True, applyTruth=False),
            dataset_transforms.CenterCropRAWandRGB(patchSize),
            dataset_transforms.ToTensor(),
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
