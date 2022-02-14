from typing import Union, Tuple

from torchvision import transforms
import numpy as np

from image_dataset import dataset_transforms
from image_dataset.dataset_loaders.CEL import CELImage

RAW_BLACK_LEVEL = 512
RAW_MAX = 16383


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
            RAW_MAX - RAW_BLACK_LEVEL
        )
        truthImage = truthImage / float(2 ** self.truthImageBps - 1)
        # to float and subtract black level
        trainImage = trainImage - RAW_BLACK_LEVEL
        trainImage *= exposureRatio
        trainImage = trainImage.clamp(0, 1)

        return [trainImage, truthImage]


def GetTrainTransforms(
    patchSize: Union[Tuple[int], int], device: str
):

    transform = transforms.Compose(
        [
            dataset_transforms.BayerUnpack(applyTrain=True, applyTruth=False),
            dataset_transforms.RandomCrop(patchSize),
            dataset_transforms.RandomFlip(),
            dataset_transforms.ToTensor(),
            dataset_transforms.Permute(2, 0, 1),
            dataset_transforms.ToDevice(device),
        ]
    )

    return transform


def GetEvalTransforms(
    patchSize: Union[int, Tuple[int]], device: str
):

    transform = transforms.Compose(
        [
            dataset_transforms.BayerUnpack(applyTrain=True, applyTruth=False),
            dataset_transforms.CenterCrop(patchSize),
            dataset_transforms.ToTensor(),
            dataset_transforms.Permute(2, 0, 1),
            dataset_transforms.ToDevice(device),
        ]
    )

    return transform
