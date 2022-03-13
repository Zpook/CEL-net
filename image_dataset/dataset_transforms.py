import numpy as np
import torch

from image_dataset.dataset_loaders import BaseImage

from typing import Any, Callable, Generic, List, TypeVar
from abc import ABC, abstractmethod


ImageMetaType = TypeVar("ImageMetaType", bound=BaseImage)


class _PairMetaTransform(ABC, Generic[ImageMetaType]):
    @abstractmethod
    def _Apply(
        self, trainImage, truthImage, trainMeta: ImageMetaType, truthMeta: ImageMetaType
    ) -> List[Any]:
        pass

    def __call__(self, sample):
        trainImage, truthImage, trainMeta, truthMeta = sample

        trainImage, truthImage = self._Apply(
            trainImage, truthImage, trainMeta, truthMeta
        )

        sample[0] = trainImage
        sample[1] = truthImage

        return sample


class _PairTransform(ABC):
    @abstractmethod
    def _Apply(self, train, truth) -> List[Any]:
        pass

    def __call__(self, sample):
        train, truth = sample[0:2]

        train, truth = self._Apply(train, truth)

        sample[0] = train
        sample[1] = truth

        return sample


class _PairTransform_Conditional(ABC):
    def __init__(self, applyTrain=True, applyTruth=True) -> None:
        self._applyTrain = applyTrain
        self._applyTruth = applyTruth

    @abstractmethod
    def _Apply(self, image):
        pass

    def __call__(self, sample):

        train, truth = sample[0:2]

        if self._applyTrain:
            train = self._Apply(train)
        if self._applyTruth:
            truth = self._Apply(truth)

        sample[0] = train
        sample[1] = truth

        return sample


class RandomCrop(_PairTransform):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def _Apply(self, train, gtruth):

        h, w = train.shape[:2]
        new_h, new_w = self.output_size

        if new_h >= h:
            top = 0
        else:
            top = np.random.randint(0, h - new_h)

        if new_w >= w:
            left = 0
        else:
            left = np.random.randint(0, w - new_w)

        train = train[top : top + new_h, left : left + new_w]
        gtruth = gtruth[top : top + new_h, left : left + new_w]

        return [train, gtruth]

class CenterCrop(_PairTransform):
    """Crop center of image in a sample.

    Args:
        output_size (tuple, list ,int): Desired output size. 
        If tuple or list: (height, width)
        If int: square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def _Apply(self, train, gtruth):

        h, w = train.shape[:2]
        new_h, new_w = self.output_size

        if new_h >= h:
            top = 0
        else:
            top = int((h - int(new_h)) / 2)

        if new_w >= w:
            left = 0
        else:
            left = int((w - int(new_w)) / 2)

        train = train[top : top + new_h, left : left + new_w]
        gtruth = gtruth[top : top + new_h, left : left + new_w]

        return [train, gtruth]




class RandomCropRAWandRGB(_PairTransform):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple, list ,int): Desired output size. 
        If tuple or list: (height, width)
        If int: square crop is made.

    Expected samples : (RAW, RGB)
    The crop dimensions are assumed to be for the RAW channels, and will be twice as large for the RGB image.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.outputSize = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.outputSize = output_size

    def _Apply(self, image, gtruth):

        h, w = image.shape[:2]
        new_h, new_w = self.outputSize

        if new_h >= h:
            top = 0
        else:
            top = np.random.randint(0, h - new_h)

        if new_w >= w:
            left = 0
        else:
            left = np.random.randint(0, w - new_w)

        image = image[top : top + new_h, left : left + new_w]
        gtruth = gtruth[top * 2 : top * 2 + new_h * 2, left * 2 : left * 2 + new_w * 2]

        return [image, gtruth]


class CenterCropRAWandRGB(_PairTransform):
    """Crop center of the image in a sample.

    Args:
        output_size (tuple, list ,int): Desired output size. 
        If tuple or list: (height, width)
        If int: square crop is made.

    Expected samples : (RAW, RGB)
    The crop dimensions are assumed to be for the RAW channels, and will be twice as large for the RGB image.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.outputSize = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.outputSize = output_size

    def _Apply(self, image, gtruth):

        h, w = image.shape[:2]
        new_h, new_w = self.outputSize

        if new_h >= h:
            top = 0
        else:
            top = int((h - int(new_h)) / 2)

        if new_w >= w:
            left = 0
        else:
            left = int((w - int(new_w)) / 2)

        image = image[top : top + new_h, left : left + new_w]
        gtruth = gtruth[top * 2 : top * 2 + new_h * 2, left * 2 : left * 2 + new_w * 2]

        return [image, gtruth]

class RandomFlip(_PairTransform):
    """Random image flip"""

    def _Apply(self, image, gtruth):

        flipX = np.random.randint(0, 2)
        flipY = np.random.randint(0, 2)

        flipDims =[]
        if flipX:
            flipDims.append(1)
        if flipY:
            flipDims.append(0)

        if flipDims.__len__() > 0:
            image = torch.flip(image, dims=flipDims)
            gtruth = torch.flip(gtruth, dims=flipDims)


        return [image, gtruth]


class Normalize(_PairTransform_Conditional):
    def __init__(
        self, mean: float, std: float, applyTrain: bool = True, applyTruth: bool = True
    ):
        super().__init__(applyTrain, applyTruth)
        self._mean = mean
        self._std = std

    def _Apply(self, image):
        return np.float32(image - self._mean) / self._std


class ToTensor(_PairTransform):
    """Convert ndarrays in sample to Tensors."""

    def _Apply(self, image, gtruth):
        image = torch.Tensor(np.ascontiguousarray(image, dtype=np.float32))
        gtruth = torch.Tensor(np.ascontiguousarray(gtruth, dtype=np.float32))

        return [image, gtruth]


class ToDevice(_PairTransform):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device: str = "cpu"):
        self._device = device

    def _Apply(self, image, gtruth):

        image = image.to(self._device)
        gtruth = gtruth.to(self._device)

        return [image, gtruth]


class Permute(_PairTransform):
    """ Permute both images """

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def _Apply(self, image, gtruth):

        image = image.permute(*self.args, **self.kwargs)
        gtruth = gtruth.permute(*self.args, **self.kwargs)

        return [image, gtruth]


class BayerUnpack(_PairTransform_Conditional):
    def __init__(self, applyTrain: bool = True, applyTruth: bool = True) -> None:
        super().__init__(applyTrain, applyTruth)

    def _Apply(self, image):
        # pack Bayer image to 4 channels
        image = image.unsqueeze(2)
        img_shape = image.shape
        H = img_shape[0]
        W = img_shape[1]

        out = torch.cat(
            (
                image[0:H:2, 0:W:2, :],
                image[0:H:2, 1:W:2, :],
                image[1:H:2, 1:W:2, :],
                image[1:H:2, 0:W:2, :],
            ),
            dim=2,
        )
        return out


class Lambda(_PairTransform):
    def __init__(
        self, callback: Callable[[np.ndarray, np.ndarray], List[np.ndarray]]
    ) -> None:
        self._callback = callback

    def _Apply(self, train, gtruth):
        return self._callback(train, gtruth)
