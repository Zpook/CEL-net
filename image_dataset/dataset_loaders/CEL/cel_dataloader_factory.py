import torch
import torch.utils.data
import os

from .cel import (
    DatasetFilterCallbackType,
    nopFilter,
    CELDatasetLoader,
)
from image_dataset.image_dataset import ImageDataset

from typing import Callable

class CELDataloaderFactory:
    def __init__(
        self,
        trainJSONDir: str,
        testJSONDir: str,
        patchSize: int = 512,
        datasetWorkers: int = 0,
        batch: int = 1,
        cacheLimit: int = 0,
    ):

        self._trainJSON = trainJSONDir
        self._testJSON = testJSONDir

        self._patchSize = patchSize
        self._datasetWorkers = datasetWorkers
        self._batch = batch
        self._cacheLimit = cacheLimit

    def GetTrain(
        self,
        transforms: Callable,
        inputFilter: DatasetFilterCallbackType = nopFilter,
        truthFilter: DatasetFilterCallbackType = nopFilter,
    ):
        datasetLoader = CELDatasetLoader(self._trainJSON, inputFilter, truthFilter)
        trainSet = datasetLoader.GetSet()

        trainDataset = ImageDataset(trainSet, transforms, cacheLimit=self._cacheLimit)

        trainDatasetLoader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=self._batch,
            shuffle=True,
            num_workers=self._datasetWorkers,
            collate_fn=ImageDataset.collate_fn,
        )
        return trainDatasetLoader

    def GetTest(
        self,
        transforms: Callable,
        inputFilter: DatasetFilterCallbackType = nopFilter,
        truthFilter: DatasetFilterCallbackType = nopFilter,
    ):
        datasetLoader = CELDatasetLoader(self._testJSON, inputFilter, truthFilter)
        testSet = datasetLoader.GetSet()

        testDataset = ImageDataset(testSet, transforms, cacheLimit=self._cacheLimit)

        testDatasetLoader = torch.utils.data.DataLoader(
            testDataset,
            batch_size=1,
            shuffle=False,
            num_workers=self._datasetWorkers,
            collate_fn=ImageDataset.collate_fn,
        )
        return testDatasetLoader
