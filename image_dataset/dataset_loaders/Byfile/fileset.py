import os
import glob

from image_dataset.dataset_loaders import (
    BaseDatasetLoader,
    BaseDatasetPair,
    BaseImage,
)

from typing import List


class DatasetImage(BaseImage):
    def __init__(
        self,
        imagePath: str,
    ):

        basename = os.path.basename(imagePath)
        formatSplit = basename.rsplit(".", 1)
        format = formatSplit[1]

        BaseImage.__init__(self, imagePath, format)


class DatasetPair(BaseDatasetPair):
    def __init__(
        self,
        input: DatasetImage,
        truth: DatasetImage,
    ) -> None:

        self.train: List[DatasetImage] = input
        self.truth: List[DatasetImage] = truth

    def GetPair(self):

        return [self.train, self.truth]




class DatasetLoaderByFiles(BaseDatasetLoader):
    def __init__(
        self,
        baseDir: str,
        inFiles: List[str],
        trueFiles: List[str]
    ) -> None:

     self._dir = baseDir
     self._inputs = inFiles
     self._truths = trueFiles


    @classmethod
    def _GenerateFullNames(cls,baseDir:str,imageList:List[str]):

        out = []
        for image in imageList:
            fullImageName = baseDir + image
            out.append(fullImageName)

        return out

    @classmethod
    def _GrabImages(cls, path: str, imageFormat: str, imagePrefix: str = ""):
        return glob.glob(path + imagePrefix + "*." + imageFormat)

    @classmethod
    def _GeneratePairs(cls,inputs:List[DatasetImage],truths:List[DatasetImage]):

        pairs = []
        for index in range(inputs.__len__()):
            input = inputs[index]
            truth = truths[index]
            newPair = DatasetPair(input,truth)
            pairs.append(newPair)

        return pairs
            



    def GetSet(self):
        inputs = self._GenerateFullNames(self._dir,self._inputs)
        truths = self._GenerateFullNames(self._dir,self._truths)

        assert inputs.__len__() == truths.__len__(), "Input / truth length mistmach"

        pairs = self._GeneratePairs(inputs,truths)

        return pairs






        




