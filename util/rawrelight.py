import imp
import numpy as np

class RawRelight:

    def __init__(self, lightmap) -> None:
        self._lightmap = lightmap

    def RelightByTruth(input: np.ndarray, truth: np.ndarray, maxWhite: int):

        inputImage = input.copy()
        imageTrue = truth.copy()

        for intensity in range(1,maxWhite):
            indexes = inputImage == intensity

            if indexes.sum() == 0:
                continue

            valuesTrue = imageTrue[indexes]
            targetMeanValue = np.mean(valuesTrue)
            meanValues = np.mean(inputImage[indexes])

            if meanValues == 0:
                continue

            multiplier = targetMeanValue / meanValues
            inputImage[indexes] = inputImage.astype('float64')[indexes] * multiplier
            inputImage = inputImage.astype('uint16')

    def Relight(input: np.ndarray, maxWhite: int):
        pass