import numpy as np


class RawRelight:
    @classmethod
    def GetLightmap(cls, input: np.ndarray, truth: np.ndarray, maxWhite: int):

        map = np.zeros((maxWhite))

        for intensity in range(1, maxWhite):
            indexes = input == intensity

            if indexes.sum() == 0:
                map[intensity] = 1
                continue

            valuesTrue = truth[indexes]
            targetMeanValue = np.mean(valuesTrue)

            map[intensity] = targetMeanValue / intensity

        return map

    @classmethod
    def RelightByTruth(cls, input: np.ndarray, truth: np.ndarray, maxWhite: int):

        output = input.copy()
        imageTrue = truth.copy()

        for intensity in range(1, maxWhite):
            indexes = input == intensity

            if indexes.sum() == 0:
                continue

            valuesTrue = imageTrue[indexes]
            targetMeanValue = np.mean(valuesTrue)

            multiplier = targetMeanValue / intensity
            output[indexes] = output.astype("float64")[indexes] * multiplier
            output = output.astype("uint16")

        return output

    def Relight(self, input: np.ndarray, lightmap, maxWhite: int):
        output = input.copy()

        for intensity in range(1, maxWhite):
            if lightmap[intensity] == 1:
                continue

            indexes = input == intensity

            if indexes.sum() == 0:
                continue
            
            output[indexes] = output.astype("float64")[indexes] * lightmap[intensity]
            output = output.astype("uint16")

        return output
