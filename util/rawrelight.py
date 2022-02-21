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

        output = input.clone()
        imageTrue = truth.clone()

        for intensity in range(1, maxWhite):
            indexes = input == intensity

            if indexes.sum() == 0:
                continue

            valuesTrue = imageTrue[indexes]
            targetMeanValue = np.mean(valuesTrue)

            multiplier = targetMeanValue / intensity
            output[indexes] = output * multiplier
            output = output

        return output

    @classmethod
    def Relight(self, input: np.ndarray, lightmap, maxWhite: int):
        output = input.clone()

        for intensity in range(1, maxWhite):
            if lightmap[intensity] == 1:
                continue

            indexes = input == intensity

            if indexes.sum() == 0:
                continue
            
            output[indexes] = output[indexes] * lightmap[intensity]
            output = output

        return output
