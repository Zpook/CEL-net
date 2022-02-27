from typing import Dict
import numpy as np
import torch
import dill
from joblib import Parallel, delayed


class LightMap:
    '''
    maxProcesses:int - set to -1 to use maximum available (maximum of 4)
    '''
    def __init__(self, maxwhite: int, channels: int, maxProcesses:int = 1) -> None:
        self.channels = channels
        self.maxwhite = maxwhite
        self._samples = []

        if maxProcesses >= 4 or maxProcesses == -1:
            maxProcesses = 4    
        self._processCount = maxProcesses

        self._map = self._NewMap()

    @classmethod
    def _GetChannelMap(cls,input: torch.Tensor, truth: torch.Tensor, maxwhite:int):
        
        map = torch.zeros((maxwhite))

        for intensity in range(1, maxwhite):
            indexes = input == intensity

            if indexes.sum() == 0:
                map[intensity] = 1
                continue

            valuesTrue = truth[indexes]
            targetMeanValue = torch.mean(valuesTrue.to(torch.float))

            map[intensity] = targetMeanValue / intensity

        return map  

    @classmethod
    def _RelightChannel(self, input: torch.Tensor, lightmap: torch.Tensor, maxWhite: int):
        output = input.clone()

        for intensity in range(1, maxWhite):
            if lightmap[intensity] == 1:
                continue

            indexes = input == intensity

            if indexes.sum() == 0:
                continue
            
            output[indexes] = (output[indexes] * lightmap[intensity]).to(torch.int)
            output = output

        return output

    def _NewMap(self):
        return torch.ones((self.maxwhite,self.channels))


    @classmethod
    def _RelightChannelByTruth(cls, input: torch.Tensor, truth: torch.Tensor, maxWhite: int):

        output = input.clone()
        imageTrue = truth.clone()

        for intensity in range(1, maxWhite):
            indexes = input == intensity

            if indexes.sum() == 0: 
                continue

            valuesTrue = imageTrue[indexes]
            targetMeanValue = torch.mean(valuesTrue.to(torch.float))

            multiplier = targetMeanValue / intensity
            output[indexes] = output * multiplier
            output = output

        return output


    def GetImageMap(self,input: torch.Tensor, truth: torch.Tensor):
        
        channels = input.shape[0]
        assert channels == self.channels, "Channel mistmach"


        newMap = self._NewMap()
        results = Parallel(n_jobs=self._processCount)(delayed(self._GetChannelMap)(input[channelIndex,:,:],truth[channelIndex,:,:],self.maxwhite) for channelIndex in range(channels))

        for index in range(channels):
            newMap[:,index] = results[index]

        return newMap


    def _GenerateAverageMap(self):
        
        newMap = self._NewMap()
        meanMap = self._NewMap() - 1


        for sampleMap in self._samples:

            for channelIndex in range (self.channels):

                for index in range(newMap.shape[0]):

                    if newMap[index,channelIndex] == 1:
                        newMap[index,channelIndex] = sampleMap[index,channelIndex]
                        meanMap[index,channelIndex] = 1
                    
                    else:
                        newMap[index,channelIndex] =  newMap[index,channelIndex] + sampleMap[index,channelIndex]
                        meanMap[index,channelIndex] += 1
        
        newMap /= meanMap

        self._map = newMap

    def SampleImage(self, input: torch.Tensor, truth: torch.Tensor):
        
        sampleMap = self.GetImageMap(input,truth)
        self.AddSample(sampleMap)
        self._GenerateAverageMap()

    def AddSample(self,sample:np.ndarray):
        assert sample.shape == (self.maxwhite,self.channels), "Dimension Mistmach"
        self._samples.append(sample)
        

    def Relight(self,input):
        channels = input.shape[0]
        assert channels == self.channels, "Channel mistmach"

        output = input.clone()

        results = Parallel(n_jobs=self._processCount)(delayed(self._RelightChannel)(input[channelIndex,:,:],self._map[:,channelIndex],self.maxwhite) for channelIndex in range(channels))

        for channelIndex in range(channels):
            output[channelIndex,:,:] = results[channelIndex]

        return output


    def Relight_Old(self,input):
        channels = input.shape[0]
        assert channels == self.channels, "Channel mistmach"

        output = input.clone()

        for channelIndex in range(channels):
            output[channelIndex,:,:] = self._RelightChannel(input[channelIndex,:,:],self._map[:,channelIndex],self.maxwhite)

        return output

    def Save(self,dir):
        file = open(dir,"wb")
        dill.dump(self,file)
        file.close()

    @classmethod
    def Load(clse,dir):
        file = open(dir,"rb")
        object = dill.load(file)
        file.close()
        return object
