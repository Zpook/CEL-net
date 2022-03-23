from typing import Dict
import numpy as np
import torch
import dill
from joblib import Parallel, delayed


class LightMap:
    '''
    maxProcesses:int - set to -1 to use maximum available (maximum of 4)
    '''
    def __init__(self, maxwhite: int, channels: int, maxProcesses:int = 1, device:str="cpu") -> None:
        self.channels = channels
        self.maxwhite = maxwhite
        self._device = device

        self._samples = []
        self._samplesMetadata = {}
        self._unindexedImageCount = 0

        self._map = self._NewMap()

        if maxProcesses >= 4 or maxProcesses == -1:
            maxProcesses = 4    
        self._processCount = maxProcesses

        self.ToDevice(device)


    def _GetChannelMap(self,input: torch.Tensor, truth: torch.Tensor):
        
        map = torch.ones((self.maxwhite)).to(device=self._device)
        input = input.to(self._device)
        truth = truth.to(self._device)

        maxValue = torch.minimum(input.max(),torch.tensor(self.maxwhite)).to(torch.int)

        for intensity in range(1, maxValue):

            if intensity not in input:
                map[intensity] = 1
                continue

            indexes = input == intensity

            valuesTrue = truth[indexes]
            targetMeanValue = torch.mean(valuesTrue.to(torch.float))

            map[intensity] = targetMeanValue / intensity

        return map  

    def _RelightChannel(self, input: torch.Tensor, lightmap: torch.Tensor):
        input=input.to(self._device)
        output = input.clone().to(self._device)

        maxValue = torch.minimum(input.max(),torch.tensor(self.maxwhite)).to(torch.int)

        for intensity in range(1, maxValue):

            if lightmap[intensity] == 1:
                continue

            indexes = input == intensity

            if indexes.sum() == 0:
                continue
            
            output[indexes] = (output[indexes] * lightmap[intensity]).to(torch.float)
            output = output

        return output

    def _NewMap(self):
        return torch.ones((self.maxwhite,self.channels)).to(self._device)


    def _RelightChannelByTruth(self, input: torch.Tensor, truth: torch.Tensor):

        input = input.to(self._device)
        output = input.clone().to(self._device)
        truth = truth.to(self._device)

        maxValue = torch.minimum(input.max(),torch.tensor(self.maxwhite)).to(torch.int)

        for intensity in range(1, maxValue):

            if intensity not in input:
                continue

            indexes = input == intensity

            valuesTrue = truth[indexes]
            targetMeanValue = torch.mean(valuesTrue.to(torch.float))

            multiplier = targetMeanValue / intensity
            output[indexes] = output * multiplier
            output = output

        return output


    def GetImageMap(self,input: torch.Tensor, truth: torch.Tensor):
        
        channels = input.shape[0]
        assert channels == self.channels, "Channel mistmach"


        newMap = self._NewMap()
        results = Parallel(n_jobs=self._processCount)(delayed(self._GetChannelMap)(input[channelIndex,:,:],truth[channelIndex,:,:]) for channelIndex in range(channels))

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

    def SampleImage(self, input: torch.Tensor, truth: torch.Tensor,index:int = None,metadata:dict = None):
        
        sampleMap = self.GetImageMap(input,truth)
        self.AddSample(sampleMap,index,metadata)
        self._GenerateAverageMap()

    def AddSample(self,sample:np.ndarray, imageIndex:int = None, metadata:dict = None):
        assert sample.shape == (self.maxwhite,self.channels), "Dimension Mistmach"

        imageMeta = {}
        imageSampleIndex = None

        # overwrite previous sample
        if imageIndex != None and imageIndex in self._samplesMetadata.keys():
            imageMeta = self._samplesMetadata[imageIndex]
            imageSampleIndex = self._samplesMetadata[imageIndex]["samples_array_index"]
            self._samples[imageSampleIndex] = sample

        # new sample
        else:
            self._samples.append(sample)
            imageSampleIndex = self._samples.__len__()-1
        
        imageMeta["samples_array_index"] = imageSampleIndex
        if metadata != None:
            imageMeta["meta"] = metadata

        if imageIndex == None:
            imageIndex = "unindexed_" + self._unindexedImageCount.__str__()
            self._unindexedImageCount += 1

        self._samplesMetadata[imageIndex] = imageMeta

    def Relight(self,input,map = None):
        channels = input.shape[0]
        assert channels == self.channels, "Channel mistmach"

        if map is None:
            map = self._map

        output = input.clone()

        results = Parallel(n_jobs=self._processCount)(delayed(self._RelightChannel)(input[channelIndex,:,:],map[:,channelIndex]) for channelIndex in range(channels))

        for channelIndex in range(channels):
            output[channelIndex,:,:] = results[channelIndex]

        return output

    def Save(self,dir):
        file = open(dir,"wb")
        torch.save(self,file)
        file.close()

    @classmethod
    def Load(clse,dir,device="cpu") -> "LightMap":
        file = open(dir,"rb")
        object = torch.load(file,map_location=device)
        object.ToDevice(device)
        file.close()
        return object

    def ToDevice(self, device:str):
        self._device = torch.device(device)
        self._map = self._map.to(device)

        for index in range(self._samples.__len__()):
            self._samples[index] = self._samples[index].to(device)