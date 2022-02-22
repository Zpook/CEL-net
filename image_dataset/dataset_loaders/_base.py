import imageio
import rawpy
from rawpy._rawpy import RawPy
import abc

from typing import List, TypeVar


class BaseImage:
    def __init__(self, path: str, format: str) -> None:
        self.path = path
        self.format = format

        self._cache = None


    def LoadHook(self):
        if self.format == "ARW":
            rawImage: RawPy = rawpy.imread(self.path)
            return rawImage.raw_image_visible

        else:
            return imageio.imread(self.path)

    def Load(self, useCache: bool = True):
        if useCache and self._cache is not None:
            return self._cache

        return self.LoadHook()

    def _LoadDefault(self, useCache: bool = True):
        if useCache and self._cache is not None:
            return self._cache

        return self._LoadHookDefault()

    
    def _LoadHookDefault(self):
        if self.format == "ARW":
            rawImage: RawPy = rawpy.imread(self.path)
            return rawImage.raw_image_visible

        else:
            return imageio.imread(self.path)


TBaseImage = TypeVar("TBaseImage", bound=BaseImage, covariant=True)


class BaseDatasetPair(abc.ABC):
    @abc.abstractmethod
    def GetPair(self) -> List[TBaseImage]:
        pass


TDatasetPair = TypeVar("TDatasetPair", bound=BaseDatasetPair, covariant=True)


class BaseDatasetLoader(abc.ABC):
    @abc.abstractmethod
    def GetSet(self) -> List[TDatasetPair]:
        pass
