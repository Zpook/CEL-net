from image_dataset.dataset_loaders.CEL import CELImage

from typing import List


def Exposures_Whitelist(exposures: List[float], images: List[CELImage]):
    newList: List[CELImage] = []

    for image in images:
        if image.exposure in exposures:
            newList.append(image)

    return newList


def Exposure_Blacklist(exposures: List[float], images: List[CELImage]):
    newList: List[CELImage] = []

    for image in images:
        if not image.exposure in exposures:
            newList.append(image)

    return newList


def Exposure_InRange(minExposure: float, maxExposure: float, images: List[CELImage]):
    newList: List[CELImage] = []

    for image in images:
        if image.exposure <= maxExposure and image.exposure >= minExposure:
            newList.append(image)

    return newList

def Scenario_Whitelist(scenarioList: List[int], images: List[CELImage]):
    newList: List[CELImage] = []

    for image in images:
        if image.scenario in scenarioList:
            newList.append(image)

    return newList

def Scenario_Blacklist(scenarioList: List[int], images: List[CELImage]):
    newList: List[CELImage] = []

    for image in images:
        if image.scenario in scenarioList:
            newList.append(image)

    return newList

def Chain(funcs: List,images: List[CELImage]):
    for func in funcs:
        images = func(images)
    return images

