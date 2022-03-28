from __future__ import division
import logging
import time
import torch
import torch.cuda
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn

from util.events import Event

from image_dataset.dataset_loaders.CEL import CELImage
from typing import Any, Callable, Dict, List, Union

CHECKPOINT_EPOCH_KEYWORD: str = "epoch"
CHECKPOINT_WEIGHTS_KEYWORD: str = "model_state_dict"
CHECKPOINT_OPTIMISER_STATE_KEYWORD: str = "optimizer_state_dict"
CHECKPOINT_METADICT_KEYWORD: str = "META"


class OnLoadDataEvent(Event[Callable[[Any], None]]):
    def __call__(self, data: Any):
        super().__call__(data)


class OnTrainIterEvent(Event[Callable[[float, float], None]]):
    def __call__(self, avgLoss: float, learningRate: float) -> Any:
        super().__call__(avgLoss, learningRate)


class OnTrainEpochEvent(Event[Callable[[int], None]]):
    def __call__(self, epochIndex: int):
        super().__call__(epochIndex)


class OnTestIterEvent(
    Event[
        Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                List[CELImage],
                List[CELImage],
                float,
            ],
            None,
        ]
    ]
):
    def __call__(
        self,
        inputImage: torch.Tensor,
        gTruthImage: torch.Tensor,
        unetOutput: torch.Tensor,
        inputMeta: List[CELImage],
        gtruthMeta: List[CELImage],
        loss: float,
    ) -> None:
        super().__call__(
            inputImage, gTruthImage, unetOutput, inputMeta, gtruthMeta, loss
        )


class ModelWrapper:
    def __init__(
        self,
        model: nn.Module,
        optimiser: optim.Optimizer,
        lossFunction: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str,
    ):
        self._model = model
        self._optimiser = optimiser
        self._lossFunction = lossFunction

        self._epoch: int = 0
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        self._device: torch.device = "cpu"
        self.ToDevice(device)

        self.metaDict: Dict[str, Any] = {}

        self.OnLoadDataEvent = OnLoadDataEvent()

        self.OnTrainIter = OnTrainIterEvent()
        self.OnTrainEpoch = OnTrainEpochEvent()

        self.OnTestIter = OnTestIterEvent()

    def ToDevice(self, device: Union[str, torch.device]):
        device = torch.device(device)

        if device.type == "cpu":
            self._logger.info("Running on CPU")

        elif device.type == "cuda":

            if not torch.cuda.is_available():
                raise Exception("CUDA not available!")

            self._logger.info("Running on GPU device: " + str(device))

        else:
            raise Exception("Unable to load " + str(device))

        self._device = device
        self._model.to(self._device)

    # ! Might crash / misbehave on some optimisers
    def _ChangeLearningRate(self, learningRate: float):
        for paramGroup in self._optimiser.param_groups:
            paramGroup["lr"] = learningRate

    def Train(
        self,
        datasetLoader: DataLoader,
        trainToEpoch: int,
        learningRate: float,
    ):
        self._model.train()
        self._ChangeLearningRate(learningRate)

        dataLoadStartTime = time.time()

        for epochIndex in range(self._epoch, trainToEpoch):

            iterationIndex: int = 0

            for datasetIndex, data in enumerate(datasetLoader):

                dataLoadTime = time.time() - dataLoadStartTime
                self.OnLoadDataEvent(data)
                (
                    inputImage,
                    gTruthImage,
                    trainMeta,
                    truthMeta,
                ) = data

                inputImage.requires_grad = True

                modelProcessingStartTime = time.time()
                iterationIndex += 1

                self._optimiser.zero_grad()
                unetOutput = self._model(inputImage)

                loss = self._lossFunction(unetOutput, gTruthImage)
                loss.backward()
                self._optimiser.step()

                avgLoss = loss.item() / datasetLoader.batch_size

                self._logger.info(
                    "Epoch %d Iter %d avgLoss=%.3f ModelTime=%.3f DataTime=%.3f Lr=%f"
                    % (
                        epochIndex,
                        iterationIndex,
                        avgLoss,
                        time.time() - modelProcessingStartTime,
                        dataLoadTime,
                        learningRate,
                    )
                )

                self.OnTrainIter(avgLoss, learningRate)
                dataLoadStartTime = time.time()

            self._epoch += 1
            self.OnTrainEpoch(self._epoch)

    def Test(self, datasetLoader: DataLoader):
        self._model.eval()

        for datasetIndex, data in enumerate(datasetLoader):

            self.OnLoadDataEvent(data)
            (
                inputImage,
                gTruthImage,
                trainMeta,
                truthMeta,
            ) = data

            modelProcessingStartTime = time.time()

            unetOutput = self._model(inputImage)

            loss = self._lossFunction(unetOutput, gTruthImage).item()
            self.OnTestIter(
                inputImage, gTruthImage, unetOutput, trainMeta, truthMeta, loss
            )

            self._logger.info(
                "Image %d ModelTime=%.3f"
                % (datasetIndex, time.time() - modelProcessingStartTime)
            )

    def LoadWeights(
        self, file: str, strictWeightLoad: bool = True, loadOptimiser: bool = True
    ):

        # model should be transferred to device BEFORE checkpoint load
        self._model = self._model.to(self._device)

        if self._device.type == "cuda":
            for name, param in self._model.named_parameters():
                if param.device.type != "cuda":
                    raise Exception(
                        "param {}, not on GPU while model is GPU".format(name)
                    )

        checkpoint = torch.load(file, map_location=self._device)
        self._model.load_state_dict(
            checkpoint[CHECKPOINT_WEIGHTS_KEYWORD], strict=strictWeightLoad
        )

        if loadOptimiser:
            self._optimiser.load_state_dict(
                checkpoint[CHECKPOINT_OPTIMISER_STATE_KEYWORD]
            )

        if checkpoint[CHECKPOINT_EPOCH_KEYWORD] == 0:
            self._epoch = 0
        else:
            self._epoch = checkpoint[CHECKPOINT_EPOCH_KEYWORD]

        self.metaDict = checkpoint[CHECKPOINT_METADICT_KEYWORD]

        self._logger.info("Loaded model on epoch " + str(self._epoch))

    def Save(self, file):

        torch.save(
            {
                CHECKPOINT_EPOCH_KEYWORD: self._epoch,
                CHECKPOINT_WEIGHTS_KEYWORD: self._model.state_dict(),
                CHECKPOINT_OPTIMISER_STATE_KEYWORD: self._optimiser.state_dict(),
                CHECKPOINT_METADICT_KEYWORD: self.metaDict,
            },
            file,
        )
