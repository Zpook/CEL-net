import torch.nn as nn
import copy

from networks.blocks import AdaptiveFM


class BaseAdanet(nn.Module):
    def __init__(self) -> None:
        self._originalWeights = {}
        self._tuneMode = False
        super().__init__()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, 0.02)
                if module.bias is not None:
                    module.bias.data.normal_(0.0, 0.02)
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(0.0, 0.02)

        for name, tensor in self.named_parameters():
            if name.find(AdaptiveFM.PARAM_NAME) >= 0:
                tensor.data.zero_()

        for name, tensor in self.named_parameters():
            if name.find("const") >= 0:
                tensor.data.normal_(1,0.1)
            if name.find("exp") >= 0:
                tensor.data.normal_(1,0.01)

    def TuningMode(self):
        """
            Sets tune mode and returns model parameters to be used by optimiser.
        """
        activate = True

        self._tuneMode = activate
        returnedParams = []

        for name, tensor in self.named_parameters():

            if not activate:
                returnedParams.append(tensor)

            tensor.requires_grad = not activate

            if name.find(AdaptiveFM.PARAM_NAME) >= 0:

                tensor.requires_grad = activate

                if activate:
                    returnedParams.append(tensor)

        return returnedParams

    def InterpolateWeights(self, coefficient: float):

        interp_dict = copy.deepcopy(self.state_dict())

        for name, tensor in interp_dict.items():

            if name.find(AdaptiveFM.PARAM_NAME) >= 0:
                interp_dict[name] = tensor * coefficient

        return interp_dict

    def InterpolateAndLoadWeights(self, coefficient: float):
        if self._originalWeights == {}:
            self._originalWeights = copy.deepcopy(self.state_dict())
        else:
            self.load_state_dict(self._originalWeights, strict=True)

        interp_dict = self.InterpolateWeights(coefficient)
        self.load_state_dict(interp_dict, strict=True)

    def train(self, mode: bool = True):
        super().train(mode)

        if self._tuneMode:
            self.TuningMode()
