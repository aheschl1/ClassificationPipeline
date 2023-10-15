import importlib
import torchvision.models as models
from src.utils.constants import *


def my_import(class_name: str, dropout_package: str = 'torch.nn'):
    """
    Returns a class based on a string name.
    :param class_name: The name of the object being searched for.
    :param dropout_package: The package to look into of the class isn't in the if/else statements
    :return: Any object defined in this long if/else sequence.
    """
    # Here checks backbones
    if class_name == ENB6:
        return models.efficientnet_b6
    elif class_name == ENV2:
        return models.efficientnet_v2_l
    elif class_name == ENB4:
        return models.efficientnet_b4
    elif class_name == ENB0:
        return models.efficientnet_b0
    elif class_name == ENB1:
        return models.efficientnet_b1
    # Here checks modules
    if class_name == 'UpsamplingConv':
        from src.json_models.src.modules import UpsamplingConv
        return UpsamplingConv
    elif class_name == 'ConvPixelShuffle':
        from src.json_models.src.modules import ConvPixelShuffle
        return ConvPixelShuffle
    elif class_name == 'SelfAttention':
        from src.json_models.src.modules import SelfAttention
        return SelfAttention
    elif class_name == 'SpatialAttentionModule':
        from src.json_models.src.modules import SpatialAttentionModule
        return SpatialAttentionModule
    elif class_name == 'DepthWiseSeparableConv':
        from src.json_models.src.modules import DepthWiseSeparableConv
        return DepthWiseSeparableConv
    elif class_name == "Residual":
        from src.json_models.src.modules import Residual
        return Residual
    elif class_name == "Linker":
        from src.json_models.src.modules import Linker
        return Linker
    elif class_name == "XModule":
        from src.json_models.src.modules import XModule
        return XModule
    elif class_name == "CBAM":
        from src.json_models.src.modules import CBAM
        return CBAM
    elif class_name == "CBAMResidual":
        from src.json_models.src.modules import CBAMResidual
        return CBAMResidual
    elif class_name == "CAM":
        from src.json_models.src.modules import CAM
        return CAM
    elif class_name == "LearnableCAM":
        from src.json_models.src.modules import LearnableCAM
        return LearnableCAM
    elif class_name == "InstanceNorm":
        from src.json_models.src.modules import InstanceNorm
        return InstanceNorm
    elif class_name == "BatchNorm":
        from src.json_models.src.modules import BatchNorm
        return BatchNorm
    elif class_name == "ConvTranspose":
        from src.json_models.src.modules import ConvTranspose
        return ConvTranspose
    elif class_name == "Conv":
        from src.json_models.src.modules import Conv
        return Conv
    elif class_name == "SpatialGatedConv2d":
        from src.json_models.src.modules import SpatialGatedConv2d
        return SpatialGatedConv2d
    elif class_name == "AveragePool":
        from src.json_models.src.modules import AveragePool
        return AveragePool
    elif class_name == "ReverseLinearBottleneck":
        from src.json_models.src.modules import ReverseLinearBottleneck
        return ReverseLinearBottleneck
    else:
        try:
            module = importlib.import_module(dropout_package)
            class_ = getattr(module, class_name)
            return class_
        except AttributeError:
            raise NotImplementedError(f'The requested module {class_name} has not been placed in my_import, and is '
                                      f'not in torch.nn.')
