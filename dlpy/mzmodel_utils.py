import torch
from torchvision import models
from dlpy.utils import *

def convert_torchvision_model_weights(model_name, cpp_model_path, python_model_path=None, imsize=(256, 256)):
    '''
    Converts a pretrained model in Python into model weights that can be loaded by C++ programs,
    and used by the MZModel() class.

    Parameters
    ----------
    model_name : string
        Specifies the name of the model.
    python_model_path : string
        Specifies the directory that contains the Python based model weights.
    cpp_model_path : string
        Specifies the derectory to store the C++ based model weights.
    imsize : tuple
        Specifies the dimension of the image passed to the trace function.
    '''
    try:
        temp = getattr(models, model_name)
        model = temp(pretrained=True)

        if python_model_path:
            python_weights = torch.load(python_model_path)
            model.load_state_dict(python_weights)
    except:
        raise DLPyError('Something is wrong while loading the pretrained weights.')

    print('NOTE: Pretrained weights loaded successfully.')

    class wrapped_model(torch.nn.Module):
        def __init__(self)->None:
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

    try:
        model_ = wrapped_model()
        img = torch.rand([1, 3, imsize[0], imsize[1]]).float()
        cpp_model = torch.jit.script(model_, img)
        cpp_model.save(cpp_model_path)
    except:
        raise DLPyError('Something is wrong while saving the pretrained weights.')

    print('NOTE: Pretrained weights saved successfully.')

def convert_yolov5_model_weights(cpp_model_path, python_model_path, imsize=(640, 480)):
    '''
    Converts a pretrained model in Python into model weights that can be loaded by C++ programs,
    and used by the MZModel() class.

    Parameters
    ----------
    python_model_path : string
        Specifies the directory that contains the Python based model weights.
    cpp_model_path : string
        Specifies the derectory to store the C++ based model weights.
    imsize : tuple
        Specifies the dimension of the image passed to the trace function.
    '''
    try:
        device = torch.device('cpu')
        ckpt = torch.load(python_model_path, map_location=device)
    except:
        raise DLPyError('Something is wrong while loading the pretrained weights.')

    print('NOTE: Pretrained weights loaded successfully.')

    model = []
    model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())
    model = model[-1]

    imsize_md = max(imsize[0], imsize[1])
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imsize_md, imsize_md).to(device).type_as(next(model.parameters())))

    img = torch.rand([1, 3, imsize[0], imsize[1]]).to(device).float()

    try:
        pred = model(img, augment=False)[0]
        cpp_model = torch.jit.trace(model, img)
        cpp_model.save(cpp_model_path)
    except:
        raise DLPyError('Something is wrong while saving the pretrained weights.')

    print('NOTE: Pretrained weights saved successfully.')