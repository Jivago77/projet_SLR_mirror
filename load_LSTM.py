import torch
import torchvision.models as models


def load_LSTM():
    # we do not specify pretrained=True, i.e. do not load default weights
    model = models.vgg16()
    model.load_state_dict(torch.load('actionNN.hpth'))
    model.eval()

    return model
