
import torchvision.models as models
def choosemodel(arch):
    if arch=='restnet18':
        return models.resnet18(pretrained=True)
    elif arch=='alexnet':
        return models.alexnet(pretrained=True)
    elif arch == 'vgg16':
        return models.vgg16(pretrained=True)
    elif arch == 'squeezenet':
        return models.squeezenet1_0(pretrained=True)
    elif arch == 'densenet':
        return models.densenet161(pretrained=True)
    elif arch == 'inception':
        return models.inception_v3(pretrained=True)
    else:
        return models.vgg16(pretrained=True)
