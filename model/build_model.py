import torch.nn as nn
from .all_models import get_model, modify_last_layer
from .model_resnet import ResNet18, ResNet34

def build_model(args):
    # choose different Neural network model for different args
    model = get_model(args.model, args.pretrained)
    model, _ = modify_last_layer(args.model, model, args.n_classes)
    # if args.model == 'Resnet18':
    #     model = ResNet18(args.n_classes)
    # elif args.model == 'Resnet34':
    #      model = ResNet34(args.n_classes)
    model = model.to(args.device)

    return model

