import torch
import models.densenet as densenet
import models.efficientnet as efficientnet
import argparse


device = ('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(model_type: str,
                device :str, 
                pretrained: bool=False, 
                progress : bool=False) -> [torch.nn.Module, torch.nn.Module]:
    dense_model_names = sorted(
        name for name in densenet.__dict__ if name.islower() and not name.startswith("__") and callable(densenet.__dict__[name]))

    efficient_model_names = sorted(
        name for name in efficientnet.__dict__ if name.islower() and not name.startswith("__") and callable(efficientnet.__dict__[name]))

    model_ema_decay = 0.99998
    if model_type in dense_model_names:
        model = densenet.__dict__[model_type](num_classes=2).to(device, memory_format=torch.channels_last)
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - model_ema_decay) * averaged_model_parameter + model_ema_decay * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
        print (f'{model_type}, Model build successfully!!!!!')
        print(ema_avg)
    elif model_type in efficient_model_names:
        model = efficientnet.__dict__[model_type](num_classes=2, pretrained=pretrained, progress=progress).to(device, memory_format=torch.channels_last)
        print (f'{model_type}, Model build successfully!!!!!')
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - model_ema_decay) * averaged_model_parameter + model_ema_decay * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
        print (f'{model_type}, Model build successfully!!!!!')
        print(ema_avg)
    return model, ema_model


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Choose model name")
    parser.add_argument("-p", "--pretrained", action='store_true', help="Pretrained")
    parser.add_argument("-pr", "--progress", action='store_true', help="Progress")
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = arg_parser()
    print('Pretrained, ', args['pretrained'])
    model, ema_model = build_model(args['model'], device, args['pretrained'], args['progress'])
 #   print(ema_model)
    print(model)
    print('done')

#model, total_params = utils.build_model(device, version=model_name)

