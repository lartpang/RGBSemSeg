import torch.nn as nn


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)


def group_weight(module, lr, mode="std"):
    weight_group = []
    if mode == "std":
        group_decay = []
        group_no_decay = []
        count = 0
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(
                    m,
                    (
                            nn.Conv1d,
                            nn.Conv2d,
                            nn.Conv3d,
                            nn.ConvTranspose2d,
                            nn.ConvTranspose3d,
                    ),
            ):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif (
                    isinstance(m, nn.BatchNorm1d)
                    or isinstance(m, nn.BatchNorm2d)
                    or isinstance(m, nn.BatchNorm3d)
                    or isinstance(m, nn.SyncBatchNorm)
                    or isinstance(m, nn.GroupNorm)
                    or isinstance(m, nn.LayerNorm)
            ):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.Parameter):
                group_decay.append(m)

        assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
        weight_group.append(dict(params=group_decay, lr=lr))
        weight_group.append(dict(params=group_no_decay, weight_decay=0.0, lr=lr))
    elif mode == "finetune":
        assert hasattr(module.backbone, "get_grouped_params")
        param_group = module.backbone.get_grouped_params()
        weight_group.append(
            dict(name="pretrained", params=param_group["pretrained"], lr=lr * 0.1)
        )
        weight_group.append(
            dict(name="retrained", params=param_group["retrained"], lr=lr)
        )
    else:
        raise NotImplementedError(mode)
    return weight_group
