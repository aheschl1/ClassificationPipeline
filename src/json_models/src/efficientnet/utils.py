from torch import nn


def create_stage(n_in, n_out, num_layers, layer_type,
                 conv_op=nn.Conv2d, conv_args=None,
                 kernel_size=3, stride=1, r=24, p=0):
    """Creates a Sequential consisting of [num_layers] layer_type"""
    if conv_args is None:
        conv_args = {}
    layers = [layer_type(n_in, n_out, conv_op, conv_args, kernel_size=kernel_size, stride=stride, r=r, p=p)]
    layers += \
        [layer_type(n_out, n_out, conv_op, conv_args, kernel_size=kernel_size, r=r, p=p) for _ in range(num_layers - 1)]
    layers = nn.Sequential(*layers)
    return layers


def scale_width(w, w_factor):
    """Scales width given a scale factor"""
    w *= w_factor
    new_w = (int(w + 4) // 8) * 8
    new_w = max(8, new_w)
    if new_w < 0.9 * w:
        new_w += 8
    return int(new_w)
