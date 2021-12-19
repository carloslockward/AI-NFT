from math import floor


def calculate_2d_conv_out(input_size, padding, kernel, stride):
    return floor(((input_size + (2 * padding) - kernel) / stride) + 1)


def calculate_2d_conv_trans_out(input_size, padding, kernel, stride):
    return (input_size - 1) * stride - (2 * padding) + kernel
