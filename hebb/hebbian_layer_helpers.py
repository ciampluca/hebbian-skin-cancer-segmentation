import torch


def to_2dvector(param, param_name):
    if type(param) == int:
        converted = (param, param)
    elif type(param) == list:
        if len(param) != 2:
            raise AttributeError(f'When {param_name} is list it should be of lenght 2 but it was {len(param)}')
        converted = tuple(param)
    elif type(param) == tuple:
        if (len(param)) != 2:
            raise AttributeError(f'When {param_name} is tuple it should be of length 2 but it was '
                                 f'{len(param)}')
        converted = param
    else:
        raise TypeError(f'{param_name} type {type(param)} is not supported. The supported types are'
                        f'int, list, and typle')

    return torch.tensor(converted)
