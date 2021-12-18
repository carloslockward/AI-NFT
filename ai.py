from tqdm import tqdm
import torch as T

def get_default_device():
    if T.cuda.is_available():
        return T.device('cuda')
    else:
        return T.device('cpu')

