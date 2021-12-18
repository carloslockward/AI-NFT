from os import listdir
from os.path import isfile, join
from pathlib import Path
import imagesize
import math


def calculate_aspect(width: int, height: int) -> str:
    r = math.gcd(width, height)
    x = int(width / r)
    y = int(height / r)
    return x, y

def get_file_list(drtr):
    return [f for f in listdir(drtr) if isfile(join(drtr, f))]

def get_avg_img_size(drtr):
    file_list = get_file_list(drtr)
    avg_x = 0
    avg_y = 0
    for file in file_list:
        try:
            x,y = imagesize.get(drtr + file)
        except Exception as e:
            raise Exception(f"Could not get image size for file: {file} \nException: {e}")
        
        avg_x += x
        avg_y += y
    return avg_x/len(file_list), avg_y/len(file_list)

def get_avg_aspect_ratio(drtr):

    file_list = get_file_list(drtr)
    avg_ratio = 0
    for file in file_list:

        try:
            width, height = imagesize.get(drtr + file)
        except Exception as e:
            raise Exception(f"Could not get image size for file: {file} \nException: {e}")

        avg_ratio += width/height

    return round(avg_ratio/len(file_list), 1).as_integer_ratio()

def get_default_device():
    if T.cuda.is_available():
        return T.device('cuda')
    else:
        return T.device('cpu')

