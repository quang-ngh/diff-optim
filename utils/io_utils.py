
import pyrallis
import torch
from torchvision.utils import  make_grid, save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import *
from configs.config import *
import pickle

def extract_from_pickle(pkl_file):

    obj = None
    with open(pkl_file, mode = 'rb') as reader:
        obj = pickle.load(reader, encoding='latin1')
    reader.close()
    return obj


def write_json(obj, json_file):

    with open(json_file, mode = 'w') as writer:
        json.dump(obj, writer)
    writer.close()
    