import torch

DATA_PATH = './data'
BATCH_SIZE = 16
INPUT_SIZE = 128
NUMBER_OF_CLASSES = 6

NUMBER_OF_EPOCHS = 20
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"