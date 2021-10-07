import os

ROOT_FOLDER = '/data/johnathon/CVLR_venv'

DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')

DATA_LIST_FOLDER = os.path.join(ROOT_FOLDER, 'ucfTrainTestlist')

CLASS_LIST_TEXT_FILE = os.path.join(DATA_LIST_FOLDER, 'classInd.txt')

TRAIN_FOLDER_PATH = os.path.join(DATA_FOLDER, 'train')

TEST_FOLDER_PATH = os.path.join(DATA_FOLDER, 'test')

VAL_FOLDER_PATH = os.path.join(DATA_FOLDER, 'val')

BATCH_SIZE = 32

LENGTH_OF_CLIP = 16

RESIZED_FRAME = 224

DATALOADER_NUM_WORKERS = 3

# in run_model.py
SAVED_MODEL_CHECKPOINT_PATH = os.path.join(ROOT_FOLDER, 'saved_model/epoch_10_model.pt')