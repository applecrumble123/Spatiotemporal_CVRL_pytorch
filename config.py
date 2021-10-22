import os

ROOT_FOLDER = '/data/johnathon/CVLR_venv'

DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')

DATA_LIST_FOLDER = os.path.join(ROOT_FOLDER, 'ucfTrainTestlist')

CLASS_LIST_TEXT_FILE = os.path.join(DATA_LIST_FOLDER, 'classInd.txt')

#TRAIN_FOLDER_PATH = os.path.join(DATA_FOLDER, 'train')

#TEST_FOLDER_PATH = os.path.join(DATA_FOLDER, 'test')

#VAL_FOLDER_PATH = os.path.join(DATA_FOLDER, 'val')

BATCH_SIZE = 32

LENGTH_OF_CLIP = 16

RESIZED_FRAME = 224

DATALOADER_NUM_WORKERS = 3

CONTRASTIVE_LOSS_TEMP = 0.5

SAVED_MODEL_FOLDER = os.path.join(ROOT_FOLDER, 'saved_model_cl_only')

TENSORBOARD_ROOT_LOGDIR = os.path.join(ROOT_FOLDER,'tensorboard_logs')

TENSORBOARD_CONTRASTIVE_LEARNING = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'contrastive_learning')

TENSORBOARD_JVS_AND_CONTRASTIVE_LEARNING = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jvs_and_contrastive_learning')

TENSORBOARD_JVS = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jvs')


# in run_model.py

TRAIN_FEATURES_PATH = os.path.join(ROOT_FOLDER, 'train_features')

X_TRAIN_PKL_PATH = os.path.join(TRAIN_FEATURES_PATH, 'x_train_cl_only.pkl')
 
Y_TRAIN_PKL_PATH = os.path.join(TRAIN_FEATURES_PATH, 'y_train_cl_only.pkl')


TEST_FEATURES_PATH = os.path.join(ROOT_FOLDER, 'test_features')

X_TEST_PKL_PATH = os.path.join(TEST_FEATURES_PATH, 'x_test_cl_only.pkl')
 
Y_TEST_PKL_PATH = os.path.join(TEST_FEATURES_PATH, 'y_test_cl_only.pkl')


SAVED_MODEL_CHECKPOINT_PATH = os.path.join(SAVED_MODEL_FOLDER, 'epoch_29_model.pt')
