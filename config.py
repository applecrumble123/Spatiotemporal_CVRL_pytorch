import os

ROOT_FOLDER = '/data/johnathon/CVLR_venv'

DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')

DATA_LIST_FOLDER = os.path.join(ROOT_FOLDER, 'ucfTrainTestlist')

CLASS_LIST_TEXT_FILE = os.path.join(DATA_LIST_FOLDER, 'classInd.txt')

#TRAIN_FOLDER_PATH = os.path.join(DATA_FOLDER, 'train')

#TEST_FOLDER_PATH = os.path.join(DATA_FOLDER, 'test')

#VAL_FOLDER_PATH = os.path.join(DATA_FOLDER, 'val')

NUM_OF_EPOCH = 60

BATCH_SIZE = 32

LENGTH_OF_CLIP = 16

RESIZED_FRAME = 224

DATALOADER_NUM_WORKERS = 3

CONTRASTIVE_LOSS_TEMP = 2

SAVED_MODEL_FOLDER = os.path.join(ROOT_FOLDER, 'saved_model_jcc')

TENSORBOARD_ROOT_LOGDIR = os.path.join(ROOT_FOLDER,'tensorboard_logs')


# ---------------- Contrastive Learning Tensorboard ------------------
TENSORBOARD_CONTRASTIVE_LEARNING = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'contrastive_learning')

TENSORBOARD_CONTRASTIVE_LEARNING_HIGH_TEMP = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'contrastive_learning_high_temp')

TENSORBOARD_CONTRASTIVE_LEARNING_LOW_TEMP = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'contrastive_learning_low_temp')


# ---------------- JVS Tensorboard ------------------
TENSORBOARD_JVS = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jvs')

TENSORBOARD_JVS_AND_CONTRASTIVE_LEARNING = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jvs_and_contrastive_learning')

TENSORBOARD_JVS_AND_CL_NO_TEMP = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jvs_and_cl_no_temp')

TENSORBOARD_JVS_AND_CONTRASTIVE_LEARNING_LOW_TEMP = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jvs_and_cl_low_temp')

TENSORBOARD_JVS_AND_CONTRASTIVE_LEARNING_HIGH_TEMP = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jvs_and_cl_high_temp')


# ---------------- JCC Tensorboard ------------------
TENSORBOARD_JCC = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jcc')

TENSORBOARD_JCC_LOW_TEMP = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jcc_low_temp')

TENSORBOARD_JCC_HIGH_TEMP = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jcc_high_temp')

TENSORBOARD_JCC_AND_CONTRASTIVE_LEARNING = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jcc_and_contrastive_learning')

TENSORBOARD_JCC_AND_CONTRASTIVE_LEARNING_LOW_TEMP = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jcc_and_cl_low_temp')

TENSORBOARD_JCC_AND_CONTRASTIVE_LEARNING_HIGH_TEMP = os.path.join(TENSORBOARD_ROOT_LOGDIR, 'jcc_and_cl_high_temp')

MULTIBATCH_UPDATE = True


# ------------ Change when running run_model.py ---------------

TRAIN_FEATURES_PATH = os.path.join(ROOT_FOLDER, 'train_features')

X_TRAIN_PKL_PATH = os.path.join(TRAIN_FEATURES_PATH, 'x_train_jcc.pkl')

Y_TRAIN_PKL_PATH = os.path.join(TRAIN_FEATURES_PATH, 'y_train_jcc.pkl')

X_TRAIN_NUMPY_PKL_PATH = os.path.join(TRAIN_FEATURES_PATH, 'x_train_numpy_jcc.pkl')


TEST_FEATURES_PATH = os.path.join(ROOT_FOLDER, 'test_features')

X_TEST_PKL_PATH = os.path.join(TEST_FEATURES_PATH, 'x_test_jcc.pkl')
 
Y_TEST_PKL_PATH = os.path.join(TEST_FEATURES_PATH, 'y_test_jcc.pkl')

X_TEST_NUMPY_PKL_PATH = os.path.join(TEST_FEATURES_PATH, 'x_test_numpy_jcc.pkl')


SAVED_MODEL_CHECKPOINT_PATH = os.path.join(SAVED_MODEL_FOLDER, 'highest_val_acc_model.pt')
