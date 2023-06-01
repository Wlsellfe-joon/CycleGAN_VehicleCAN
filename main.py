import os
import matplotlib.pyplot as plt

from Cycle_model import CycleGAN
from Loader import DataLoader

# run params
model_path = 'C:/~Runfolder~/Model/'

mode = 'build' # 'build' #
IMAGE_row = 100
IMAGE_col = 80
DATA_NAME = 'CAN_RGBA_img'


data_loader = DataLoader(dataset_name=DATA_NAME, img_res=(IMAGE_row, IMAGE_col))
gan = CycleGAN(
        input_dim = (IMAGE_row,IMAGE_col,4)
        , learning_rate = 0.0002
        , lambda_validation = 1
        , lambda_reconstr = 10
        , lambda_id = 5
        , generator_type = 'resnet'
        , gen_n_filters = 32
        , disc_n_filters = 64
        )

if mode == 'build':
    gan.save(model_path)
else:
    gan.load_weights(os.path.join(model_path, 'weights/weights.h5'))

#gan.d_A.summary()       # model
#gan.d_B.summary()       # model 1
#gan.g_AB.summary()      # model 2
#gan.g_BA.summary()      # model 3




# Training
BATCH_SIZE = 1
EPOCHS = 100  # 100 회 학습함
PRINT_EVERY_N_BATCHES = 10

TEST_A_FILE = 'DoS_3510.png'
TEST_B_FILE = 'Normal_3510.png'

gan.train(data_loader
        , run_folder = model_path
        , epochs=EPOCHS
        , test_A_file = TEST_A_FILE
        , test_B_file = TEST_B_FILE
        , batch_size=BATCH_SIZE
        , sample_interval=PRINT_EVERY_N_BATCHES)
