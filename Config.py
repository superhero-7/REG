class Config:
    # MODEL
    MODEL_DISABLE_CUDA = False

    # IMG_PROCESSING
    IMG_PROCESSING_USE_IMAGE = True

    # DATASET
    DATASET_IMG_ROOT = './images/train2014'  # path to the image directory
    DATASET_VAL_DATA_ROOT = './google_refexp_dataset_release/google_refexp_val_201511_coco_aligned.json'  # 这个一会再写
    DATASET_TRAIN_DATA_ROOT = './google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'
    IMG_PROCESSING_IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_PROCESSING_IMG_STD = [0.229, 0.224, 0.225]
    IMG_PROCESSING_TRANSFORM_SIZE = 224
    DATASET_VOCAB = 'vocab.txt'

    # TRAINING
    TRAINING_N_CONSTRAST_OBJECT = 5
    TRAINING_LEARNING_RATE = 0.01  # Adam Optimizer Learning Rate 这一块跟论文中的不太一样
    TRAINING_L2_FRACTION = 1e-5  # Adam Optimizer weight decay
    TRAINING_BATCH_SIZE = 16  # batch_size先弄成4吧，这块没办法...设备内存比较小
    TRAINING_DROPOUT = 0.5
    TRAINING_N_EPOCH = 300

    # OUTPUT
    OUTPUT_CHECKPOINT_PREFIX = 'full'

    # TEST
    TEST_DO_ALL = False

    # --- LSTM Variables ---#

    LSTM_HIDDEN = 1024  # Size of LSTM hidden layer if there is an LSTM in the network
    LSTM_EMBED = 1024  # Size of LSTM embedding if there is an LSTM in the network

    # --- Image feature network Variables ---#
    IMG_NET_FEATS = 2005
    IMG_NET_N_LABELS = 1000
    IMG_NET_FIX_WEIGHTS = list(range(40))  # Which layers to freeze weights for in image network

    DEBUG = False