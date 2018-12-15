# To generate the datasets as .npy, you must have a folder 'datasets' two levels up the data directory
# The datasets folder must contain the CK+, FERPlus and JAFFE datasets as provided by its creators (with no changes in files, folders etc)

# JAFFE
JAFFE_DATASET_PATH = '../../datasets/jaffe/'

JAFFE_EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

SAVE_DATASET_JAFFE_IMAGES_FILENAME = 'jaffe/data_set_jaffe.npy'
SAVE_DATASET_JAFFE_LABELS_FILENAME = 'jaffe/data_labels_jaffe.npy'
SAVE_DATASET_JAFFE_IMAGES_TEST_FILENAME = 'jaffe/test_set_jaffe.npy'
SAVE_DATASET_JAFFE_LABELS_TEST_FILENAME = 'jaffe/test_labels_jaffe.npy'

# CK+
CK_EXTENDED_DATASET_PATH = '../../datasets/extended-cohn-kanade/cohn-kanade-images/'
CK_EXTENDED_LABELS_PATH = '../../datasets/extended-cohn-kanade/emotion-labels/'

CK_EXTENDED_EMOTIONS = ['neutral', 'angry', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

SAVE_DATASET_CK_EXTENDED_IMAGES_FILENAME = 'ck/data_set_ck_extended.npy'
SAVE_DATASET_CK_EXTENDED_LABELS_FILENAME = 'ck/data_labels_ck_extended.npy'
SAVE_DATASET_CK_EXTENDED_IMAGES_TEST_FILENAME = 'ck/test_set_ck_extended.npy'
SAVE_DATASET_CK_EXTENDED_LABELS_TEST_FILENAME = 'ck/test_labels_ck_extended.npy'

# Same dataset but not resizing the images to 112x150
SAVE_DATASET_CK_EXTENDED_NO_RESIZE_IMAGES_FILENAME = 'ck/data_set_ck_extended_no_resize.npy'
SAVE_DATASET_CK_EXTENDED_NO_RESIZE_LABELS_FILENAME = 'ck/data_labels_ck_extended_no_resize.npy'
SAVE_DATASET_CK_EXTENDED_NO_RESIZE_IMAGES_TEST_FILENAME = 'ck/test_set_ck_extended_no_resize.npy'
SAVE_DATASET_CK_EXTENDED_NO_RESIZE_LABELS_TEST_FILENAME = 'ck/test_labels_ck_extended_no_resize.npy'

# FER2013
FER_2013_DATASET_PATH = '../../datasets/FERPlus/'
FER_2013_TEST_FOLDER = 'FER2013Test/'
FER_2013_TRAIN_FOLDER = 'FER2013Train/'

FER_2013_EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

SAVE_DATASET_FER_2013_IMAGES_FILENAME = 'fer2013/data_set_fer_2013.npy'
SAVE_DATASET_FER_2013_LABELS_FILENAME = 'fer2013/data_labels_fer_2013.npy'
SAVE_DATASET_FER_2013_IMAGES_TEST_FILENAME = 'fer2013/test_set_fer_2013.npy'
SAVE_DATASET_FER_2013_LABELS_TEST_FILENAME = 'fer2013/test_labels_fer_2013.npy'

SAVE_DATASET_FER_2013_IMAGES_MEDIUM_FILENAME = 'fer2013/data_set_medium_fer_2013.npy'
SAVE_DATASET_FER_2013_LABELS_MEDIUM_FILENAME = 'fer2013/data_labels_medium_fer_2013.npy'
SAVE_DATASET_FER_2013_IMAGES_TEST_MEDIUM_FILENAME = 'fer2013/test_set_medium_fer_2013.npy'
SAVE_DATASET_FER_2013_LABELS_TEST_MEDIUM_FILENAME = 'fer2013/test_labels_medium_fer_2013.npy'

SAVE_DATASET_FER_2013_IMAGES_SMALL_FILENAME = 'fer2013/data_set_small_fer_2013.npy'
SAVE_DATASET_FER_2013_LABELS_SMALL_FILENAME = 'fer2013/data_labels_small_fer_2013.npy'
SAVE_DATASET_FER_2013_IMAGES_TEST_SMALL_FILENAME = 'fer2013/test_set_small_fer_2013.npy'
SAVE_DATASET_FER_2013_LABELS_TEST_SMALL_FILENAME = 'fer2013/test_labels_small_fer_2013.npy'