import os

ROOT_FOLDER = os.path.abspath("../")
IMAGES_FOLDER = os.path.abspath(os.path.join("../../", "data"))
assert os.path.isdir(IMAGES_FOLDER), "Images folder not found at {}".format(IMAGES_FOLDER)

ORIGINAL_IMAGES = os.path.join(IMAGES_FOLDER, "images")
assert os.path.isdir(ORIGINAL_IMAGES), \
    "Original  folder not found at {}".format(ORIGINAL_IMAGES)
ORIGINAL_IMAGES_TRAINING = os.path.join(ORIGINAL_IMAGES, "training")
assert os.path.isdir(ORIGINAL_IMAGES_TRAINING), \
    "Original training folder not found at {}".format(ORIGINAL_IMAGES_TRAINING)
ORIGINAL_IMAGES_VALIDATION = os.path.join(ORIGINAL_IMAGES, "training")
assert os.path.isdir(ORIGINAL_IMAGES_VALIDATION), \
    "Original validation folder not found at {}".format(ORIGINAL_IMAGES_VALIDATION)


PROCESSED_IMAGES = os.path.join(IMAGES_FOLDER, "processed_images")
assert os.path.isdir(PROCESSED_IMAGES), \
    "Processed folder not found at {}".format(PROCESSED_IMAGES)
PROCESSED_IMAGES_TRAINING = os.path.join(PROCESSED_IMAGES, "training")
assert os.path.isdir(PROCESSED_IMAGES_TRAINING), \
    "Processed training folder not found at {}".format(PROCESSED_IMAGES_TRAINING)
PROCESSED_IMAGES_VALIDATION = os.path.join(PROCESSED_IMAGES, "training")
assert os.path.isdir(PROCESSED_IMAGES_TRAINING), \
    "Processed validation folder not found at {}".format(PROCESSED_IMAGES_TRAINING)

ALARMS_FOLDER = os.path.join(ROOT_FOLDER, "alarms")
assert os.path.isdir(ALARMS_FOLDER), "Alarms folder not found at {}".format(ALARMS_FOLDER)

ALARMS = [os.path.join(ALARMS_FOLDER, x) for x in os.listdir(ALARMS_FOLDER)]
assert len(ALARMS) > 3, "Not enough alarms provided in {}".format(ALARMS_FOLDER)

PREDICTOR_PATH = os.path.join(ROOT_FOLDER, "models", "shape_predictor_68_face_landmarks.dat")
assert os.path.isfile(PREDICTOR_PATH), "Landmark predictor not found at {}".format(PREDICTOR_PATH)

IMAGE_MODEL = os.path.join(ROOT_FOLDER, "models", "model.json")
IMAGE_MODEL_WEIGHTS = os.path.join(ROOT_FOLDER, "models", "model.h5")
IMAGE_MODEL_BEST_WEIGHTS = os.path.join(ROOT_FOLDER, "models", "model_best.h5")
IMAGE_MODEL_HISTORY = os.path.join(ROOT_FOLDER, "models", "history.p")
IMAGE_MODEL_BEST_HISTORY = os.path.join(ROOT_FOLDER, "models", "history_best.p")

MASK_MODEL = os.path.join(ROOT_FOLDER, "models", "model_mask.json")
MASK_MODEL_WEIGHTS = os.path.join(ROOT_FOLDER, "models", "model_mask.h5")
MASK_MODEL_BEST_WEIGHTS = os.path.join(ROOT_FOLDER, "models", "model_mask_best.h5")
MASK_MODEL_HISTORY = os.path.join(ROOT_FOLDER, "models", "history_mask.p")
MASK_MODEL_BEST_HISTORY = os.path.join(ROOT_FOLDER, "models", "history_mask_best.p")

PHYSICAL_WEBCAM = 0
IP_WEBCAM = "http://192.168.0.159:4747/video"

DEFAULT_WEBCAM = IP_WEBCAM

RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

IMAGE_DIMS = (200, 200, 3)

# does not resize
VIDEO_STREAM_DIMENSIONS = None
# VIDEO_STREAM_DIMENSIONS = (320, 180)
