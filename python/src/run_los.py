# Specify device
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Import all necessary libraries.
import numpy as np
import sys
import matplotlib.image as mpimg
import glob

###
# --- отладка

# for filename in glob.glob('/usr/src/app/src/nomeroff-net/cars/*'):
#     print(filename)
# print("777")
# sys.exit()

# ---
###

# NomeroffNet path
# NOMEROFF_NET_DIR = os.path.abspath('../')
# NOMEROFF_NET_DIR = os.path.abspath('./NomeroffNet')
# NOMEROFF_NET_DIR = os.path.abspath('./')
NOMEROFF_NET_DIR = os.path.abspath('./src/nomeroff-net/')
sys.path.append(NOMEROFF_NET_DIR)

###
# --- отладка

# # centermask2_path= os.path.join(nomeroffnet_path, "centermask2")
# centermask2_path= os.path.join(NOMEROFF_NET_DIR, "centermask2")
# print(centermask2_path)
# sys.path.append(centermask2_path)
# from centermask.config import get_cfg
#
# print(1)
#
# sys.exit()

# ---
###

# Import license plate recognition tools.
from NomeroffNet import Detector
from NomeroffNet import filters
from NomeroffNet import RectDetector
from NomeroffNet import OptionsDetector
from NomeroffNet import TextDetector
from NomeroffNet import textPostprocessing

# load models
rectDetector = RectDetector()

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

# textDetector = TextDetector.get_static_module("eu")()
textDetector = TextDetector.get_static_module("ru")()
textDetector.load("latest")

nnet = Detector()
nnet.loadModel(NOMEROFF_NET_DIR)

# Detect numberplate
# img_path = 'images/example2.jpeg'
# img_path = '/usr/src/app/src/nomeroff-net/examples/images/example2.jpeg'
# img_path = '/usr/src/app/src/nomeroff-net/examples/images/example3.jpg'

for filename in glob.glob('/usr/src/app/src/nomeroff-net/cars/*'):
    print(filename)

    img = mpimg.imread(filename)

    # Generate image mask.
    cv_imgs_masks = nnet.detect_mask([img])

    for cv_img_masks in cv_imgs_masks:
        # Detect points.
        arrPoints = rectDetector.detect(cv_img_masks)

        # cut zones
        zones = rectDetector.get_cv_zonesBGR(img, arrPoints, 64, 295)

        # find standart
        regionIds, stateIds, countLines = optionsDetector.predict(zones)
        regionNames = optionsDetector.getRegionLabels(regionIds)

        # find text with postprocessing by standart
        textArr = textDetector.predict(zones)
        textArr = textPostprocessing(textArr, regionNames)
        print(textArr)
        # ['JJF509', 'RP70012']