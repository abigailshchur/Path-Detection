import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


import cv2

vidcap = cv2.VideoCapture('test_videos/testclip.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  cv2.imwrite("video_frames/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1
  #if (count > 500):
  #      success = False