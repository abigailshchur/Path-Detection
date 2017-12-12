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
from PIL import Image, ImageChops


import cv2

vidcap = cv2.VideoCapture('test_videos/M_2.mp4')
success,image = vidcap.read()
count = 0
success = True
box = None
while success:
  success,image = vidcap.read()
  cv2.imwrite("video_frames/frame%d.jpg" % count, image)     # save frame as JPEG file
  img = Image.open("video_frames/frame%d.jpg" % count)
  if box == None:
  	bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
  	diff = ImageChops.difference(img, bg)
  	diff = ImageChops.add(diff, diff, 2.0, -100)
  	box = diff.getbbox()
  img = img.crop(box)
  cv2.imwrite("video_frames/frame%d.jpg" % count, np.asarray( img, dtype="int32" ))
  count+=1
