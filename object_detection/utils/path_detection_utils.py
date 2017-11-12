import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def get_segmentation(path, detection_graph, category_index, display_image = True):
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			image = Image.open(path)
			width, height = image.size
			IMAGE_SIZE = (width/96.0, height/96.0)
			image_np = load_image_into_numpy_array(image)
			image_np_expanded = np.expand_dims(image_np, axis=0)
			(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})
			if display_image:
				# Visualization of the results of a detection.
				vis_util.visualize_boxes_and_labels_on_image_array(
					image_np,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),
					category_index,
					use_normalized_coordinates=True,
					line_thickness=8)
				plt.figure(figsize=IMAGE_SIZE)
				plt.imshow(image_np)
	return boxes, scores, classes

def remove_non_human(boxes, scores, classes):
	human_idx = np.where(classes[0]==1)
	human_boxes = boxes[0][human_idx]
	human_scores = scores[0][human_idx]
	return human_boxes, human_scores

def remove_low_prob(boxes, scores, threshold = 0.25):
	good_idx = np.where(scores > threshold)
	good_boxes = boxes[good_idx]
	good_scores = scores[good_idx]
	return good_boxes, good_scores

def remove_poorly_sized_people(boxes, scores):
	return "well shit"