import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
import visualization_utils as vis_util
from matplotlib.patches import Circle

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

"""
Removes bounding boxes that are not classified as people
"""
def remove_non_human(boxes, scores, classes):
	human_idx = np.where(classes[0]==1)
	human_boxes = boxes[0][human_idx]
	human_scores = scores[0][human_idx]
	return human_boxes, human_scores

"""
Removes bounding boxes with low probabilities
"""
def remove_low_prob(boxes, scores, threshold = 0.40):
	good_idx = np.where(scores > threshold)
	good_boxes = boxes[good_idx]
	good_scores = scores[good_idx]
	return good_boxes, good_scores

"""
Ensures that people are rectangles such that height > width
"""
def remove_poorly_sized_people(boxes, scores):
	good_boxes = []
	good_scores = []
	for i in range(len(boxes)):
		if abs(boxes[i][3]-boxes[i][1]) < abs(boxes[i][2]-boxes[i][0]):
			good_boxes.append(boxes[i])
			good_scores.append(scores[i])
	return good_boxes, good_scores


"""
Returns a similarity score between two boxes
box1: [y_coord (top left), x_coord (top left), y_coord (bottom right), x_coord (bottom_right)]

Method 1 - average Euclidean distance between the two 
"""
def get_box_similarity_score(box1, box2, score1, score2, img_w, img_h, method=1):
    if (method == 1):
        d1 = np.array([box1[1],box1[0]]) - np.array([box2[1],box2[0]])
        d1 = (d1[0]*d1[0]+d1[1]*d1[1])**(0.5)
        d2 = np.array([box1[3],box1[2]]) - np.array([box2[3],box2[2]])
        d2 = (d2[0]*d2[0]+d2[1]*d2[1])**(0.5)
        return (d1+d2)/2.0
    else:
        return 0


def draw_box(box, width, height, image_np):
    x = [box[1]*width, box[3]*width]
    y = [box[0]*height, box[2]*height]
    # Create a figure. Equal aspect so circles look circular
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    # Show the image
    ax.imshow(image_np)
    # Now, loop through coord arrays, and create a circle at each x,y pair
    for xx,yy in zip(x,y):
        circ = Circle((xx,yy),50)
        ax.add_patch(circ)
    # Show the image
    plt.show()

#def clasify_box(box):
