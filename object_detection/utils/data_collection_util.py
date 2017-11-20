import cv2
import numpy as np
import path_detection_utils as path_util

def collect_data(path, start_frame, end_frame, detection_graph, category_index):
	in_frame = [] # ids of people in frame at current time step
	available_people = range(1000000)[::-1] #queue of people ids
	person_to_box = {} # maps person id -> box
	person_to_label = {} # maps person id -> label
	magic_number = 0.05 # the most magical of magical numbers
	all_data = [] # will be pandas matrix

	for i in range(start_frame, end_frame + 1):
		filename = path + str(i) + '.jpg'
		# next file is probably needed for optical flow
		next_file = path + str(i+1) + '.jpg' if i < end_frame else "end"
		# flow should contain optical flow matrix?
		flow = get_optical_flow(filename, next_file)

		# running image segmentation on current frame
		boxes, scores, classes = path_util.get_segmentation(filename,detection_graph, category_index, False)
		boxes, scores = path_util.remove_non_human(boxes, scores, classes)
		boxes, scores = path_util.remove_low_prob(boxes, scores)
		boxes, scores = path_util.remove_poorly_sized_people(boxes, scores)

		# getting similarity scores between in_frame people and people in current frame
		sim_scores = []
		for j in range(len(in_frame)):
			box_j = person_to_box[in_frame[j]]
			for k in range(len(boxes)):
				# sim score - list of 3-tuples (person_id from in_frame, person index in boxes, similarity score)
				sim_scores.append((in_frame[j], k, path_util.get_box_similarity_score(box_j, boxes[k],0,0,0,0)))

		# sort sim_scores (lower sim score is better)
		sim_scores = sorted(sim_scores, key=lambda x: x[2])
		sim_scores = [i for i in sim_scores if i[2] < magic_number]

		#match the good stuff in sim_scores
		matched_old = [] # what in in_frame has been matched (stores person id)
		matched_new = [] # what in the current frame has been matched (stores index in boxes)
		for score in sim_scores:
			if score[0] not in matched_old and score[1] not in matched_new:
				person_to_box[score[0]] = boxes[score[1]]
				matched_old.append(score[0])
				matched_new.append(score[1])

		# remove box details for in_frame people that were not matched
		for person in in_frame:
			if person not in matched_old:
				del person_to_box[person]

		# add people in current frame that are new
		for j in range(len(boxes)):
			if j not in matched_new:
				new_person = available_people.pop()
				matched_old.append(new_person)
				person_to_box[new_person] = boxes[j]

		# update who is in frame
		in_frame = matched_old

		# update necessary labels
		for person in in_frame:
			label = classify_box(person_to_box[person])
			person_to_label[person] = label

		# calculate optical flow per person
		in_frame_flows = [get_optical_flow_vector(flow, person_to_box[person]) for person in in_frame]

		# save everything to all_data
		for j in range(len(in_frame)):
			data_entry = {}
			data_entry["file"] = filename
			data_entry["frame_number"] = i
			data_entry["person_id"] = in_frame[j]
			data_entry["box"] = person_to_box[in_frame[j]]
			data_entry["flow"] = in_frame_flows[j]
			data_entry["label"] = person_to_label[in_frame[j]]
			all_data.append(data_entry)
	return all_data, person_to_label

"""
Provides label for a box
box = [y coord (top left), x coord (top left), y coord (bottom right), x coord (bottom right)]
"""
def classify_box(box):
	x_t_l = box[1]
	y_t_l = box[0]
	x_b_r = box[3]
	y_b_r = box[2]
	# passing on the left at the bottom 
	if y_b_r > 0.99 and x_b_r < 0.5:
		return "lb"
	# passing on the right at the bottom
	if y_b_r > 0.99 and x_t_l >= 0.5:
		return "rb"
	# passing on the left on the side
	if x_t_l < 0.01 and x_b_r < 0.5:
		return "ll"
	# passing on the right on the side
	if x_b_r > 0.99 and x_t_l >= 0.5:
		return "ll"
	# passing on the left at the top
	if y_t_l < 0.01 and x_b_r < 0.5:
		return "lu"
	# passing on the right at the top
	if y_t_l < 0.01 and x_t_l >= 0.5:
		return "ru"
	#otherwise
	return "unknown"

"""
path1: path to first image frame
path2: path to second image frame
returns: optical flow matrix
"""
def get_optical_flow(path1, path2):
	frame1 = cv2.imread(path1)
	frame2 = cv2.imread(path2)

	hsv = np.zeros_like(frame1)
	hsv[...,1] = 255
	prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	#flow = cv2.calcOpticalFlowFarneback(prvs,nxt,None, 0.5,3.0,15.0,3.0,5.0,1.2,0.0)
	flow = cv2.calcOpticalFlowFarneback(prvs,nxt,0.5,1,3,15,3,5,1)

	#mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	#hsv[...,0] = ang*180/np.pi/2
	#hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	#bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	#cv2.imshow('frame2',bgr)
	#k = cv2.waitKey(30) & 0xff

	#cv2.imwrite('opticalfb.png',frame2)
	#cv2.imwrite('opticalhsv.png',bgr)
#prvs = next

	#print np.shape(flow)
#print("in polar")
#print mag,ang
#cap.release()
#np.savetxt("foo.csv", ang , delimiter=",")
	#cv2.destroyAllWindows()
	return flow

"""
flow: optical flow matrix:
box: coordinates of box around person, has coordinates of top left point and bottom right point
-> box = [y coord (top left), x coord (top left), y coord (bottom right), x coord (bottom right)]
returns: optical flow vector corresponding to that person (2d array or 3d array)
"""
def get_optical_flow_vector(flow, box):
	if flow != []:
		box = [int(box[0]*Y_SIZE),int(box[1]*X_SIZE), int(box[2]*Y_SIZE), int(box[3]*X_SIZE)]
		print(box)
		xflow = []
		yflow = []
		for i in range(box[1],box[3]):
			for j in range(box[0],box[2]):
				xflow.append(flow[i][j][0])
				yflow.append(flow[i][j][1])
		boxflow = [xflow,yflow]
		#print boxflow
		x = np.mean(boxflow[0])
		y = np.mean(boxflow[1])
		return [x,y]
	else:
		return None






