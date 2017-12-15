# Path Detection

* start.sh: loads model for object detection
* end.sh: removes large files

* object_detection/LSTM.ipynb :  All code relating to analysis and LSTM in general
* object_detection/Optical Flow.ipynb : Fixes optical flow vectors for some of our data
* object_detection/video_object_detection.ipynb : Collects frame by frame bounding boxes and optical flow
* object_detection/util/data_collection_util.py: Actual code to collect frame by frame data
* object_detection/util/path_detection_utils.py: util for data_collection_util.py
* object_detection/util/saved_frame_data/.. : all of our data is saved here
* object_detection/load_base_video.py: gets frame by frame rep of video
* object_detection/load_model.py: loads tensorflow model