#!/bin/bash  
echo "Loading model..."
cd object_detection
python load_model.py
echo "Loading in video..."
mkdir video_frames
python load_base_video.py
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo "Lets goooo"