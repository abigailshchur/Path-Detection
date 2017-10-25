#!/bin/bash  
echo "Loading model..."
cd object_detection
python load_model.py
mkdir video_frames
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo "Lets goooo"