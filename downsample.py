# sub-sample frames every X seconds

import os
from tqdm import tqdm

video_dirs = [] # update this list with the actual video directories

frame_dir = '' # update this with the actual frame directory

total_frames = 500 # update this with the actual total frames to sample per video

for video_dir in video_dirs:
    
    num_frames = len([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        
    # figure out how many frames to skip
    skip = num_frames // total_frames
    if skip == 0:
        skip = 1  # Ensure at least one frame is copied if there are fewer than total_frames
        
    print(f'Using a skip of {skip} for {video_dir} of {num_frames} frames')
    
    # iterate through all jpg and txt files in the directory
    #for file in os.listdir(video_dir):
    for file in tqdm(os.listdir(video_dir), desc=f"Processing {video_dir}", unit="file"):
        
        # iterate through all jpg and txt files in the directory
        
        if file.endswith('.jpg'):
            # get the frame number
            frame_num = int(file.split('.')[0].split('_')[-1])
            
            # if the frame number is divisible by skip, copy the file
            if frame_num % skip == 0:
                # copy the file to the new directory
                os.system(f'cp {video_dir}/{file} {frame_dir}')
                os.system(f"cp {video_dir}/{file.replace('.jpg', '.txt')} {frame_dir}")
                
    print(f'Finished down-sampling {video_dir}')