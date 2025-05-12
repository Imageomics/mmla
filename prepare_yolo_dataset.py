import os
import random
from pathlib import Path
import shutil

#######################################################
# CONFIGURATION SECTION - MODIFY THESE VALUES
#######################################################

# Define source directories for each location
SOURCE_DIRS = {
    'location_1': 'mpala',  # REPLACE WITH YOUR ACTUAL PATH
    'location_2': 'opc',  # REPLACE WITH YOUR ACTUAL PATH
    'location_3': 'wilds'   # REPLACE WITH YOUR ACTUAL PATH
}

# Destination directory
DEST_DIR = "/data"  # REPLACE WITH YOUR ACTUAL PATH

# Define your class labels
CLASS_LABELS = {
    0: "Zebra",
    1: "Giraffe",
    2: "Onager", 
    3: "Dog",
}

# Sampling rate (adjust as needed - higher values mean fewer frames)
SAMPLING_RATE = 10

# Define the splits (train/test) for the 70/30 strategy
splits = {
    'train': {
        'location_3': {
            'session_1': ['DJI_0034', 'DJI_0035_part1'],  # African Painted Dog (70%)
            'session_2': ['P0140018'],  # Giraffe (70%)
            'session_3': ['P0100010', 'P0110011', 'P0080008', 'P0090009'],  # Persian Onanger (70%)
            
        },
        'location_1': {
            'session_1': ['DJI_0001', 'DJI_0002'],  # Giraffe
            'session_2': ['DJI_0005', 'DJI_0006'],  # Plains zebra
            'session_3': ['DJI_0068', 'DJI_0069'],  # Grevy's zebra
            'session_4': ['DJI_0142', 'DJI_0143', 'DJI_0144'],  # Grevy's zebra
            'session_5': ['DJI_0206', 'DJI_0208'],  # Mixed species
        },
        'location_2': {
            'session_1': ['P0800081', 'P0830086', 'P0840087', 'P0870091'],  # Plains zebra
            'session_2': ['P0910095'],  # Plains zebra
        }
    },
    'test': {
        'location_3': {
            'session_1': ['DJI_0035_part2'],  # African Painted Dog (30%)
            'session_3': ['P0070007', 'P0160016', 'P0120012'],  # Persian Onanger (30%)
            'session_2': ['P0150019'],  # Giraffe (30%)
            'session_4': ['P0070010'],  # Grevy's Zebra (100%)
        },
        'location_1': {
            'session_3': ['DJI_0070', 'DJI_0071'],  # Grevy's zebra
            'session_4': ['DJI_0145', 'DJI_0146', 'DJI_0147'],  # Grevy's zebra
            'session_5': ['DJI_0210', 'DJI_0211'],  # Mixed species
        },
        'location_2': {
            'session_1': ['P0860090'],  # Plains zebra
            'session_2': ['P0940098'],  # Plains zebra
        }
    }
}

#######################################################
# SCRIPT CODE - DO NOT MODIFY UNLESS NECESSARY
#######################################################

# Create destination directories
for split in ['train', 'test']:
    os.makedirs(f"{DEST_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{DEST_DIR}/labels/{split}", exist_ok=True)

def find_images_in_directory(dir_path):
    """Find all image files in a directory"""
    try:
        return [f for f in os.listdir(dir_path) 
                if f.endswith(('.jpg', '.png', '.jpeg')) and os.path.isfile(dir_path / f)]
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        print(f"Error accessing {dir_path}: {e}")
        return []

def find_partitions(session_path):
    """Find partition directories in a session"""
    try:
        return [d for d in os.listdir(session_path) 
                if os.path.isdir(session_path / d) and d.startswith('partition_')]
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        print(f"Error accessing {session_path}: {e}")
        return []

def find_video_images(session_path, video_name):
    """
    Find all images for a specific video in all partitions or video directory
    Returns a list of tuples: (image_path, image_name, partition_name)
    """
    all_images = []
    
    # First, check if the video is directly a directory
    video_path = session_path / video_name
    if os.path.isdir(video_path):
        # Check for partitions within video directory
        partitions = find_partitions(video_path)
        
        if partitions:
            # If partitions exist in video directory
            for partition in partitions:
                partition_path = video_path / partition
                images = find_images_in_directory(partition_path)
                all_images.extend([(partition_path, img, partition) for img in images])
        else:
            # Check for direct images in video directory (no partitions)
            images = find_images_in_directory(video_path)
            all_images.extend([(video_path, img, "") for img in images])
    
    # Also check for partitions directly in session directory
    partitions = find_partitions(session_path)
    for partition in partitions:
        partition_path = session_path / partition
        
        # Look for images matching this video name pattern
        for img in find_images_in_directory(partition_path):
            # Check if image filename contains this video name
            if video_name in img:
                all_images.append((partition_path, img, partition))
    
    return all_images

# Process each location and session
for split_name, locations in splits.items():
    for location_name, sessions in locations.items():
        # Get the source directory for this location
        if location_name not in SOURCE_DIRS:
            print(f"Warning: No source directory defined for {location_name}. Skipping.")
            continue
            
        location_source_dir = Path(SOURCE_DIRS[location_name])
        
        for session_name, video_info in sessions.items():
            session_path = location_source_dir / session_name
            
            if not os.path.exists(session_path):
                print(f"Warning: Session path {session_path} does not exist. Skipping.")
                continue
            
            # Get all videos in this session
            if isinstance(video_info, bool) and video_info:
                # Use all videos in the session - detect them from directories or video files
                try:
                    # First check for video directories
                    videos = [v for v in os.listdir(session_path) 
                             if os.path.isdir(session_path / v) and not v.startswith('partition_')]
                    
                    # If no video directories, try to infer from partition files
                    if not videos:
                        partitions = find_partitions(session_path)
                        if partitions:
                            # Get all images in first partition to extract video names
                            first_partition = session_path / partitions[0]
                            all_imgs = find_images_in_directory(first_partition)
                            # Extract potential video names from image filenames
                            videos = list(set([img.split('_')[0] for img in all_imgs if '_' in img]))
                            
                except (FileNotFoundError, NotADirectoryError) as e:
                    print(f"Warning: Could not list directory {session_path}: {e}")
                    continue
            else:
                # Use specific videos
                videos = video_info
            
            # Process each video
            for video in videos:
                print(f"Processing {location_name}/{session_name}/{video}...")
                
                # Find all images for this video (in all partitions)
                frame_info = find_video_images(session_path, video)
                
                if not frame_info:
                    print(f"Warning: No frames found for {video} in {session_name}")
                    continue
                
                # Sort frames by name to ensure temporal order
                frame_info.sort(key=lambda x: x[1])
                
                # Sample frames at regular intervals
                sampled_frame_info = frame_info[::SAMPLING_RATE]
                
                # Copy sampled frames and labels to destination
                for frame_dir, frame_name, partition in sampled_frame_info:
                    # Create a path component for the partition if it exists
                    partition_str = "" if partition == "" else f"_{partition}"
                    
                    # Copy image
                    src_img = frame_dir / frame_name
                    dest_img_name = f"{location_name}_{session_name}_{video}{partition_str}_{frame_name}"
                    dest_img = Path(DEST_DIR) / "images" / split_name / dest_img_name
                    
                    try:
                        shutil.copy(src_img, dest_img)
                    except (FileNotFoundError, IOError) as e:
                        print(f"Error copying image {src_img}: {e}")
                        continue
                    
                    # Handle different possible label locations
                    label_name = frame_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
                    
                    # Possible label locations (in order of priority)
                    possible_label_paths = [
                        # 1. Same directory as image
                        frame_dir / label_name,
                        
                        # 2. Labels subdirectory in partition
                        frame_dir / "labels" / label_name,
                        
                        # 3. Labels directory parallel to partition with same structure
                        session_path / "labels" / partition / label_name,
                        
                        # 4. Flat labels directory for session
                        session_path / "labels" / label_name,
                        
                        # 5. In video directory (if it exists)
                        session_path / video / "labels" / label_name,
                    ]
                    
                    src_label = None
                    for label_path in possible_label_paths:
                        if os.path.exists(label_path):
                            src_label = label_path
                            break
                    
                    if src_label:
                        dest_label_name = dest_img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
                        dest_label = Path(DEST_DIR) / "labels" / split_name / dest_label_name
                        try:
                            shutil.copy(src_label, dest_label)
                        except (FileNotFoundError, IOError) as e:
                            print(f"Error copying label {src_label}: {e}")
                    else:
                        print(f"Warning: No label found for {src_img}")

print("Dataset split completed successfully!")

# Create dataset.yaml file
def create_dataset_yaml():
    with open(f"{DEST_DIR}/dataset.yaml", "w") as f:
        f.write(f"# YOLOv11 dataset config\n")
        f.write(f"path: {os.path.abspath(DEST_DIR)}  # dataset root dir\n")
        f.write(f"train: images/train  # train images\n")
        f.write(f"val: images/train  # validation uses train images\n")
        f.write(f"test: images/test  # test images\n\n")
        
        f.write(f"# Classes\n")
        f.write(f"names:\n")
        for class_id, class_name in CLASS_LABELS.items():
            f.write(f"  {class_id}: {class_name}\n")

create_dataset_yaml()

# Analyze the distribution
stats = {"train": {}, "test": {}}

for split in ['train', 'test']:
    # Count images by location
    locations = {}
    species_count = {}
    
    # Get all images in this split
    img_dir = Path(DEST_DIR) / "images" / split
    if not os.path.exists(img_dir):
        print(f"Warning: Directory {img_dir} does not exist.")
        continue
        
    total_count = 0
    
    for img in os.listdir(img_dir):
        parts = img.split('_')
        if len(parts) < 2:
            continue
            
        location = parts[0]
        session = parts[1]
        
        # Count by location
        if location not in locations:
            locations[location] = 0
        locations[location] += 1
        
        # Extract species information if possible
        species_key = f"{location}_{session}"
        if species_key not in species_count:
            species_count[species_key] = 0
        species_count[species_key] += 1
        
        # Increment total
        total_count += 1
    
    stats[split]["total"] = total_count
    stats[split]["locations"] = locations
    stats[split]["species"] = species_count

# Print stats
for split, data in stats.items():
    print(f"\n{split.upper()} set:")
    print(f"Total images: {data['total']}")
    
    print("Distribution by location:")
    for loc, count in data["locations"].items():
        percentage = (count/data['total']*100) if data['total'] > 0 else 0
        print(f"  - {loc}: {count} ({percentage:.1f}%)")
    
    print("\nDistribution by location_session:")
    for species_key, count in data["species"].items():
        percentage = (count/data['total']*100) if data['total'] > 0 else 0
        print(f"  - {species_key}: {count} ({percentage:.1f}%)")

print("\nOverall train/test ratio:", 
      f"{stats['train']['total'] / (stats['train']['total'] + stats['test']['total']):.1%}",
      f"/ {stats['test']['total'] / (stats['train']['total'] + stats['test']['total']):.1%}")