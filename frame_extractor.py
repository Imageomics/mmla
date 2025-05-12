import os
import cv2
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
import numpy as np
import glob
import datetime
import re

def parse_annotation_file(annotation_path):
    """Parse CVAT XML annotation file and extract bounding box information."""
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Get video source and original dimensions
        # First check for the structure from the sample you provided
        source_name = None
        width = None
        height = None
        
        # Try to find meta information in different possible structures
        meta = root.find('meta')
        if meta is not None:
            task = meta.find('task')
            if task is not None:
                # Try to get source
                source_element = task.find('source')
                if source_element is not None and source_element.text:
                    source_name = source_element.text
                
                # Try to get original size
                original_size = task.find('original_size')
                if original_size is not None:
                    width_element = original_size.find('width')
                    height_element = original_size.find('height')
                    if width_element is not None and height_element is not None:
                        width = int(width_element.text)
                        height = int(height_element.text)
        
        # If we couldn't find the source or dimensions in the expected structure,
        # try to extract from the annotation filename
        if source_name is None:
            # Try to extract from filename
            base_name = os.path.basename(annotation_path)
            source_name = base_name.replace('_tracks.xml', '').replace('.xml', '')
            
        # If dimensions are not found, use default values
        if width is None or height is None:
            print(f"Warning: Could not find dimensions in {annotation_path}, using defaults (3840x2160)")
            width = 3840  # Default width
            height = 2160  # Default height
        
        print(f"Annotation source: {source_name}")
        print(f"Dimensions: {width}x{height}")
        
        # Extract all tracks (objects)
        tracks = {}
        for track in root.findall('.//track'):
            track_id = track.get('id')
            label = track.get('label')
            
            if track_id not in tracks:
                tracks[track_id] = {
                    'label': label,
                    'boxes': []
                }
            
            # Extract all bounding boxes for this track
            for box in track.findall('box'):
                frame_id = int(box.get('frame'))
                outside = int(box.get('outside'))
                
                # Only process boxes that are not outside the frame
                if outside == 0:
                    xtl = float(box.get('xtl'))
                    ytl = float(box.get('ytl'))
                    xbr = float(box.get('xbr'))
                    ybr = float(box.get('ybr'))
                    
                    tracks[track_id]['boxes'].append({
                        'frame': frame_id,
                        'xtl': xtl,
                        'ytl': ytl,
                        'xbr': xbr,
                        'ybr': ybr
                    })
        
        # If no tracks found, check if we need to look in a different structure
        if not tracks:
            print("No tracks found in standard format, checking for alternative XML structure...")
            # Try alternative structure that might exist
            for annotation in root.findall('.//annotation'):
                obj_elements = annotation.findall('.//object')
                if obj_elements:
                    frame_id_element = annotation.find('filename')
                    if frame_id_element is not None and frame_id_element.text:
                        # Try to extract frame number from filename (assuming format like frame_000123.jpg)
                        frame_id_str = frame_id_element.text
                        frame_id_match = re.search(r'(\d+)', frame_id_str)
                        if frame_id_match:
                            frame_id = int(frame_id_match.group(1))
                            
                            for i, obj in enumerate(obj_elements):
                                label_element = obj.find('name')
                                if label_element is None or not label_element.text:
                                    continue
                                    
                                label = label_element.text
                                bbox = obj.find('bndbox')
                                if bbox is None:
                                    continue
                                    
                                xmin = float(bbox.find('xmin').text if bbox.find('xmin') is not None else 0)
                                ymin = float(bbox.find('ymin').text if bbox.find('ymin') is not None else 0)
                                xmax = float(bbox.find('xmax').text if bbox.find('xmax') is not None else 0)
                                ymax = float(bbox.find('ymax').text if bbox.find('ymax') is not None else 0)
                                
                                track_id = f"alt_{i}"
                                if track_id not in tracks:
                                    tracks[track_id] = {
                                        'label': label,
                                        'boxes': []
                                    }
                                
                                tracks[track_id]['boxes'].append({
                                    'frame': frame_id,
                                    'xtl': xmin,
                                    'ytl': ymin,
                                    'xbr': xmax,
                                    'ybr': ymax
                                })
        
        # Print statistics
        total_boxes = sum(len(track['boxes']) for track in tracks.values())
        print(f"Parsed {len(tracks)} tracks with {total_boxes} boxes")
        
        return {
            'source': source_name,
            'width': width,
            'height': height,
            'tracks': tracks
        }
    except Exception as e:
        print(f"Error parsing annotation file: {e}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()
        return {
            'source': os.path.basename(annotation_path).replace('_tracks.xml', '').replace('.xml', ''),
            'width': 3840,  # Default width
            'height': 2160,  # Default height
            'tracks': {}
        }

def organize_by_frame(annotation_data):
    """Reorganize annotation data by frame for easier processing."""
    frames = {}
    
    for track_id, track in annotation_data['tracks'].items():
        label = track['label']
        
        for box in track['boxes']:
            frame_id = box['frame']
            
            if frame_id not in frames:
                frames[frame_id] = []
            
            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = (box['xtl'] + box['xbr']) / 2 / annotation_data['width']
            y_center = (box['ytl'] + box['ybr']) / 2 / annotation_data['height']
            width = (box['xbr'] - box['xtl']) / annotation_data['width']
            height = (box['ybr'] - box['ytl']) / annotation_data['height']
            
            frames[frame_id].append({
                'label': label,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'original': {
                    'xtl': box['xtl'],
                    'ytl': box['ytl'],
                    'xbr': box['xbr'],
                    'ybr': box['ybr']
                }
            })
    
    return frames

def extract_video_date(video_path):
    """
    Attempt to extract date from video filename.
    Returns a formatted date string or None if no date found.
    """
    # Pattern for common date formats in filenames (YYYYMMDD or YYYY-MM-DD)
    patterns = [
        r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})',  # YYYY-MM-DD, YYYY_MM_DD, YYYYMMDD
        r'(\d{2})[-_]?(\d{2})[-_]?(\d{4})',  # DD-MM-YYYY, DD_MM_YYYY, DDMMYYYY
        r'DJI_(\d{4})',                       # DJI_XXXX format where XXXX is sequence
    ]
    
    filename = os.path.basename(video_path)
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            
            # If it's the DJI_XXXX format, just use the sequence number
            if len(groups) == 1 and 'DJI_' in filename:
                return f"DJI{groups[0]}"
                
            # If it's a YYYY-MM-DD format
            elif len(groups) == 3 and len(groups[0]) == 4:
                year, month, day = groups
                return f"{year}{month}{day}"
                
            # If it's a DD-MM-YYYY format
            elif len(groups) == 3 and len(groups[2]) == 4:
                day, month, year = groups
                return f"{year}{month}{day}"
    
    # If no date pattern found, get video file modification date
    try:
        mtime = os.path.getmtime(video_path)
        date_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y%m%d')
        return date_str
    except:
        # If all else fails, use current date
        return datetime.datetime.now().strftime('%Y%m%d')

def extract_frames(video_path, annotation_data, output_dir, labels_dir=None, visualize=False, 
                  extract_margin=0.0, class_mapping=None, include_metadata=True, 
                  include_background=False, background_interval=30, background_max=100):
    """
    Extract frames from video based on annotation data and save YOLO format labels.
    
    Args:
        video_path: Path to the video file
        annotation_data: Parsed annotation data
        output_dir: Directory to save extracted frames
        labels_dir: Directory to save YOLO label files (defaults to output_dir if None)
        visualize: Whether to draw bounding boxes on extracted frames
        extract_margin: Margin to add around objects (percentage of image dimensions)
        class_mapping: Dictionary mapping class names to YOLO class indices
        include_metadata: Include video name and date in output filenames
        include_background: Whether to include frames without annotations as background
        background_interval: Interval for sampling background frames (every Nth frame)
        background_max: Maximum number of background frames to extract per video
    """
    # Default class mapping if none provided
    if class_mapping is None:
        # Create class mapping from unique labels in annotation
        unique_labels = set()
        for track_id, track in annotation_data['tracks'].items():
            unique_labels.add(track['label'])
        
        class_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print(f"Auto-generated class mapping: {class_mapping}")
    
    # Set labels directory if not specified
    if labels_dir is None:
        labels_dir = output_dir
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Save class mapping to file if it doesn't exist
    class_file_path = os.path.join(os.path.dirname(labels_dir), "classes.txt")
    if not os.path.exists(class_file_path):
        with open(class_file_path, 'w') as f:
            for label, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
                f.write(f"{label}\n")
    
    # Get video basename and extract date
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    date_str = extract_video_date(video_path)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get frame data organized by frame
    frames_data = organize_by_frame(annotation_data)
    max_frame_id = max(frames_data.keys()) if frames_data else 0
    
    print(f"Video: {video_path}")
    print(f"Video basename: {video_basename}")
    print(f"Date string: {date_str}")
    print(f"Total frames in video: {total_frames}")
    print(f"Max frame ID in annotations: {max_frame_id}")
    print(f"FPS: {fps}")
    print(f"Number of annotated frames: {len(frames_data)}")
    
    # Process each frame
    frame_count = 0
    processed_count = 0
    background_count = 0
    
    # Set for quick lookup of frames with annotations
    annotated_frames = set(frames_data.keys())
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create base filename with metadata if requested
        if include_metadata:
            base_filename = f"{video_basename}_{date_str}_{frame_count:06d}"
        else:
            base_filename = f"{video_basename}_{frame_count:06d}"
        
        # Check if current frame has annotations
        if frame_count in annotated_frames:
            objects = frames_data[frame_count]
            
            # Create YOLO format label file
            label_filename = f"{base_filename}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for obj in objects:
                    class_id = class_mapping[obj['label']]
                    x_center = obj['x_center']
                    y_center = obj['y_center']
                    width = obj['width']
                    height = obj['height']
                    
                    # Write YOLO format: class_id center_x center_y width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    # Draw bounding box on frame if visualization is enabled
                    if visualize:
                        x1, y1 = int(obj['original']['xtl']), int(obj['original']['ytl'])
                        x2, y2 = int(obj['original']['xbr']), int(obj['original']['ybr'])
                        label = obj['label']
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        cv2.putText(frame, f"{label} ({class_id})", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the frame image
            image_filename = f"{base_filename}.jpg"
            image_path = os.path.join(output_dir, image_filename)
            cv2.imwrite(image_path, frame)
            
            processed_count += 1
            
            # Print progress
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(frames_data)} annotated frames")
                
        # Handle background frames (frames without annotations)
        elif include_background and frame_count % background_interval == 0 and background_count < background_max:
            # Create empty label file (YOLO requires a label file for each image)
            label_filename = f"{base_filename}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            # Create an empty file
            with open(label_path, 'w') as f:
                pass  # Empty file indicates no objects
            
            # Save the frame image
            image_filename = f"{base_filename}.jpg"
            image_path = os.path.join(output_dir, image_filename)
            cv2.imwrite(image_path, frame)
            
            background_count += 1
            
            if background_count % 10 == 0:
                print(f"Extracted {background_count}/{background_max} background frames")
        
        frame_count += 1
    
    cap.release()
    print(f"Extraction complete for {video_basename}:")
    print(f"- Processed {processed_count} frames with annotations")
    print(f"- Extracted {background_count} background frames")
    return processed_count, background_count

def process_videos_and_annotations(video_dir, annotation_dir, output_dir, labels_dir=None, 
                                 visualize=False, extract_margin=0.0, class_mapping=None,
                                 include_background=False, background_interval=30, background_max=100):
    """
    Process multiple videos and their corresponding annotation files.
    
    Args:
        video_dir: Directory containing video files
        annotation_dir: Directory containing annotation XML files
        output_dir: Directory to save extracted frames
        labels_dir: Directory to save YOLO label files
        visualize: Whether to draw bounding boxes on extracted frames
        extract_margin: Margin to add around objects
        class_mapping: Dictionary mapping class names to YOLO class indices
        include_background: Whether to include frames without annotations as background
        background_interval: Interval for sampling background frames (every Nth frame)
        background_max: Maximum number of background frames to extract per video
    """
    # Find all video files
    video_extensions = ['.MP4']
    #video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    video_files = []
    
    # Check if video_dir is actually a direct path to a video file
    if os.path.isfile(video_dir) and any(video_dir.lower().endswith(ext.lower()) for ext in video_extensions):
        video_files = [video_dir]
        print(f"Found single video file: {video_dir}")
    else:
        # Search for video files in the directory
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(video_dir, f'*{ext}')))
            # Also check case-insensitive extension
            if ext.lower() != ext:
                video_files.extend(glob.glob(os.path.join(video_dir, f'*{ext.lower()}')))
    
    # Find all annotation files
    annotation_files = []
    
    # Check if annotation_dir is actually a direct path to an XML file
    if os.path.isfile(annotation_dir) and annotation_dir.lower().endswith('.xml'):
        annotation_files = [annotation_dir]
        print(f"Found single annotation file: {annotation_dir}")
    else:
        # Search for annotation files in the directory
        annotation_files = glob.glob(os.path.join(annotation_dir, '*.xml'))

    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    if not annotation_files:
        print(f"No annotation files found in {annotation_dir}")
        return
    
    print(f"Found {len(video_files)} video files and {len(annotation_files)} annotation files")
    
    # Create a mapping from video source name to annotation file
    annotation_map = {}
    for ann_file in annotation_files:
        try:
            tree = ET.parse(ann_file)
            root = tree.getroot()
            meta = root.find('meta')
            task = meta.find('task')
            source = task.find('source').text
            
            # Extract the basename without extension
            source_base = os.path.splitext(os.path.basename(source))[0]
            annotation_map[source_base] = ann_file
        except Exception as e:
            print(f"Error parsing annotation file {ann_file}: {e}")
    
    # Process each video file
    total_processed = 0
    videos_processed = 0
    
    for video_file in video_files:
        video_base = os.path.splitext(os.path.basename(video_file))[0]
        
        # Find matching annotation file
        ann_file = None
        for source_name, file_path in annotation_map.items():
            # Check if the source name is in the video filename or vice versa
            if source_name in video_base or video_base in source_name:
                ann_file = file_path
                break
        
        if ann_file:
            print(f"\nProcessing video: {video_file}")
            print(f"Using annotation file: {ann_file}")
            
            try:
                # Parse annotation file
                annotation_data = parse_annotation_file(ann_file)
                
                # Extract frames
                frames_processed, backgrounds_processed = extract_frames(
                    video_file,
                    annotation_data,
                    output_dir,
                    labels_dir,
                    visualize,
                    extract_margin,
                    class_mapping,
                    include_metadata=True,
                    include_background=include_background,
                    background_interval=background_interval,
                    background_max=background_max
                )
                
                total_processed += frames_processed
                videos_processed += 1
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
        else:
            print(f"\nNo matching annotation file found for video: {video_file}")
    
    print(f"\nAll processing complete!")
    print(f"Processed {videos_processed} videos")
    print(f"Extracted {total_processed} frames with annotations")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos based on CVAT XML annotations')
    parser.add_argument('--video_dir', help='Directory containing video files')
    parser.add_argument('--video', help='Path to a specific video file')
    parser.add_argument('--annotation_dir', help='Directory containing annotation XML files')
    parser.add_argument('--annotations', help='Path to a specific CVAT XML annotation file')
    parser.add_argument('--output', required=True, help='Directory to save extracted frames')
    parser.add_argument('--labels', help='Directory to save YOLO label files (defaults to output dir)')
    parser.add_argument('--visualize', action='store_true', help='Draw bounding boxes on extracted frames')
    parser.add_argument('--margin', type=float, default=0.0, help='Margin to add around objects (percentage)')
    parser.add_argument('--classes', help='Path to class mapping file (label:id per line)')
    parser.add_argument('--include_background', action='store_true', help='Include frames without annotations as background')
    parser.add_argument('--background_interval', type=int, default=30, help='Interval for sampling background frames')
    parser.add_argument('--background_max', type=int, default=100, help='Maximum number of background frames per video')
    
    args = parser.parse_args()
    
    # Load class mapping if provided
    class_mapping = None
    if args.classes and os.path.exists(args.classes):
        class_mapping = {}
        with open(args.classes, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                label = line.strip()
                if label:
                    class_mapping[label] = idx
    
    # Check if processing multiple videos or a single video
    if args.video_dir and args.annotation_dir:
        process_videos_and_annotations(
            args.video_dir,
            args.annotation_dir,
            args.output,
            args.labels,
            args.visualize,
            args.margin,
            class_mapping,
            args.include_background,
            args.background_interval,
            args.background_max
        )
    elif args.video and args.annotations:
        # Parse annotation file
        annotation_data = parse_annotation_file(args.annotations)
        
        # Extract frames
        extract_frames(
            args.video,
            annotation_data,
            args.output,
            args.labels,
            args.visualize,
            args.margin,
            class_mapping,
            include_metadata=True,
            include_background=args.include_background,
            background_interval=args.background_interval,
            background_max=args.background_max
        )
    else:
        print("Error: You must provide either video_dir and annotation_dir OR video and annotations")

if __name__ == "__main__":
    main()