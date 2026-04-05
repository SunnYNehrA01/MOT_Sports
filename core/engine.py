import cv2
import os
from tqdm import tqdm
from core.detector import SportsDetector
from core.reid import ReIDExtractor
from modules.tracker import ByteTracker
from modules.motion import CameraMotionCompensation
from modules.post_process import TemporalInterpolation
from utils.video_io import VideoHandler
from utils.visualization import Visualizer

class TrackingEngine:
    def __init__(self, args):
        self.args = args
        self.detector = SportsDetector(model_weights=args.get('model', 'yolo11n.pt'))
        self.reid = ReIDExtractor() if args.get('use_reid', True) else None
        self.tracker = ByteTracker(args)
        self.cmc = CameraMotionCompensation() if args.get('use_cmc', True) else None
        self.visualizer = Visualizer()
        
    def process_video(self, input_path, output_path, sport_type, progress_callback=None):
        video = VideoHandler(input_path, output_path)
        fps = video.fps
        total_frames = video.total_frames
        
        # To store data for Method 4 (Interpolation)
        all_tracks_history = {} 

        prev_frame = None
        
        for frame_idx in tqdm(range(total_frames), desc="Processing Video"):
            ret, frame = video.read_frame()
            if not ret:
                break
            
            # 1. Camera Motion Compensation (Method 1)
            warp_matrix = None
            if self.cmc and prev_frame is not None:
                warp_matrix = self.cmc.compute_motion(prev_frame, frame)
            
            # 2. Object Detection (Method 5)
            detections = self.detector.detect(
                frame, 
                sport_type=sport_type, 
                conf_threshold=self.args.get('conf', 0.25)
            )
            
            # 3. Appearance Feature Extraction (Method 3)
            embeddings = []
            if self.reid and len(detections) > 0:
                bboxes = [d[:4] for d in detections]
                embeddings = self.reid.extract(frame, bboxes)
            
            # 4. Multi-Object Tracking (Method 2)
            # Updates tracks and handles ByteTrack logic
            online_targets = self.tracker.update(frame, detections, embeddings, warp_matrix)
            
            # 5. Store for Post-Processing (Method 4)
            for t in online_targets:
                if t.track_id not in all_tracks_history:
                    all_tracks_history[t.track_id] = []
                all_tracks_history[t.track_id].append({
                    'frame': frame_idx,
                    'bbox': t.tlbr, # Top-left, bottom-right
                    'class': t.class_id
                })
            
            # 6. Immediate Visualization (for the progress preview)
            annotated_frame = self.visualizer.draw_tracks(frame, online_targets)
            video.write_frame(annotated_frame)
            
            prev_frame = frame.copy()
            
            if progress_callback:
                progress_callback(frame_idx / total_frames)

        # 7. Post-Processing: Temporal Interpolation (Method 4)
        print("Refining tracks with Temporal Interpolation...")
        interpolator = TemporalInterpolation(max_gap=self.args.get('max_gap', 20))
        refined_history = interpolator.apply(all_tracks_history)
        
        # Final cleanup and closing files
        video.release()
        
        # Note: In a production app, we would re-render the video 
        # using 'refined_history' for maximum smoothness.
        
        return output_path