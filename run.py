import argparse
import sys
import os
from core.engine import TrackingEngine

def parse_args():
    parser = argparse.ArgumentParser(description="MOT Sports Tracker CLI")
    
    # Required
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--sport", type=str, default="general", choices=["football", "basketball", "cricket", "racing", "general"])
    
    # Model Settings
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    
    # Algorithm Toggles
    parser.add_argument("--no-reid", action="store_false", dest="use_reid", help="Disable Re-ID embedding matching")
    parser.add_argument("--no-cmc", action="store_false", dest="use_cmc", help="Disable Camera Motion Compensation")
    
    # Tracking Hyperparameters
    parser.add_argument("--high-thresh", type=float, default=0.6, help="ByteTrack high confidence threshold")
    parser.add_argument("--max-gap", type=int, default=20, help="Max frames for temporal interpolation")
    
    # Output
    parser.add_argument("--output", type=str, default=None, help="Output path (default: tracked_{input_name})")
    
    return parser.parse_args()

def main():
    args_namespace = parse_args()
    
    # Convert namespace to dict for the Engine
    args = vars(args_namespace)
    
    # Determine output path if not provided
    if args['output'] is None:
        filename = os.path.basename(args['video'])
        args['output'] = f"output/tracked_{filename}"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args['output']), exist_ok=True)

    print("\n" + "="*50)
    print(f"🚀 Starting MOT Sports Tracker CLI")
    print(f"📍 Input: {args['video']}")
    print(f"🏟️ Sport: {args['sport']}")
    print(f"🧠 Model: {args['model']}")
    print("="*50 + "\n")

    try:
        # Initialize the master pipeline
        engine = TrackingEngine(args)
        
        # Execute processing
        # Note: No progress callback provided here, uses tqdm inside engine.py
        result_path = engine.process_video(
            input_path=args['video'], 
            output_path=args['output'], 
            sport_type=args['sport']
        )
        
        print(f"\n✅ Processing complete!")
        print(f"📁 Saved to: {result_path}\n")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()