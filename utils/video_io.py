import cv2
import os

class VideoHandler:
    def __init__(self, input_path, output_path):
        self.cap = cv2.VideoCapture(input_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

    def read_frame(self):
        return self.cap.read()

    def write_frame(self, frame):
        self.writer.write(frame)

    def release(self):
        self.cap.release()
        self.writer.release()