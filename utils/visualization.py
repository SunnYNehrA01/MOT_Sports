import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.colors = {} 
        self.history = {} 
        self.class_names = {0: "Player", 32: "Ball", 34: "Bat", 2: "Car"}

    def get_color(self, idx):
        if idx not in self.colors:
            np.random.seed(idx)
            # Brighter colors for better visibility
            self.colors[idx] = tuple(np.random.randint(100, 255, 3).tolist())
        return self.colors[idx]

    def draw_tracks(self, frame, tracks):
        for t in tracks:
            tid = t.track_id
            cls_id = getattr(t, 'class_id', 0)
            color = self.get_color(tid)
            x1, y1, x2, y2 = map(int, t.tlbr)
            
            # 1. Thicker Bounding Box (Increased to 3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # 2. Larger Label with "High-Contrast" Background
            label = f"{self.class_names.get(cls_id, 'Obj')} {tid}"
            font_scale = frame.shape[1] / 1500
            thickness = max(2, int(frame.shape[1] / 400))     # Increased thickness
            
            (t_w, t_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Draw label background slightly above the box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1) # Black shadow
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness) # White text
            
            # 3. Enhanced Trajectory (Thicker lines)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if tid not in self.history: self.history[tid] = []
            
            if len(self.history[tid]) > 0:
                dist = np.linalg.norm(np.array(center) - np.array(self.history[tid][-1]))
                if dist > 200: self.history[tid] = [] # Anti-teleport
            
            self.history[tid].append(center)
            if len(self.history[tid]) > 30: self.history[tid].pop(0)
                
            for i in range(1, len(self.history[tid])):
                # Draw a thicker "glowing" line
                cv2.line(frame, self.history[tid][i-1], self.history[tid][i], color, 4)
                
        return frame