import numpy as np

class TemporalInterpolation:
    def __init__(self, max_gap=20):
        self.max_gap = max_gap

    def apply(self, history):
        """
        history: {track_id: [{'frame': f, 'bbox': [x1,y1,x2,y2]}, ...]}
        """
        refined_history = {}
        
        for tid, data in history.items():
            if len(data) < 2:
                refined_history[tid] = data
                continue
                
            sorted_data = sorted(data, key=lambda x: x['frame'])
            new_data = []
            
            for i in range(len(sorted_data) - 1):
                curr_p = sorted_data[i]
                next_p = sorted_data[i+1]
                new_data.append(curr_p)
                
                gap = next_p['frame'] - curr_p['frame']
                
                # If gap is small, interpolate (Method 4)
                if 1 < gap <= self.max_gap:
                    for t in range(1, gap):
                        ratio = t / gap
                        # Linear math for each coordinate
                        interp_bbox = curr_p['bbox'] + (next_p['bbox'] - curr_p['bbox']) * ratio
                        new_data.append({
                            'frame': curr_p['frame'] + t,
                            'bbox': interp_bbox,
                            'class': curr_p['class']
                        })
            
            new_data.append(sorted_data[-1])
            refined_history[tid] = new_data
            
        return refined_history