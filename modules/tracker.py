import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# --- Helper: IoU Calculation ---
def iou_batch(bboxes1, bboxes2):
    if bboxes1.size == 0 or bboxes2.size == 0:
        return np.empty((len(bboxes1), len(bboxes2)))
    bboxes1 = np.expand_dims(bboxes1, 1)
    bboxes2 = np.expand_dims(bboxes2, 0)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    return wh / (area1 + area2 - wh + 1e-7)

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class STrack:
    def __init__(self, tlbr, score, class_id, embedding=None, color_hist=None):
        self._tlbr = np.asarray(tlbr)
        self.score = score
        self.class_id = class_id
        self.track_id = 0
        self.state = TrackState.New
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        
        # Kalman Filter
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.x[:4] = self.tlbr_to_xyah(self._tlbr).reshape(4, 1)
        self.kf.F = np.eye(8)
        for i in range(4): self.kf.F[i, i+4] = 1 
        self.kf.H = np.eye(4, 8) 
        
        # Appearance & Team Logic
        self.smooth_feat = embedding
        self.curr_feat = embedding
        self.color_hist = color_hist
        self.alpha = 0.9

    @property
    def tlbr(self):
        return self._tlbr

    @staticmethod
    def tlbr_to_xyah(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]; ret[:2] += ret[2:] / 2; ret[2] /= ret[3]
        return ret

    def predict(self, warp_matrix=None):
        if warp_matrix is not None:
            pos = self.kf.x[:2].reshape(1, 1, 2)
            self.kf.x[:2] = cv2.transform(pos, warp_matrix).reshape(2, 1)
        self.kf.predict()

    def activate(self, frame_id, track_id):
        self.track_id = track_id
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.state = TrackState.Tracked
        self.is_activated = True

    def update(self, new_track, frame_id, is_crowded=False):
        """
        Main update for visible tracks. 
        If is_crowded is true, we don't update the Re-ID features to avoid poisoning.
        """
        self.frame_id = frame_id
        self._tlbr = new_track.tlbr
        self.kf.update(self.tlbr_to_xyah(self._tlbr))
        self.state = TrackState.Tracked
        self.is_activated = True
        
        if not is_crowded: 
            self.update_features(new_track)

    def update_features(self, new_track):
        if new_track.curr_feat is not None:
            if self.smooth_feat is None: self.smooth_feat = new_track.curr_feat
            else: self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * new_track.curr_feat
            self.smooth_feat /= np.linalg.norm(self.smooth_feat)
        
        if new_track.color_hist is not None:
            if self.color_hist is None: self.color_hist = new_track.color_hist
            else: self.color_hist = 0.9 * self.color_hist + 0.1 * new_track.color_hist

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

class ByteTracker:
    def __init__(self, args):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.args = args
        self.id_count = 0
        self.max_time_lost = args.get('max_lost', 60)

    def get_color_hist(self, frame, tlbr):
        x1, y1, x2, y2 = map(int, tlbr)
        h_h = y2 - y1
        crop = frame[max(0, y1+int(h_h*0.2)):min(frame.shape[0], y1+int(h_h*0.7)), 
                     max(0, x1):min(frame.shape[1], x2)]
        if crop.size == 0: return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        return hist / (hist.sum() + 1e-7)

    def get_dists(self, tracks, detections, reid_weight=0.7):
        iou_d = 1.0 - iou_batch(np.array([t.tlbr for t in tracks]), np.array([d.tlbr for d in detections]))
        iou_d[iou_d > 0.9] = 1e5 # Spatial Gating
        if not tracks or not detections or tracks[0].smooth_feat is None: return iou_d
        
        t_feats = np.array([t.smooth_feat for t in tracks])
        d_feats = np.array([d.curr_feat for d in detections])
        cos_d = 1.0 - np.dot(t_feats, d_feats.T)

        # Team Color Logic
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                if t.color_hist is not None and d.color_hist is not None:
                    sim = cv2.compareHist(t.color_hist, d.color_hist, cv2.HISTCMP_CORREL)
                    if sim < 0.3: cos_d[i, j] += 5.0 # Logic: Reject if jerseys don't match
        
        return (1.0 - reid_weight) * iou_d + reid_weight * cos_d

    def update(self, frame, detections, embeddings, warp_matrix=None):
        self.frame_id += 1
        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # 1. Process detections
        high_dets, low_dets = [], []
        for i, det in enumerate(detections):
            emb = embeddings[i] if len(embeddings) > 0 else None
            hist = self.get_color_hist(frame, det[:4])
            track = STrack(det[:4], det[4], det[5], emb, hist)
            if det[4] >= self.args.get('high_thresh', 0.6): high_dets.append(track)
            else: low_dets.append(track)

        # 2. Kalman Prediction
        for t in self.tracked_stracks + self.lost_stracks:
            t.predict(warp_matrix)

        # 3. Crowd Detection
        det_boxes = np.array([d.tlbr for d in high_dets])
        crowded_idx = set()
        if det_boxes.size > 0:
            iou_self = iou_batch(det_boxes, det_boxes)
            crowded_idx = set(np.where((iou_self > 0.3).sum(axis=1) > 1)[0])

        # --- ASSOCIATION 1: Active Tracks + High Confidence Dets ---
        dists = self.get_dists(self.tracked_stracks, high_dets)
        matches, u_track, u_det = self.linear_assignment(dists, 0.7)

        for it, idet in matches:
            track = self.tracked_stracks[it]
            track.update(high_dets[idet], self.frame_id, idet in crowded_idx)
            activated_stracks.append(track)

        # --- ASSOCIATION 2: High Confidence Dets + Lost Tracks (Recovery) ---
        lost_candidates = [t for t in self.lost_stracks]
        remaining_high_dets = [high_dets[i] for i in u_det]
        dists_rec = self.get_dists(lost_candidates, remaining_high_dets, reid_weight=0.9)
        matches_rec, u_lost, u_det_rec = self.linear_assignment(dists_rec, 0.4)

        for it, idet in matches_rec:
            track = lost_candidates[it]
            track.update(remaining_high_dets[idet], self.frame_id)
            refind_stracks.append(track)

        # Update remaining unmatched high detections
        u_det = [u_det[i] for i in u_det_rec]

        # --- ASSOCIATION 3: Remaining Tracks + Low Confidence Dets ---
        remaining_tracks = [self.tracked_stracks[i] for i in u_track] + [t for t in self.lost_stracks if t not in refind_stracks]
        dists_low = 1.0 - iou_batch(np.array([t.tlbr for t in remaining_tracks]), np.array([d.tlbr for d in low_dets]))
        matches_low, u_remain, _ = self.linear_assignment(dists_low, 0.5)

        for it, idet in matches_low:
            track = remaining_tracks[it]
            track.update(low_dets[idet], self.frame_id, is_crowded=True)
            if track.state == TrackState.Tracked: activated_stracks.append(track)
            else: refind_stracks.append(track)

        # --- FINAL MANAGEMENT ---
        for track in remaining_tracks:
            if track not in activated_stracks and track not in refind_stracks:
                if track.state == TrackState.Tracked:
                    track.mark_lost()
                    lost_stracks.append(track)
                elif track.state == TrackState.Lost:
                    lost_stracks.append(track)

        # Cleanup expired lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Init New Tracks
        for i in u_det:
            track = high_dets[i]
            if track.score < self.args.get('high_thresh', 0.6) + 0.1: continue
            self.id_count += 1
            track.activate(self.frame_id, self.id_count)
            activated_stracks.append(track)

        # Sync tracked_stracks
        self.tracked_stracks = [t for t in activated_stracks + refind_stracks if t.state == TrackState.Tracked]
        self.lost_stracks = [t for t in lost_stracks if t not in removed_stracks and t not in refind_stracks]
        self.removed_stracks.extend(removed_stracks)
        
        return self.tracked_stracks

    def linear_assignment(self, dist_matrix, threshold):
        if dist_matrix.size == 0: return [], list(range(dist_matrix.shape[0])), list(range(dist_matrix.shape[1]))
        r, c = linear_sum_assignment(dist_matrix)
        matches = [(r_idx, c_idx) for r_idx, c_idx in zip(r, c) if dist_matrix[r_idx, c_idx] <= threshold]
        u_t = list(set(range(dist_matrix.shape[0])) - set([m[0] for m in matches]))
        u_d = list(set(range(dist_matrix.shape[1])) - set([m[1] for m in matches]))
        return matches, u_t, u_d