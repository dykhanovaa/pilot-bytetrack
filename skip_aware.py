"""ByteTrack-SkipAware v1. Explicit adaptation of ultralytics 8.3.0.

An unobserved frame is NOT an observed frame with zero detections.
Unobserved: advance original-frame clock and Kalman predictions, retain
confirmation state; output causal KF forecasts for confirmed tracked objects. Observed: delegate to upstream update,
including its treatment of a legitimately empty Boxes input.
No interpolation, no ReID, no learned motion model.
"""
from types import SimpleNamespace
import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.trackers.basetrack import TrackState

PARAMS=dict(track_high_thresh=.5,track_low_thresh=.1,new_track_thresh=.6,
            track_buffer=30,match_thresh=.8,fuse_score=True)

def boxes(rows=(), shape=(540,960)):
    return Boxes(np.asarray(rows,dtype=np.float32).reshape(-1,6),orig_shape=shape)

def stock():
    return BYTETracker(SimpleNamespace(**PARAMS),frame_rate=25)

class SkipAware(BYTETracker):
    def __init__(self):
        super().__init__(SimpleNamespace(**PARAMS),frame_rate=25)

    def step(self, detections=None, *, observed):
        if observed:
            if detections is None:
                raise ValueError('Observed frame requires Boxes, possibly empty')
            return super().update(detections)
        if detections is not None:
            raise ValueError('Skipped frame has no detector result; use None')
        self.frame_id += 1
        # Same one-frame KF prediction used by upstream for confirmed and lost.
        pool=self.joint_stracks([t for t in self.tracked_stracks if t.is_activated],
                               self.lost_stracks)
        self.multi_predict(pool)
        # Unconfirmed tracks are held, without KF prediction, just as upstream
        # does not predict them in its observed-frame update.
        # Expiry remains in ORIGINAL frames, never in detector-call counts.
        for attr in ('tracked_stracks','lost_stracks'):
            kept=[]
            for t in getattr(self,attr):
                if self.frame_id-t.end_frame > self.max_time_lost:
                    t.mark_removed();self.removed_stracks.append(t)
                else: kept.append(t)
            setattr(self,attr,kept)
        self.removed_stracks=self.removed_stracks[-999:]
        return np.asarray([t.result for t in self.tracked_stracks if t.is_activated],
                          dtype=np.float32).reshape(-1,8)

def state(tracker):
    rows=[]
    names={TrackState.New:'new',TrackState.Tracked:'tracked',
           TrackState.Lost:'lost',TrackState.Removed:'removed'}
    for collection in ('tracked_stracks','lost_stracks','removed_stracks'):
        for t in getattr(tracker,collection):
            rows.append(dict(id=int(t.track_id),state=names[t.state],
                confirmed=bool(t.is_activated),collection=collection,
                start=int(t.start_frame),last_observed=int(t.end_frame),
                xyxy=np.asarray(t.xyxy).round(3).tolist()))
    return rows
