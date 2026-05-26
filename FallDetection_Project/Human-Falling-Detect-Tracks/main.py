import os
import cv2
import time
import torch
import argparse
import numpy as np
import csv
import threading
from datetime import datetime

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

source = '../Data/falldata/Home/Videos/video (1).avi'

# ── SHARED STATE (for web app) ─────────────────────
frame_lock = threading.Lock()
latest_frame = None

state_lock = threading.Lock()
shared_state = {
    'action': 'Unknown',
    'confidence': 0.0,
    'fps': 0.0,
    'fall_detected': False,
    'fall_time': None,
    'person_detected': False,
    'frame_brightness': 0.0,
    'track_count': 0
}

# ── FALL TRACKING ───────────────────────────────────
fall_start_time = None
fall_duration = 0.0
resize_fn = None  # initialized in run_detection()

# ── CSV LOGGING ─────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
           '..', 'FallDetection_Combined', 'fall_log.csv')

def init_csv():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'source',
                             'confidence', 'duration', 'type'])

def log_fall(source, confidence, duration, fall_type):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            source,
            f'{confidence:.2f}',
            f'{duration:.1f}',
            fall_type
        ])

# ── SCREENSHOT ──────────────────────────────────────
FALLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'FallDetection_Combined', 'falls')

def save_screenshot(frame):
    os.makedirs(FALLS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(FALLS_DIR, f'fall_{timestamp}.jpg')
    cv2.imwrite(path, frame[:, :, ::-1])
    return path
# ────────────────────────────────────────────────────


def preproc(image):
    """preprocess function for CameraLoader."""
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that holds all keypoints (x,y)"""
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


def run_detection(cam_source_override=None):
    """Main detection loop — can be called directly or as a thread."""
    global fall_start_time, fall_duration, latest_frame, resize_fn

    # Change to Track 1 directory so relative model paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384)
    par.add_argument('--pose_input_size', type=str, default='224x160')
    par.add_argument('--pose_backbone', type=str, default='resnet50')
    par.add_argument('--show_detected', default=False, action='store_true')
    par.add_argument('--show_skeleton', default=True, action='store_true')
    par.add_argument('--save_out', type=str, default='')
    par.add_argument('--device', type=str, default='cuda')
    args, _ = par.parse_known_args()

    # Override camera source if provided
    if cam_source_override is not None:
        args.camera = cam_source_override

    device = args.device

    # DETECTION MODEL
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)
    init_csv()

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0

    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        image = frame.copy()

        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        tracker.predict()
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []
        if detected is not None:
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        tracker.update(detections)

        current_action = 'Unknown'
        current_confidence = 0.0

        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)

            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action_conf = float(out[0].max() * 100)
                action = '{}: {:.2f}%'.format(action_name, action_conf)

                current_action = action_name
                current_confidence = action_conf

                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                    if fall_start_time is None:
                        fall_start_time = time.time()
                        threading.Thread(
                            target=save_screenshot,
                            args=(image.copy(),),
                            daemon=True
                        ).start()
                        log_fall('CAMERA', action_conf, 0.0, 'DETECTED')
                    fall_duration = time.time() - fall_start_time
                    with state_lock:
                        shared_state['fall_detected'] = True
                        shared_state['fall_time'] = fall_start_time

                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)
                else:
                    if fall_start_time is not None:
                        fall_start_time = None
                        fall_duration = 0.0
                    with state_lock:
                        shared_state['fall_detected'] = False
                        shared_state['fall_time'] = None

            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.4, clr, 1)

                if fall_start_time is not None and fall_duration > 0:
                    dur_text = f'Duration: {fall_duration:.1f}s'
                    frame = cv2.putText(frame, dur_text,
                                        (bbox[0] + 5, bbox[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0), 1)

        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        elapsed = time.time() - fps_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, fps),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()

        brightness = float(np.mean(image))
        with state_lock:
            shared_state['action'] = current_action
            shared_state['confidence'] = current_confidence
            shared_state['fps'] = fps
            shared_state['person_detected'] = len(tracker.tracks) > 0
            shared_state['frame_brightness'] = brightness
            shared_state['track_count'] = len(tracker.tracks)

        with frame_lock:
            latest_frame = frame.copy()

        if outvid:
            writer.write(frame)

        # Only show window if running standalone (not from app.py)
        if __name__ == '__main__':
            # Skip imshow when running as thread (web app shows feed)
            pass

    cam.stop()
    if outvid:
        writer.release()


if __name__ == '__main__':
    import sys
    cam_arg = '1'  # default Iriun camera
    for i, arg in enumerate(sys.argv):
        if arg == '-C' and i + 1 < len(sys.argv):
            cam_arg = sys.argv[i + 1]
    run_detection(cam_source_override=cam_arg)