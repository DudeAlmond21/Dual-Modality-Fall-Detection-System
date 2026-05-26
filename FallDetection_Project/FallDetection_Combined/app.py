import os
import sys
import cv2
import time
import threading
import csv
import requests
from datetime import datetime
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit

# ── ADD TRACK PATHS ───────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Human-Falling-Detect-Tracks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FallDetection_Track2'))

# ── TELEGRAM CONFIG ───────────────────────────────
BOT_TOKEN = "7661152384:AAHh-6mQQQq944M7E1hV95wavqnqB8ESKkg"
CHAT_ID   = "8738893566"
# ─────────────────────────────────────────────────

# ── FLASK APP ─────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'falldetection2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── PATHS (FIX 1 — absolute base dir) ────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_FILE  = os.path.join(BASE_DIR, 'fall_log.csv')
FALLS_DIR = os.path.join(BASE_DIR, 'falls')

# ── FUSION STATE ──────────────────────────────────
fusion_lock = threading.Lock()
fusion_state = {
    'result':     'NORMAL',
    'confidence': 'NONE',
    'color':      'green',
    'last_alert': 0
}

ALERT_COOLDOWN = 10

# ─────────────────────────────────────────────────
# TELEGRAM FUNCTIONS
# ─────────────────────────────────────────────────

def send_telegram_text(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=5)
        print(f"Telegram text sent: {message[:50]}...")
    except Exception as e:
        print(f"Telegram text error: {e}")


def send_telegram_photo(photo_path, caption):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        with open(photo_path, 'rb') as photo:
            requests.post(url,
                data={"chat_id": CHAT_ID, "caption": caption},
                files={"photo": photo},
                timeout=10)
        print(f"Telegram photo sent!")
    except Exception as e:
        print(f"Telegram photo error: {e}")


def get_latest_screenshot():
    try:
        if not os.path.exists(FALLS_DIR):
            return None
        files = [os.path.join(FALLS_DIR, f) for f in os.listdir(FALLS_DIR)
                 if f.endswith('.jpg')]
        if not files:
            return None
        return max(files, key=os.path.getctime)
    except:
        return None


def grab_live_screenshot():
    """Capture current live frame at alert time — person already on floor."""
    try:
        import main as track1
        with track1.frame_lock:
            frame = track1.latest_frame
        if frame is None:
            return get_latest_screenshot()
        os.makedirs(FALLS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(FALLS_DIR, f'fall_alert_{timestamp}.jpg')
        cv2.imwrite(path, frame)
        return path
    except Exception as e:
        print(f"Live screenshot error: {e}")
        return get_latest_screenshot()


def trigger_alert(result, camera_conf, imu_conf, confidence_label, use_screenshot=True):
    now = time.time()

    if result == 'POSSIBLE FALL':
        with fusion_lock:
            if now - fusion_state['last_alert'] < ALERT_COOLDOWN:
                return
            fusion_state['last_alert'] = now
    else:
        with fusion_lock:
            fusion_state['last_alert'] = now

    timestamp = datetime.now().strftime('%H:%M:%S')

    if result == 'CONFIRMED FALL':
        message = (f"🚨 CONFIRMED FALL DETECTED!\n"
                   f"Time: {timestamp}\n"
                   f"Confidence: {confidence_label}\n"
                   f"Camera: {camera_conf:.1f}%\n"
                   f"Wearable: {imu_conf:.1f}%")

        screenshot = grab_live_screenshot() if use_screenshot else None

        if screenshot:
            threading.Thread(
                target=send_telegram_photo,
                args=(screenshot, message),
                daemon=True
            ).start()
        else:
            threading.Thread(
                target=send_telegram_text,
                args=(message,),
                daemon=True
            ).start()

    elif result == 'POSSIBLE FALL':
        source = []
        if camera_conf > 0: source.append('CAMERA')
        if imu_conf > 0:    source.append('WEARABLE')
        message = (f"⚠️ Possible Fall Detected\n"
                   f"Time: {timestamp}\n"
                   f"Confidence: {confidence_label}\n"
                   f"Source: {' + '.join(source) if source else 'UNKNOWN'}")
        threading.Thread(
            target=send_telegram_text,
            args=(message,),
            daemon=True
        ).start()

# ─────────────────────────────────────────────────
# FUSION LOGIC — L1 + L2 + L3
# ─────────────────────────────────────────────────

def get_adaptive_weights(brightness, person_detected, imu_connected):
    if not imu_connected:
        return 1.0, 0.0
    if not person_detected:
        return 0.0, 1.0
    if brightness < 50:
        return 0.2, 0.8
    elif brightness < 100:
        return 0.4, 0.6
    else:
        return 0.6, 0.4


def run_fusion():
    try:
        from main import shared_state, state_lock
        from fall_detection_wrist import imu_state, imu_lock
        print("✅ Fusion connected to both tracks!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return

    camera_fall_time    = None
    imu_fall_time       = None
    cam_fall_start      = None
    possible_fall_start = None
    possible_alerted    = False
    confirmed_alerted   = False   # ← NEW

    CAM_SUSTAIN_SECS  = 2.0
    POSSIBLE_TIMEOUT  = 8.0

    while True:
        try:
            with state_lock:
                cam_action      = shared_state['action']
                cam_conf        = shared_state['confidence']
                cam_fall        = shared_state['fall_detected']
                cam_fall_time   = shared_state['fall_time']
                brightness      = shared_state['frame_brightness']
                person_detected = shared_state['person_detected']
                fps             = shared_state['fps']

            with imu_lock:
                imu_fall         = imu_state['fall_detected']
                imu_fall_time_v  = imu_state['fall_time']
                imu_connected    = imu_state['connected']
                imu_conf_raw     = imu_state['imu_confidence']
                svm              = imu_state['svm']
                jerk             = imu_state['jerk']
                imu_status       = imu_state['status']

            imu_conf_pct = imu_conf_raw * 100

            if cam_fall and cam_fall_time:
                camera_fall_time = cam_fall_time
            if imu_fall and imu_fall_time_v:
                imu_fall_time = imu_fall_time_v

            if cam_fall:
                if cam_fall_start is None:
                    cam_fall_start = time.time()
            else:
                cam_fall_start = None

            cam_sustained_secs = (time.time() - cam_fall_start) if cam_fall_start else 0.0

            cam_w, imu_w = get_adaptive_weights(
                brightness, person_detected, imu_connected)

            camera_score = (cam_conf / 100.0) if cam_fall else 0.0
            imu_score    = imu_conf_raw        if imu_fall else 0.0
            combined     = (camera_score * cam_w) + (imu_score * imu_w)

            time_agreed = False
            if camera_fall_time and imu_fall_time:
                time_agreed = abs(camera_fall_time - imu_fall_time) < 4.0

            camera_only_mode = not imu_connected
            imu_only_mode    = not person_detected and imu_connected

            # ── FINAL DECISION ────────────────────────────────

            if combined > 0.75 and time_agreed:
                result     = 'CONFIRMED FALL'
                confidence = 'HIGH'
                color      = 'red'
                if not confirmed_alerted:
                    trigger_alert(result, cam_conf, imu_conf_pct, confidence)
                    confirmed_alerted = True
                possible_fall_start = None
                possible_alerted    = False

            elif camera_only_mode and cam_fall and cam_conf > 75:
                result     = 'CONFIRMED FALL'
                confidence = 'HIGH (Camera Only)'
                color      = 'red'
                if not confirmed_alerted:
                    trigger_alert(result, cam_conf, 0, confidence)
                    confirmed_alerted = True
                possible_fall_start = None
                possible_alerted    = False

            elif imu_only_mode and imu_fall and imu_conf_raw > 0.9:
                result     = 'CONFIRMED FALL'
                confidence = 'HIGH (Wearable Only)'
                color      = 'red'
                if not confirmed_alerted:
                    trigger_alert(result, 0, imu_conf_pct, confidence, use_screenshot=False)
                    confirmed_alerted = True
                possible_fall_start = None
                possible_alerted    = False

            elif cam_fall and cam_conf > 75 and cam_sustained_secs >= CAM_SUSTAIN_SECS:
                result     = 'CONFIRMED FALL'
                confidence = 'HIGH (Camera Sustained)'
                color      = 'red'
                if not confirmed_alerted:
                    trigger_alert(result, cam_conf, 0, confidence)
                    confirmed_alerted = True
                possible_fall_start = None
                possible_alerted    = False

            elif camera_only_mode and cam_fall and cam_sustained_secs >= 4.0:
                result     = 'CONFIRMED FALL'
                confidence = 'HIGH (Camera Only - Lying)'
                color      = 'red'
                if not confirmed_alerted:
                    trigger_alert(result, cam_conf, 0, confidence)
                    confirmed_alerted = True
                possible_fall_start = None
                possible_alerted    = False

            elif combined > 0.75:
                result     = 'POSSIBLE FALL'
                confidence = 'MEDIUM'
                color      = 'orange'
                if not possible_alerted:
                    trigger_alert(result, cam_conf, imu_conf_pct, confidence)
                    possible_alerted = True
                if possible_fall_start is None:
                    possible_fall_start = time.time()

            elif cam_fall or imu_fall:
                result     = 'POSSIBLE FALL'
                confidence = 'LOW'
                color      = 'orange'
                if not possible_alerted:
                    trigger_alert(result, cam_conf, imu_conf_pct, confidence)
                    possible_alerted = True
                if possible_fall_start is None:
                    possible_fall_start = time.time()

            else:
                result              = 'NORMAL'
                confidence          = 'NONE'
                color               = 'green'
                possible_fall_start = None
                possible_alerted    = False
                confirmed_alerted   = False   # ← reset only on genuine NORMAL
                if not cam_fall:
                    camera_fall_time = None
                if not imu_fall:
                    imu_fall_time = None

            # Force NORMAL after timeout
            if result == 'POSSIBLE FALL' and possible_fall_start:
                if time.time() - possible_fall_start > POSSIBLE_TIMEOUT:
                    result              = 'NORMAL'
                    confidence          = 'NONE'
                    color               = 'green'
                    possible_fall_start = None
                    # neither flag reset — no more alerts until genuine NORMAL
                    camera_fall_time    = None
                    imu_fall_time       = None

            with fusion_lock:
                fusion_state['result']     = result
                fusion_state['confidence'] = confidence
                fusion_state['color']      = color

            socketio.emit('update', {
                'camera_action':     cam_action,
                'camera_conf':       round(cam_conf, 1),
                'camera_fall':       cam_fall,
                'fps':               round(fps, 1),
                'person_detected':   person_detected,
                'brightness':        round(brightness, 1),
                'imu_svm':           round(svm, 2),
                'imu_jerk':          round(jerk, 2),
                'imu_fall':          imu_fall,
                'imu_connected':     imu_connected,
                'imu_status':        imu_status,
                'imu_conf':          round(imu_conf_pct, 1),
                'cam_weight':        round(cam_w, 2),
                'imu_weight':        round(imu_w, 2),
                'fusion_result':     result,
                'fusion_confidence': confidence,
                'fusion_color':      color,
                'timestamp':         datetime.now().strftime('%H:%M:%S')
            })

        except Exception as e:
            print(f"Fusion error: {e}")

        time.sleep(0.1)


# ─────────────────────────────────────────────────
# VIDEO STREAM
# ─────────────────────────────────────────────────

def generate_frames():
    try:
        import main as track1
    except ImportError:
        return

    while True:
        with track1.frame_lock:
            frame = track1.latest_frame

        if frame is not None:
            try:
                ret, buffer = cv2.imencode('.jpg', frame,
                              [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n'
                           + buffer.tobytes()
                           + b'\r\n')
            except:
                pass

        time.sleep(0.033)


# ─────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fall_log')
def fall_log():
    events = []
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                reader = csv.DictReader(f)
                events = list(reader)[-20:]
                events.reverse()
    except Exception as e:
        print(f"Log read error: {e}")
    return jsonify(events)


@socketio.on('connect')
def on_connect():
    print("Web client connected!")
    emit('status', {'message': 'Connected to Fall Detection System'})


# ─────────────────────────────────────────────────
# START
# ─────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(FALLS_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'source',
                             'confidence', 'duration', 'type'])

    # Start fusion engine
    fusion_thread = threading.Thread(target=run_fusion, daemon=True)
    fusion_thread.start()
    print("✅ Fusion engine started!")

    # Start Track 1 — Vision detection
    import main as track1
    track1.LOG_FILE  = LOG_FILE    # FIX 1 — unify CSV path
    track1.FALLS_DIR = FALLS_DIR
    detection_thread = threading.Thread(
        target=track1.run_detection,
        args=('0',),
        daemon=True
    )
    detection_thread.start()
    print("✅ Track 1 detection started!")

    # Start Track 2 — IMU socket + watchdog (FIX 2)
    import fall_detection_wrist as track2
    track2.LOG_FILE = LOG_FILE     # FIX 1 — unify CSV path
    track2.start_imu_thread()      # FIX 2 — starts BOTH watchdog + socket
    print("✅ Track 2 IMU started!")

    print("🌐 Open browser: http://localhost:5000")
    print("📱 Open on phone: http://YOUR_LAPTOP_IP:5000")

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)