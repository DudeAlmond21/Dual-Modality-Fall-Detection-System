import socket
import numpy as np
import time
import collections
import csv
import threading
import os
from datetime import datetime

# ── TUNING ────────────────────────────────────
JERK_THRESHOLD   = 16.0
SVM_STILL_MAX    = 13.5
STILL_WINDOW     = 8
CONFIRM_WINDOW   = 25
# ─────────────────────────────────────────────

# ── SHARED STATE (for web app) ────────────────
imu_lock = threading.Lock()
imu_state = {
    'svm': 0.0,
    'jerk': 0.0,
    'fall_detected': False,
    'fall_time': None,
    'connected': False,
    'status': 'Normal',
    'imu_confidence': 0.0,
    'last_seen': 0.0          # ← heartbeat timestamp
}
# ─────────────────────────────────────────────

# ── CSV LOGGING ───────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
           '..', 'FallDetection_Combined', 'fall_log.csv')

def log_fall(confidence, fall_type):
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'IMU',
                f'{confidence:.2f}',
                '0.0',
                fall_type
            ])
    except Exception as e:
        print(f"CSV log error: {e}")
# ─────────────────────────────────────────────


def imu_watchdog():
    """Marks IMU as disconnected if no data received for 3 seconds."""
    while True:
        time.sleep(1)
        with imu_lock:
            if imu_state['connected']:
                if time.time() - imu_state['last_seen'] > 3.0:
                    imu_state['connected'] = False
                    imu_state['status'] = 'Disconnected'
                    imu_state['svm'] = 0.0
                    imu_state['jerk'] = 0.0
                    imu_state['imu_confidence'] = 0.0
                    print("⚠️ IMU connection lost (watchdog)!")


def run_imu_socket():
    """Main IMU socket listener — runs in background thread."""
    global imu_state

    s = socket.socket()
    s.bind(('0.0.0.0', 80))
    s.listen(0)
    print("Waiting for ESP32 connection...")

    client, addr = s.accept()
    print(f"ESP32 connected from {addr}")

    with imu_lock:
        imu_state['connected'] = True
        imu_state['last_seen'] = time.time()   # ← init heartbeat on connect

    p = ""
    tp = []
    svm_buffer = collections.deque(maxlen=50)
    fall_candidate = False
    fall_candidate_timer = 0

    while True:
        try:
            content = client.recv(1).decode("utf-8")
        except:
            print("ESP32 disconnected!")
            with imu_lock:
                imu_state['connected'] = False
                imu_state['status'] = 'Disconnected'
            break

        if content == '!':
            tp = []
            p = ''

        elif content == ',':
            try:
                tp.append(float(p))
            except:
                pass
            p = ''

        elif content == '@':
            try:
                tp.append(float(p))
            except:
                pass
            p = ''

            if len(tp) == 6:
                ax, ay, az = tp[0], tp[1], tp[2]
                svm = np.sqrt(ax**2 + ay**2 + az**2)
                svm_buffer.append(svm)

                if len(svm_buffer) >= 2:
                    jerk = abs(svm_buffer[-1] - svm_buffer[-2])
                else:
                    jerk = 0

                # Update shared state + heartbeat
                with imu_lock:
                    imu_state['svm'] = round(svm, 3)
                    imu_state['jerk'] = round(jerk, 3)
                    imu_state['status'] = 'Normal'
                    imu_state['imu_confidence'] = 0.0
                    imu_state['last_seen'] = time.time()   # ← update heartbeat

                # ── FALL DETECTION ────────────────────────
                if jerk > JERK_THRESHOLD and not fall_candidate:
                    fall_candidate = True
                    fall_candidate_timer = 0
                    print(f"⚡ Impact detected! Jerk={jerk:.2f}")
                    with imu_lock:
                        imu_state['status'] = 'Impact Detected'
                        imu_state['imu_confidence'] = 0.5

                if fall_candidate:
                    fall_candidate_timer += 1

                    if fall_candidate_timer >= CONFIRM_WINDOW:
                        recent_svm = list(svm_buffer)[-STILL_WINDOW:]
                        svm_std = np.std(recent_svm)
                        svm_mean = np.mean(recent_svm)

                        if svm_std < 1.5 and svm_mean < SVM_STILL_MAX:
                            print("=" * 50)
                            print("🚨 FALL CONFIRMED by IMU!")
                            print(f"   SVM mean: {svm_mean:.2f} | std: {svm_std:.2f}")
                            print("=" * 50)

                            with imu_lock:
                                imu_state['fall_detected'] = True
                                imu_state['fall_time'] = time.time()
                                imu_state['status'] = 'FALL CONFIRMED'
                                imu_state['imu_confidence'] = 0.95

                            log_fall(95.0, 'CONFIRMED')
                            time.sleep(3)

                            with imu_lock:
                                imu_state['fall_detected'] = False

                        else:
                            print(f"✅ False alarm cleared")
                            with imu_lock:
                                imu_state['status'] = 'Normal'
                                imu_state['imu_confidence'] = 0.0

                        fall_candidate = False
                        fall_candidate_timer = 0

                if not fall_candidate and len(svm_buffer) % 10 == 0:
                    print(f"Status: {imu_state['status']} | "
                          f"SVM={svm:.2f} | Jerk={jerk:.2f}")

        else:
            p += content


def start_imu_thread():
    """Start IMU watchdog + socket listener as background threads."""
    watchdog_thread = threading.Thread(target=imu_watchdog, daemon=True)
    watchdog_thread.start()

    socket_thread = threading.Thread(target=run_imu_socket, daemon=True)
    socket_thread.start()

    return socket_thread


if __name__ == '__main__':
    run_imu_socket()