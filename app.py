import os,socket,select,collections,threading,time
from datetime import datetime
import numpy as np
import tensorflow as tf
import joblib
import pygame
from flask import Flask,jsonify,render_template

DIR=os.path.dirname(os.path.abspath(__file__))
MODEL_FILE=os.path.join(DIR,"model","vitalguard_cnn.keras")
SCALER_FILE=os.path.join(DIR,"model","sensor_scaler.pkl")
ALERT_SOUND=os.path.join(DIR,"static","music","alert.mp3")

UDP_IP="0.0.0.0"
PORT_MOTION=5000
PORT_PULSE=5001

SAMPLE_RATE=50
WINDOW_SIZE=100
STEP_SIZE=50
FEATURE_COLUMNS=["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]
LABELS={1:"Falling",2:"Sitting",3:"Sleeping",4:"Standing",5:"Still",6:"Walking"}

app=Flask(__name__)

pygame.mixer.init()
pygame.mixer.music.load(ALERT_SOUND)

state_lock=threading.Lock()
buffer=collections.deque(maxlen=WINDOW_SIZE)
history=collections.deque()

dashboard={
    "activity":"Still",
    "confidence":0.0,
    "duration":0,
    "heart_rate":None,
    "spo2":None,
    "alerts":[],
    "device_connected":False,
    "device_ip":None,
    "last_update":0,
    "last_motion":0,
    "last_pulse":0,
}

activity_started_at=time.time()
previous_state=None
total_samples=0
next_prediction=WINDOW_SIZE
last_alerts=[]

print("="*60)
print("VITALGUARD AI - FLASK DASHBOARD")
print("="*60)

print("\nLoading model...")
model=tf.keras.models.load_model(MODEL_FILE)
print("Model loaded.")

print("Loading scaler...")
scaler=joblib.load(SCALER_FILE)
print("Scaler loaded.")

expected_shape=(WINDOW_SIZE,len(FEATURE_COLUMNS))
if model.input_shape[1:]!=expected_shape:
    raise ValueError(f"Model input shape {model.input_shape} does not match expected {expected_shape}")

print(f"Model input : {model.input_shape}")
print(f"Window      : {WINDOW_SIZE} samples ({WINDOW_SIZE/SAMPLE_RATE:.1f}s)")
print(f"Sensor rate : {SAMPLE_RATE} Hz")

def add_history(activity,confidence=0.0,is_fall=False):
    now=datetime.now().strftime("%H:%M:%S")
    history.appendleft({
        "time":now,
        "activity":"Fall Detected" if is_fall else activity,
        "fall":is_fall
    })

def play_alert():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

def build_alerts():
    global last_alerts
    alerts=[]
    if dashboard["activity"]=="Falling":
        alerts.append("Fall Detected")
    hr=dashboard["heart_rate"]
    spo2=dashboard["spo2"]
    if hr is not None:
        if hr<60:
            alerts.append("Low Heart Rate")
        elif hr>100:
            alerts.append("High Heart Rate")
    if spo2 is not None and spo2<95:
        alerts.append("Low SpO₂")
    new_alerts=[a for a in alerts if a not in last_alerts]
    if new_alerts:
        play_alert()
    last_alerts=alerts.copy()
    return alerts

def update_activity(predicted_class,confidence,device_ip):
    global previous_state,activity_started_at

    if confidence<75:
        return

    # Keep FALLING after a fall when the model sees Sleeping or Still.
    if previous_state==1 and predicted_class in (3,5):
        predicted_class=1

    # Ignore Sitting and Standing.
    if predicted_class in (2,4):
        return

    activity=LABELS.get(predicted_class)
    if not activity:
        return

    now=time.time()

    with state_lock:
        if predicted_class!=previous_state:
            previous_state=predicted_class
            activity_started_at=now
            add_history(activity,confidence,predicted_class==1)

        dashboard["activity"]=activity
        dashboard["confidence"]=round(confidence,2)
        dashboard["device_connected"]=True
        dashboard["device_ip"]=device_ip
        dashboard["last_update"]=now
        dashboard["last_motion"]=now
        dashboard["duration"]=int(now-activity_started_at)
        dashboard["alerts"]=build_alerts()

def parse_motion_packet(data):
    packet_str=data.decode("utf-8",errors="replace").strip()
    lines=packet_str.splitlines()
    if not lines:
        return []

    header=lines[0].split(",")
    if len(header)!=3 or header[0]!="FRAME":
        return []

    rows=[]
    for line in lines[1:]:
        values=line.split(",")
        if len(values)!=8:
            continue
        try:
            rows.append([
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
                float(values[5]),
                float(values[6])
            ])
        except ValueError:
            continue
    return rows

def parse_pulse_packet(data):
    text=data.decode("utf-8",errors="replace").strip()
    if not text:
        return None,None

    text=text.replace("HR:","").replace("SpO2:","")
    parts=[p.strip() for p in text.replace(";",",").split(",")]

    if len(parts)<2:
        return None,None

    try:
        hr=float(parts[0])
        spo2=float(parts[1])
        return hr,spo2
    except ValueError:
        return None,None

def udp_listener():
    global total_samples,next_prediction

    sock_motion=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock_motion.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,4*1024*1024)
    sock_motion.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock_motion.bind((UDP_IP,PORT_MOTION))

    sock_pulse=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock_pulse.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock_pulse.bind((UDP_IP,PORT_PULSE))

    sockets_list=[sock_motion,sock_pulse]

    print(f"\nListening on UDP {PORT_MOTION} (Motion) and UDP {PORT_PULSE} (Pulse)")
    print("Flask dashboard: http://0.0.0.0:8000\n")

    try:
        while True:
            readable,_,_=select.select(sockets_list,[],[],1.0)

            for ready_sock in readable:
                data,addr=ready_sock.recvfrom(8192)
                receiving_port=ready_sock.getsockname()[1]
                device_ip=addr[0]

                if receiving_port==PORT_MOTION:
                    rows=parse_motion_packet(data)

                    for row in rows:
                        buffer.append(row)
                        total_samples+=1

                        if total_samples>=next_prediction and len(buffer)==WINDOW_SIZE:
                            x=np.asarray(buffer,dtype=np.float32)
                            x=scaler.transform(x).reshape(1,WINDOW_SIZE,len(FEATURE_COLUMNS))

                            prediction=model.predict(x,verbose=0)[0]
                            predicted_class=int(np.argmax(prediction))+1
                            confidence=float(np.max(prediction))*100

                            update_activity(
                                predicted_class,
                                confidence,
                                device_ip
                            )

                            next_prediction+=STEP_SIZE

                elif receiving_port==PORT_PULSE:
                    hr,spo2=parse_pulse_packet(data)

                    if hr is not None and spo2 is not None:
                        now=time.time()

                        with state_lock:
                            dashboard["heart_rate"]=round(hr,1)
                            dashboard["spo2"]=round(spo2,1)
                            dashboard["alerts"]=build_alerts()
                            dashboard["device_connected"]=True
                            dashboard["device_ip"]=device_ip
                            dashboard["last_pulse"]=now
                            dashboard["last_update"]=now

    except Exception as e:
        print(f"UDP listener error: {e}")
    finally:
        sock_motion.close()
        sock_pulse.close()

def format_duration(seconds):
    seconds=max(0,int(seconds))
    hours,rem=divmod(seconds,3600)
    minutes,seconds=divmod(rem,60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    now=time.time()

    with state_lock:
        connected=(now-dashboard["last_update"])<5
        current=dashboard.copy()
        current["device_connected"]=connected
        current["duration"]=format_duration(now-activity_started_at)
        current["history"]=list(history)
        current["alerts"]=list(dashboard["alerts"])

    return jsonify(current)

if __name__=="__main__":
    listener=threading.Thread(target=udp_listener,daemon=True)
    listener.start()
    app.run(host="0.0.0.0",port=8000,debug=False,threaded=True)