# Smart Audio Security System using Convolutional Neural Network
# Developer: Charles Adriel Del Rosario

# Import libraries
import pyaudio
import numpy as np
import librosa
import pywt
import time
from tensorflow.keras.models import load_model
import threading
import queue
import serial
import serial.tools.list_ports
import sys
import logging
import yaml
from contextlib import contextmanager

# Load configuration
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load model and classes
model = load_model(config['model_path'])
classes = config['classes']

# Audio recording parameters
FORMAT = pyaudio.paFloat32
CHANNELS = config['audio']['channels']
RATE = config['audio']['rate']
CHUNK = int(RATE * config['audio']['chunk_duration'])

# Global variables
last_alert_time = 0
is_paused = False
audio_queue = queue.Queue()
ser = None
stream = None
p = None

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "Arduino" in p.description:
            return p.device
    return None

@contextmanager
def serial_context():
    global ser
    if config['use_serial']:
        try:
            port = find_arduino_port()
            if port:
                ser = serial.Serial(port, 9600, timeout=1)
                logger.info(f"Serial connection established on {port}")
                yield ser
            else:
                logger.warning("Arduino not found. Check connection.")
                yield None
        except serial.SerialException as e:
            logger.error(f"Error opening serial port: {e}")
            yield None
        finally:
            if ser:
                ser.close()
    else:
        logger.info("Serial communication disabled.")
        yield None

def preprocess_audio(audio_data, target_shape=(128, 128)):
    coeffs = pywt.wavedec(audio_data, 'db4', level=5)
    wavelet_features = np.concatenate([coeff for coeff in coeffs])
    total_elements = target_shape[0] * target_shape[1]
    wavelet_features = np.pad(wavelet_features, (0, max(0, total_elements - len(wavelet_features))))[:total_elements]
    return np.expand_dims(wavelet_features.reshape(target_shape), axis=(0, -1))

def predict_audio(audio_data):
    processed_audio = preprocess_audio(audio_data)
    prediction = model.predict(processed_audio)
    predicted_class_index = np.argmax(prediction)
    return classes[predicted_class_index], prediction[0][predicted_class_index]

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

def stop_recording():
    global stream, p
    if stream:
        stream.stop_stream()
        stream.close()
    if p:
        p.terminate()
    logger.info("Recording stopped")

def start_recording():
    global stream, p
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                    frames_per_buffer=CHUNK, stream_callback=audio_callback)
    logger.info("Recording started")

def process_audio_thread():
    global last_alert_time, is_paused
    cooldown_period = config['cooldown_period']
    pause_duration = config['pause_duration']

    with serial_context() as serial_conn:
        while True:
            try:
                if is_paused:
                    time.sleep(0.1)
                    continue

                audio_data = audio_queue.get(timeout=1)
                audio_np = np.frombuffer(audio_data, dtype=np.float32).copy()
                predicted_class, confidence = predict_audio(audio_np)
                
                if confidence < config['confidence_threshold']:
                    predicted_class = 'others'

                current_time = time.time()
                time_since_last_alert = current_time - last_alert_time

                if predicted_class in ['others', 'vocal']:
                    safety_status = 'safe'
                    arduino_command = b'1'
                else:
                    if time_since_last_alert >= cooldown_period:
                        safety_status = 'not safe'
                        arduino_command = b'0'
                        last_alert_time = current_time
                        
                        logger.info(f"Alert triggered. Stopping recording for {cooldown_period} seconds...")
                        stop_recording()
                        time.sleep(cooldown_period)
                        
                        logger.info(f"Pausing for additional {pause_duration - cooldown_period} seconds...")
                        time.sleep(pause_duration - cooldown_period)
                        
                        logger.info("Restarting recording...")
                        start_recording()
                    else:
                        logger.info(f"Cooldown active. {cooldown_period - time_since_last_alert:.1f} seconds remaining.")
                        continue

                logger.info(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}, Status: {safety_status}")

                if config['use_serial'] and serial_conn:
                    try:
                        serial_conn.write(arduino_command)
                    except serial.SerialException as e:
                        logger.error(f"Error writing to serial port: {e}")
                else:
                    logger.info(f"Would send to Arduino: {arduino_command}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in prediction: {str(e)}")

def main():
    start_recording()
    
    processing_thread = threading.Thread(target=process_audio_thread)
    processing_thread.daemon = True
    processing_thread.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    finally:
        stop_recording()

if __name__ == "__main__":
    main()