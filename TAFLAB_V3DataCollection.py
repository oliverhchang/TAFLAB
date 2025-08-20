import serial
import time


import csv
import os
from datetime import datetime

SERIAL_PORT = '/dev/cu.usbmodem11101'
BAUD_RATE = 115200
CSV_FILENAME = os.path.expanduser('~/Downloads/serial_data.csv')

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
ser.reset_input_buffer()

with open(CSV_FILENAME, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Timestamp', 'Data'])  # Header row

try:
    with open(CSV_FILENAME, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        prev_time = time.time()

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    now = datetime.now()
                    dt = time.time() - prev_time
                    prev_time = time.time()

                    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    print(f"{timestamp} | Δt={dt:.3f}s | Data: {line}")
                    writer.writerow([timestamp, line])
                    csvfile.flush()
            time.sleep(0.001)

except KeyboardInterrupt:
    ser.close()
    print(f"\nStopped - Data saved to {CSV_FILENAME}")
except Exception as e:
    ser.close()
    print(f"Error: {e}")
    print(f"Data saved to {CSV_FILENAME}")
