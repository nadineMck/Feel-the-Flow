# Feel-the-Flow

This project generates music dynamically based on motion data collected from sensors such as accelerometers and gyroscopes. The generated music reflects the movement patterns and emotional tones captured in real-time or from pre-recorded motion data.

Features
Motion-to-Music Mapping: Converts motion data into musical elements such as melody and chords, adjusting pitch, rhythm, and intensity based on sensor input.
Custom Chord Progressions: Includes multiple scales and chord progressions that adapt to the intensity of the motion.
Real-Time Audio Output: Generates WAV audio files based on motion data.
Visualization: Plots accelerometer and gyroscope data along with the spectrogram and waveform of the generated music.
Requirements
Python 3.x
Required Python packages:
numpy
matplotlib
scipy
sounddevice 
csv

Input Format
The project reads motion data from a CSV file with the following columns:

Time, Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z

How to Use
Prepare Motion Data:

Record real-time motion data from sensors or use pre-recorded data stored in a CSV file, using physics tools sensors suite.
Run the Generator:

Place your motion data in the same directory as the script.
Modify the script to read the data from your CSV file.
Generate Music:

Run the script to generate the music based on your motion data. A WAV file will be saved, along with a visual plot of the motion data and the music waveform.
Save and Visualize Results:

The generated music will be saved as motion_music.wav.
The motion data can be visualized in graphs showing accelerometer and gyroscope readings.
