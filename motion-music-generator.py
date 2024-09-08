import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from IPython.display import Audio, display
import csv

class ImprovedMotionMusicGenerator:
    def __init__(self):
        self.sample_rate = 44100
        self.duration = 10  #seconds
        self.bpm = 120
        self.scales = {
            'C_major': [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88],
            'A_minor': [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00],
            'D_dorian': [293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25],
            'G_mixolydian': [392.00, 440.00, 493.88, 523.25, 587.33, 659.25, 698.46]
        }
        self.chord_progressions = [
            [0, 3, 4], [3, 4, 0], [4, 0, 3],
            [0, 5, 3, 4], [2, 5, 0, 3],
            [0, 4, 5, 3], [3, 0, 4, 5]
        ]
    '''use this function if you  want to generate a random chord progression without your phone sensors'''
    #def simulate_motion_data(self):
    #    t = np.linspace(0, self.duration, num=int(self.sample_rate * self.duration))
    #    frequencies = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    #    noise = 1
    #    accel_x = sum([np.sin(2 * np.pi * f * t) for f in frequencies[:2]]) + np.random.normal(0, noise, t.shape)
    #    accel_y = sum([np.sin(2 * np.pi * f * t) for f in frequencies[1:3]]) + np.random.normal(0, noise, t.shape)
    #    accel_z = sum([np.sin(2 * np.pi * f * t) for f in frequencies[2:4]]) + np.random.normal(0, noise, t.shape)
    #    gyro_x = sum([np.cos(2 * np.pi * f * t) for f in frequencies[3:5]]) + np.random.normal(0, noise, t.shape)
    #    gyro_y = sum([np.cos(2 * np.pi * f * t) for f in frequencies[1:4]]) + np.random.normal(0, noise, t.shape)
    #    gyro_z = sum([np.cos(2 * np.pi * f * t) for f in frequencies[2:5]]) + np.random.normal(0, noise, t.shape)
    #    return t, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

    '''use this function if you  want to generate a chord progression with your phone sensors'''
    def load_motion_data(self, filename):
        t = []
        accel_x = []
        accel_y = []
        accel_z = []
        gyro_x = []
        gyro_y = []
        gyro_z = []
        
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                t.append(float(row['Time']))
                accel_x.append(float(row['Accel_X']))
                accel_y.append(float(row['Accel_Y']))
                accel_z.append(float(row['Accel_Z']))
                gyro_x.append(float(row['Gyro_X']))
                gyro_y.append(float(row['Gyro_Y']))
                gyro_z.append(float(row['Gyro_Z']))
        
        t = np.array(t)
        accel_x = np.array(accel_x)
        accel_y = np.array(accel_y)
        accel_z = np.array(accel_z)
        gyro_x = np.array(gyro_x)
        gyro_y = np.array(gyro_y)
        gyro_z = np.array(gyro_z)
        
        return t, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    def generate_note(self, frequency, duration, amplitude=0.5):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        note = amplitude * np.sin(2 * np.pi * frequency * t)
        
        attack = int(0.1 * len(t))
        decay = int(0.2 * len(t))
        sustain = int(0.5 * len(t))
        release = len(t) - attack - decay - sustain
        
        envelope = np.concatenate([
            np.linspace(0, 1, attack),
            np.linspace(1, 0.7, decay),
            np.ones(sustain) * 0.7,
            np.linspace(0.7, 0, release)
        ])
        
        return note * envelope

    def generate_chord(self, root_freq, duration, chord_type='major'):
        root = self.generate_note(root_freq, duration)
        if chord_type == 'major':
            third = self.generate_note(root_freq * 1.25, duration, 0.3)
            fifth = self.generate_note(root_freq * 1.5, duration, 0.3)
        elif chord_type == 'minor':
            third = self.generate_note(root_freq * 1.2, duration, 0.3)
            fifth = self.generate_note(root_freq * 1.5, duration, 0.3)
        elif chord_type == 'diminished':
            third = self.generate_note(root_freq * 1.2, duration, 0.3)
            fifth = self.generate_note(root_freq * 1.4, duration, 0.3)
        else:  #augmented
            third = self.generate_note(root_freq * 1.25, duration, 0.3)
            fifth = self.generate_note(root_freq * 1.6, duration, 0.3)
        return root + third + fifth

    def map_motion_to_music(self, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
        beat_duration = 60 / self.bpm
        num_beats = int(self.duration / beat_duration)
        melody = np.zeros(int(self.sample_rate * self.duration))
        
        motion_intensity = np.mean(np.abs(accel_x) + np.abs(accel_y) + np.abs(accel_z))
        scale_index = min(int(motion_intensity * len(self.scales)), len(self.scales) - 1)
        scale_name = list(self.scales.keys())[scale_index]
        scale = self.scales[scale_name]
        
        for i in range(num_beats):
            t_start = int(i * beat_duration * self.sample_rate)
            t_end = min(int((i + 1) * beat_duration * self.sample_rate), len(melody))
            
            note_index = int(np.interp(accel_x[t_start], (-2, 2), (0, len(scale) - 1)))
            frequency = scale[note_index]
            amplitude = np.interp(np.abs(gyro_y[t_start]), (0, 2), (0.3, 0.7))
            duration = min(np.interp(np.abs(gyro_z[t_start]), (0, 2), (0.1, 0.5)) * beat_duration, beat_duration)
            
            note = self.generate_note(frequency, duration, amplitude)
            note_length = min(len(note), t_end - t_start)
            melody[t_start:t_start + note_length] += note[:note_length]
        
        chords = np.zeros_like(melody)
        chord_duration = 4 * beat_duration
        chord_progression = self.chord_progressions[int(np.mean(gyro_x) * len(self.chord_progressions)) % len(self.chord_progressions)]
        chord_types = ['major', 'minor', 'diminished', 'augmented']
        
        for i, chord in enumerate(chord_progression):
            t_start = int(i * chord_duration * self.sample_rate)
            t_end = min(int((i + 1) * chord_duration * self.sample_rate), len(chords))
            root_freq = scale[chord]
            chord_type = chord_types[int(np.abs(accel_z[t_start]) * len(chord_types)) % len(chord_types)]
            chord_notes = self.generate_chord(root_freq, chord_duration, chord_type)
            chord_length = min(len(chord_notes), t_end - t_start)
            chords[t_start:t_start + chord_length] += chord_notes[:chord_length]
        
        reverb = np.zeros_like(melody)
        reverb_time = int(1.5 * self.sample_rate)
        for i in range(len(melody)):
            if i < reverb_time:
                reverb[i] = np.sum(melody[:i] * np.linspace(0.3, 0, i))
            else:
                reverb[i] = np.sum(melody[i-reverb_time:i] * np.linspace(0.3, 0, reverb_time))
        
        music = melody * 0.5 + chords * 0.3 + reverb * 0.2
        return music / np.max(np.abs(music))

    def generate_music(self, csv_filename):
        t, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = self.load_motion_data(csv_filename)
        '''change load_motion_data to simulate_motion_data if you want to use random input generator'''

        music = self.map_motion_to_music(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
        return t, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, music

    def plot_results(self, t, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, audio):
        fig, axs = plt.subplots(4, 1, figsize=(12, 16))
        
        axs[0].plot(t, accel_x, label='X')
        axs[0].plot(t, accel_y, label='Y')
        axs[0].plot(t, accel_z, label='Z')
        axs[0].set_title('Accelerometer Data')
        axs[0].legend()
        
        axs[1].plot(t, gyro_x, label='X')
        axs[1].plot(t, gyro_y, label='Y')
        axs[1].plot(t, gyro_z, label='Z')
        axs[1].set_title('Gyroscope Data')
        axs[1].legend()
        
        axs[2].specgram(audio, Fs=self.sample_rate, scale='dB', sides='default')
        axs[2].set_title('Spectrogram of Generated Music')
        axs[2].set_ylabel('Frequency [Hz]')
        
        axs[3].plot(t, audio)
        axs[3].set_title('Generated Music Waveform')
        axs[3].set_xlabel('Time [s]')
        
        plt.tight_layout()
        plt.show()

    def save_audio(self, audio, filename="motion_music.wav"):
        scaled = np.int16(audio * 32767)
        wavfile.write(filename, self.sample_rate, scaled)
        print(f"Audio saved as {filename}")

    '''use this function if you generate random input data''' 
    #def save_motion_data(self, t, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, filename="motion_data.csv"):
    #    with open(filename, 'w', newline='') as csvfile:
    #        writer = csv.writer(csvfile)
    #        writer.writerow(['Time', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'])
    #        for i in range(len(t)):
    #            writer.writerow([t[i], accel_x[i], accel_y[i], accel_z[i], gyro_x[i], gyro_y[i], gyro_z[i]])
    #    print(f"Motion data saved as {filename}")

#Run the music generation
csv_filename = 'motion_data.csv'
generator = ImprovedMotionMusicGenerator()
t, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, audio = generator.generate_music(csv_filename)
print("hey")
#Plot results
generator.plot_results(t, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, audio)

#Save the generated audio
generator.save_audio(audio)

#Save the motion data
#generator.save_motion_data(t, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)

print("Motion-based music generation complete!")
print("To access the generated data:")
print("1. 'motion_music.wav': The generated music file.")
print("2. 'motion_data.csv': CSV file containing the simulated motion data.")
print("Use your preferred audio player for the WAV file and a spreadsheet program or data analysis tool for the CSV file.")

#Optionally, if you're running this in a local Jupyter notebook, you can try:
#display(Audio(audio, rate=generator.sample_rate))
