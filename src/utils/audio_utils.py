import pyaudio
import wave
import requests
import tempfile
import numpy as np # remove this//here for testing purposes
import time
import keyboard
import collections
from pydub import AudioSegment

from pydub.silence import detect_nonsilent
from noisereduce import reduce_noise
from scipy.io import wavfile

import os
import pandas as pd

## declarations
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_INTERVAL = 10 # Time in seconds for each recording
DOWNLOAD_PODCAST_PATH = "/audio/podcast/"
CHUNKS_DIR = 'test/audio/chunks/'
#CHUNKS_DIR = 'enhanced_audio/chunks/'


def crop_wav(file_name, output_file_name, start, end):
    '''
        Crop audio file in specified boundaries using start/stop
    '''
    sound = AudioSegment.from_wav(file_name)
    
    sound = sound.set_channels(1).set_frame_rate(16000) #mono channel
    
    start_ms = start * 1000
    end_ms = end * 1000
    #extract the first 60000 frames = 60 seconds
    window = sound[start_ms : end_ms]

    window.export(output_file_name, format="wav")


def speech_exists(filename):
    '''
        Check if speech exists in audio
    '''
    longest_speech = 0
    speech_exist = False
    speech_df = speach_activity_detection(filename=filename)
    
    for i, row in speech_df.iterrows():
        start = row['start']
        stop = row['stop']
        speech_length = stop - start
        if speech_length > longest_speech:
            longest_speech = speech_length
    
    #if longest speech is at least 1 second
    if(longest_speech >= 1):
        speech_exist = True
    
    return speech_exist

def speach_activity_detection(filename):
    '''
        detect speech acitivity with timestamp

        return: data frame of speech activity with timestamps in seconds
    '''
    df = pd.DataFrame()
    timeStamps = {
        'file_name' : [],
        'start': [],
        'stop': [],
        'total_length': []
    }
    # Load each audio segment and apply VAD
    #for filename in os.listdir(segments_folder):
    if filename.endswith('.wav'):
        segment_path = filename #os.path.join(segments_folder, filename)
        
        # Load the audio segment using pydub
        audio = AudioSegment.from_wav(segment_path)

        #normalize audio using max dbFS of audio
        normalized_sound = audio.apply_gain(-audio.max_dBFS) # library adjusts the amplitude (volume level) of an audio segment

        # Define silence threshold for VAD (adjust as needed)
        silence_threshold = -40  #in dB (human conversational speech starts above 40 db)

        
        #Print detected non-silent chunks, which in our case would be spoken words.
        nonsilent_data = detect_nonsilent(audio, min_silence_len=500, silence_thresh=silence_threshold, seek_step=1)

        path_tokens = segment_path.split('/')

        file_id = filename if len(path_tokens) < 3 else path_tokens[2] 
        
        for speech_segment in nonsilent_data:#speech_segments:
            start_stop_list = [chunk/1000 for chunk in speech_segment]
            timeStamps['file_name'].append(file_id)
            timeStamps['start'].append(start_stop_list[0])
            timeStamps['stop'].append(start_stop_list[1])
            timeStamps['total_length'].append(len(normalized_sound)/1000)

        df = pd.DataFrame.from_dict(timeStamps)

    return df


def enhance_audio_signal(audio_path, output_path):
    '''
        Enhance audio and resample signal to 16000Hz
    '''
    target_sr = 16000

    try:
        audio = AudioSegment.from_wav(audio_path)
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return

    # Normalize audio using max dBFS of audio
    normalized_sound = audio.apply_gain(-audio.max_dBFS)

    # Resample audio to the target sampling rate (if needed)
    if audio.frame_rate != target_sr:
        normalized_sound = normalized_sound.set_channels(1).set_frame_rate(target_sr)  # Convert to mono if necessary

    # Convert to numpy array
    #audio_data = normalized_sound.get_array_of_samples()

    # Perform noise reduction
    #reduced_audio_data = reduce_noise(y=audio_data, sr=target_sr)

    # Create an AudioSegment from the reduced audio data
    #reduced_audio = AudioSegment(
    #    reduced_audio_data.tobytes(),
    #    frame_rate=target_sr,
    #    sample_width=reduced_audio_data.dtype.itemsize,
    #    channels=1  # Adjust this if your audio is stereo
    #)

    try:
        # Export the denoised audio to the specified file
        #reduced_audio.export(output_path, format="wav")
        normalized_sound.export(output_path, format="wav")
    except Exception as e:
        print(f"Error exporting audio: {e}")


def save_recorded_audio(frames, filename):
    '''
        save the recorded audio
        frames
        filename
    '''
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def start_continuous_recording(exit_signal, condition):
    '''
        records continuous stream
        and create 10 sec segements until recording stops
    '''
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("\nRecording... Press 'q' to stop.")
    frames = collections.deque(maxlen=int(RATE / CHUNK * RECORD_INTERVAL))
    start_time = time.time()

    while True:
        
        if keyboard.is_pressed("q"):
            break

        data = stream.read(CHUNK)
        frames.append(data)

        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= RECORD_INTERVAL:
            save_filename = f"{CHUNKS_DIR}recorded_audio_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            save_recorded_audio(list(frames), save_filename)
            start_time = time.time()

    save_filename = f"{CHUNKS_DIR}recorded_audio_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    save_recorded_audio(list(frames), save_filename)  # Save any remaining frames

    print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

#@Deprecated
def record_audio_chunk(chunk_index):
    WAVE_OUTPUT_FILENAME = f"audio/chunks/{chunk_index}_voice_chunk.wav"
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_INTERVAL)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def concatenate_stream(path_file1:str, path_file2:str, output_file):
    audiofile1 = path_file1
    audiofile2 = path_file2

    frames = []

    wave0 = wave.open(audiofile1,'rb')
    frames.append([wave0.getparams(),wave0.readframes(wave0.getnframes())])
    wave0.close()

    wave1 = wave.open(audiofile2,'rb')
    frames.append([wave1.getparams(),wave1.readframes(wave1.getnframes())])
    wave1.close()

    result = wave.open(output_file,'wb')
    result.setparams(frames[0][0])
    # And the order of concatenation is exactly the order of the writing here :
    result.writeframes(frames[0][1]) #audiofile1
    result.writeframes(frames[1][1]) #audiofile2

    result.close()


def download_mp3(audio_url:str, file_name:str = None):
    '''
        download mp3 and write to local file

    '''

    #download file
    doc = requests.get(audio_url)
    if file_name:
        tempfile_name = file_name
        
        with open(tempfile_name, 'wb') as file:
            file.write(doc.content)

    else:
        #create temp file
        tempWaveFile = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tempfile_name = tempWaveFile.name

        #write to file
        tempWaveFile.write(doc.content)

    return tempfile_name

