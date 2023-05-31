import pyaudio
import wave


## declarations
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 10



def record_audio_chunk(chunk_index):
    WAVE_OUTPUT_FILENAME = f"audio/{chunk_index}_voice_chunk.wav"
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
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

###
#
#
###
def concatenate_stream(path_file1:str, path_file2:str):
    audiofile1 = path_file1#"youraudiofile1.wav"
    audiofile2 = path_file2#"youraudiofile2.wav"

    concantenated_file = "full_audio.wav"
    frames = []

    #wave0 = wave.open(audiofile2,'rb')
    wave0 = wave.open(audiofile1,'rb')
    frames.append([wave0.getparams(),wave0.readframes(wave0.getnframes())])
    wave.close()

    wave1 = wave.open(audiofile2,'rb')
    frames.append([wave1.getparams(),wave1.readframes(wave1.getnframes())])
    wave1.close()

    result = wave.open(concantenated_file,'wb')
    result.setparams(frames[0][0])
    # And the order of concatenation is exactly the order of the writing here :
    result.writeframes(frames[0][1]) #audiofile1
    result.writeframes(frames[1][1]) #audiofile2

    result.close()