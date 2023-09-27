import modules.pyannote as pyannote
import modules.whisper as whisper
import src.utils.audio_utils as audio_utils
import src.utils.system_utils as system_utils


import keyboard
import os
import time
import multiprocessing
import threading
import time
import random
from queue import Queue



def initialize_folders(root_folder):
    print(">>> Initializing folders system structure...")
    audio_subfolders = ['chunks', 'temp', 'speaker_segments', 'temp/speaker_segements', 'temp/normalized_audio']

    for folder in audio_subfolders:
        system_utils.create_folder_structure(root_folder, folder)

def kill_thread(exit_signal, condition):
    '''
        Help send singal to kill all threads that are currently running
    '''
    print("Keyboard kill_thread setup...")
    while not exit_signal.value:
        if keyboard.is_pressed("q"):
            print("Received 'q'. Signaling threads to exit...")
            exit_signal.value = True
            with condition:  # Acquire the lock associated with the condition
                condition.notify_all()
            break


if __name__ == "__main__":

    DOTENV_PATH = './venv/env_variables.env'
    PYANNOTE_ACCESS_TOKEN = pyannote.get_pyannote_access_token(DOTENV_PATH)

    transcription_model = whisper.load_model()

    pyannote_pipeline = pyannote.get_pyannote_pipeline(pyannote_access_token=PYANNOTE_ACCESS_TOKEN)

    embedding_model = pyannote.get_embedding_model(PYANNOTE_ACCESS_TOKEN)
    
    print("========================================= Finished loading models ===================================================================")
    ROOT_FOLDER = 'test/audio'
    folder_path = 'test/audio/chunks/'
    ENHANCED_OUTPUT_PATH = 'test/audio/temp/normalized_audio/output_normalized_audio.wav'
    MERGED_FILE_PATH = 'test/audio/temp/merged_audio/output_normalized_audio.wav'
    ENHANCED_AUDIO_CHUNKS_PATH = 'test/audio/temp/normalized_audio/'
    SPEAKER_SEGMENT_PATH = 'test/audio/temp/speaker_segements/'

    #initialize_folders(ROOT_FOLDER)

    # Shared variable to signal the threads to exit
    signal = False

    # Create a pool of processes for f3
    max_processes = 4
    pool = multiprocessing.Pool(processes=max_processes)

    # Create a manager object for diarization results
    manager = multiprocessing.Manager()
    diarization_dict = manager.dict()
    speaker_embedding_dict = manager.dict()
    start_stop_tuples = manager.list()
    transcriptions_record = manager.list()
    exit_signal = manager.Value('b', signal)
    time_counter = manager.Value('f', 0.0)

    # Create an empty FIFO queue
    fifo_queue = Queue()
    cluster_files_list = []
    clusterd_files_queue = Queue()
    cluster_point_cloud = {}

    # Create a condition variable
    condition = threading.Condition()

    #transcription thread
    transcription_thread = threading.Thread(target=whisper.continous_transcription, args=(diarization_dict, fifo_queue, exit_signal, condition, SPEAKER_SEGMENT_PATH, transcription_model, time_counter, 
                                                                                          embedding_model, speaker_embedding_dict, start_stop_tuples, clusterd_files_queue, 
                                                                                          transcriptions_record, cluster_point_cloud, cluster_files_list))
    transcription_thread.daemon = True  # Set the thread as a daemon so it exits when the main program exits
    transcription_thread.name = "Thread continous_transcription"

    continous_recording_thread = threading.Thread(target=audio_utils.start_continuous_recording, args=(exit_signal, condition))
    continous_recording_thread.daemon = True
    continous_recording_thread.name = "Thread continous_recording_thread"

    keyboard_thread = threading.Thread(target=kill_thread, args=(exit_signal, condition))
    keyboard_thread.daemon = True # Set the thread as a daemon so it exits when the main program exits 
    keyboard_thread.name = "Keyboard kill_thread"



    # Start the transcription_thread & keyboard_thread
    continous_recording_thread.start()
    transcription_thread.start()
    keyboard_thread.start()

    while not exit_signal.value:
        with condition:
            folder_contents = os.listdir(folder_path)
            #random.shuffle(folder_contents) #shuffle sample
            if len(folder_contents) < 0:
                print("stop run threads")
                
            # Iterate over the current contents of the folder
            for item in folder_contents:
                item_path = os.path.join(folder_path, item)
                
                #check if speech exists in audio
                speech_exists = audio_utils.speech_exists(item_path)

                if speech_exists:
                    ehanced_item_path = os.path.join(ENHANCED_AUDIO_CHUNKS_PATH, item)
                    
                    if ehanced_item_path not in fifo_queue.queue:
                        print(f"\nFile being added to queue for evaluation :-> {item_path}\n")
                        audio_utils.enhance_audio_signal(item_path, ehanced_item_path)
                        fifo_queue.put(ehanced_item_path)
                        condition.notify_all()#notify all threads

                        pool.apply_async(whisper.cont_transcrpt_work, args=(ehanced_item_path, ehanced_item_path, diarization_dict, pyannote_pipeline))
                        system_utils.delete_specific_file(item_path) #delete file
                else:
                    #no speech
                    system_utils.delete_specific_file(item_path) #delete file

    print("closing threads and multiprocessing")

    # Close the pool
    pool.close()

    # Wait for all processes to complete
    pool.join()

    #wait for processes to complete
    continous_recording_thread.join()
    transcription_thread.join()
    keyboard_thread.join()

    # Allow some time for f4 to process any remaining items in diarization_array
    time.sleep(5)  # Adjust the sleep duration as needed