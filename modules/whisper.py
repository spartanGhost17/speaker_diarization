import whisper
import torch
import numpy as np
import pandas as pd
import os
from models.segment import AudioSegment
from models.cluster_point import ClusterPoint
import time
import datetime


from modules.pyannote import get_diarization_speaker_info_df, cluster_audio_chunks, init_embeddings_cluster, get_speaker_key, get_speaker_embedding_vector
from src.utils.audio_utils import speach_activity_detection
from src.utils.system_utils import delete_specific_file
from sklearn.metrics.pairwise import cosine_similarity





DEVICE = "cpu"

def load_model(MODEL_TYPE: str = "base.en"):
    #torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(MODEL_TYPE, device=DEVICE)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
    return model


def get_transcription_object(model, audio_file: str):
    file_name = audio_file
    segments = model.transcribe(file_name)
    
    segments_list = []
    segments_obj_list = []
    idx = 0
    for segment in segments['segments']:
        if 'tokens' in segment.keys():
            segment.pop('tokens')

        df = pd.DataFrame.from_dict({idx: segment}, orient='index')

        segments_list.append(df) #might need to get rid of this
        #print(f"segments_list {segments_list}\n\n")
        segment = AudioSegment(id=df['id'], start=df['start'], end=df['end'], text=df['text'], temperature=df['temperature'], avg_logprob=df['avg_logprob'], compression_ratio=df['compression_ratio'], no_speech_prob=df['no_speech_prob'])

        segments_obj_list.append(segment)

        idx += 1

    output_text = pd.concat(segments_list, axis=0)
    return segments_obj_list, output_text


def transcription_without_overlapping_speakers(diarization, transcript_obj):
    overlaps_list = []
    for _, row in diarization.iterrows():
        start = row['start']
        stop = row['stop']
        speaker = row['speaker']

        xx_inds = ~((transcript_obj[1]['end'] < start) | (transcript_obj[1]['start'] > stop))
        overlapped_text = transcript_obj[1].loc[xx_inds, :]
        overlapped_text['speaker_start'] = start
        overlapped_text['speaker_end'] = stop
        overlapped_text['speaker'] = speaker
        overlaps_list.append(overlapped_text)

    #compute overlap duration
    columns_to_drop = ['seek', 'temperature', 'avg_logprob', 'compression_ratio', 'no_speech_prob']
    all_overlaps = pd.concat(overlaps_list)
    all_overlaps['max_start'] = np.maximum(all_overlaps['start'], 
                                        all_overlaps['speaker_start'])
    all_overlaps['min_end'] = np.minimum(all_overlaps['end'],
                                        all_overlaps['speaker_end'])
    all_overlaps['overlap_duration'] = all_overlaps['min_end'] - all_overlaps['max_start']

    all_overlaps.drop(columns=columns_to_drop, inplace=True)
    all_overlaps = all_overlaps.reset_index(drop=True)

    #pick only one text/speaker combination for each text
    max_overlap_idx = all_overlaps.groupby('id')['overlap_duration'].idxmax()
    all_overlaps = all_overlaps.loc[max_overlap_idx, :]
    return all_overlaps


def rebalance_all_embeddings(speaker_embedding_dict):
    # Create a list to store the filtered entries
    filtered_entries = {}

    # Define a threshold for cosine similarity
    threshold = 0.6

    # Calculate and filter entries based on cosine similarity
    for key1, embedding_list1 in speaker_embedding_dict.items():
        filtered_embeddings = []
        for key2, embedding_list2 in speaker_embedding_dict.items():
            if key1 != key2:  # Exclude self-comparison
                similarity_scores = []
                for emb1 in embedding_list1:
                    for emb2 in embedding_list2:
                        similarity = 1 - cosine_similarity(emb1, emb2) #cosine_similarity(embedding_arrays, embedding_arrays)
                        similarity_scores.append(similarity)
                max_similarity = max(similarity_scores)
                if max_similarity >= threshold:
                    filtered_embeddings.extend(embedding_list2)
        if filtered_embeddings:
            filtered_entries[key1] = filtered_embeddings



def transcribe_new(all_text_overlaps, speaker_embedding_dict, speaker_file, embedding_model, speaker_start, speaker_stop):

    
    if len(speaker_embedding_dict) <= 0: #if no speaker clusters have been created yet
        for _, row in all_text_overlaps.iterrows():
            start = row['start']
            stop = row['end']
            text = row['text']
            speaker = row['speaker']
            
            print(f"start: {start} - stop: {stop}\n unknown speaker: {text} ")
    else:
        
        # get speaker embedding
        speaker_embeddings = get_speaker_embedding_vector(embedding_model, speaker_file)
        #determine which cluster the speaker is part of
        speaker_ID_cluster, confidence = get_speaker_key(speaker_embeddings, speaker_embedding_dict)

        if speaker_ID_cluster == -1: #new speaker discovered
            print(f"\nnew speaker discovered confidence:{1 - confidence}\n")
            print(f"keys {speaker_embedding_dict.keys()}, len: {len(speaker_embedding_dict)}")
            key = len(speaker_embedding_dict) + 1
            speaker_embedding_dict[key] = []
            speaker_embedding_dict[key].append(speaker_embeddings)
            
            #filter text overlap by speaker_start & speaker_stop
            filtered_df = all_text_overlaps[(all_text_overlaps['speaker_start'] == speaker_start) & (all_text_overlaps['speaker_end'] == speaker_stop)]
            #filtered_df_2 = all_text_overlaps[(all_text_overlaps['speaker_start'] > speaker_start)]
            #last_instance = filtered_df#.iloc[-1]
            for _, row in filtered_df.iterrows():
                start = row['start']
                stop = row['end']
                text = row['text']

                print(f"start: {start} - stop: {stop}\n speaker SPEAKER_0{key}: {text} ")

            #for _, row in filtered_df_2.iterrows():
            #    start = row['start']
            #    stop = row['end']
            #    text = row['text']

            #    print(f"start: {start} - stop: {stop}\n speaker unkown: {text} ")
        else:
            speaker_ID_cluster += 1
            speaker_embedding_dict[speaker_ID_cluster].append(speaker_embeddings) #add embedding to list of embedding
            
            #filter text overlap by speaker_start & speaker_stop
            filtered_df = all_text_overlaps[(all_text_overlaps['speaker_start'] == speaker_start) & (all_text_overlaps['speaker_end'] == speaker_stop)]
            #filtered_df_2 = all_text_overlaps[(all_text_overlaps['speaker_start'] > speaker_start)]

            print(f"=========> Known SPEAKER found confidence {confidence} <=========")
            for _, row in filtered_df.iterrows():
                start = row['start']
                stop = row['end']
                text = row['text']
                
                print(f"start: {start} - stop: {stop}\n speaker SPEAKER_0{speaker_ID_cluster}: {text} ")


##!!!!!!    DEPRECATED  !!!!!!
def transcribe(file_name, model, start, end, time_counter):
    '''
        transcribe the audio if not silent

        returns start & stop time of transcription
    '''
    #print(f" transcribe {file_name} ")
    df = speach_activity_detection(filename=file_name)
    start_stop = df.iloc[:][['start', 'stop']].values

    if (len(start_stop) >= 1) and (df.iloc[:][['start', 'stop']].values[0][1] > 0.1):
        current_caption = get_transcription_object(model, file_name)
        if(len(current_caption[1]) <= 1):
            print(f"start: {(start + time_counter.value)} stop: {(end + time_counter.value)} \n    speaker: {current_caption[1].iloc[0]['text']}\n")
        else:
            for _, row in current_caption[1].iterrows():
                print(f"start: {(float(row['start']) + time_counter.value)} stop: {(float(row['end']) + time_counter.value)} \n    speaker: {row['text']}\n")


# Function to process a file using f3_work
def cont_transcrpt_work(file, enhanced_file, diarization_dict, pyannote_pipeline):
    '''
        constinuous transcription worker 
        gets diarization object for file
    '''
    #print(f"file {file} and diarization array {diarization_dict}")
    result = get_diarization_speaker_info_df(pyannote_pipeline, enhanced_file)# Replace this with your actual processing logic to get diarization object
    diarization_dict[file] = result

def check_model_available(fifo_queue, diarization_array):
    exists = False
    for i, path in enumerate(fifo_queue.queue):
        if(path in diarization_array) and (i == 0):
            exists = True
            break
        break # leave if first queue item (file) has not been completed yet
    return exists

# Function to continuously process diarization_array
def continous_transcription(diarization_dict, fifo_queue, exit_signal, condition, SPEAKER_SEGMENT_PATH, 
                            transcription_model, time_counter, embedding_model, speaker_embedding_dict, start_stop_tuples, 
                            clusterd_files_queue, transcriptions_record, cluster_point_cloud, cluster_files_list):

    while not exit_signal.value:#not exit_signal: not pause_event.is_set()
        with condition:
            print(f"        f4")
            should_pop = check_model_available(fifo_queue, diarization_dict)

            if len(fifo_queue.queue) > 0:
                if len(diarization_dict) > 0 and should_pop: #not pause_event.is_set() and 
                    fifo_file_item = fifo_queue.get()
                    diarization_item = diarization_dict.pop(fifo_file_item)

                    print(f"\n!!!!Processing transcription for file: {fifo_file_item}\n")
                    #print(f"    -----------------> timestamp after popping {time.strftime('%Y%m%d_%H%M%S')}")
                    
                    transcript_obj = get_transcription_object(transcription_model, fifo_file_item)                    
                    all_text_overlaps = transcription_without_overlapping_speakers(diarization_item, transcript_obj)

                    transcriptions_record.append(all_text_overlaps)

                    #start_stop_tuples_ = []
                    #start_stop_tuples = cluster_audio_chunks(diarization_item, SPEAKER_SEGMENT_PATH, fifo_file_item)
                    cluster_audio_chunks(all_text_overlaps, SPEAKER_SEGMENT_PATH, fifo_file_item, start_stop_tuples, time_counter, cluster_files_list)

                    #start_stop_tuples.append(start_stop_tuples_)


                        
                        
                    speakers_path = cluster_files_list#os.listdir(SPEAKER_SEGMENT_PATH)

                    for i, path in enumerate(speakers_path):
                        _path = path#os.path.join(SPEAKER_SEGMENT_PATH, path)
                        #print(f"INDEX: {i} file cluster length: {len(clusterd_files_queue.queue)}")
                        #print(f"\n is file \n{_path}   \nin queue {clusterd_files_queue.queue}  \n")
                        #if value in queue in means clustering as well as transcription
                        if _path in clusterd_files_queue.queue:
                            #print(f"    ******** continue file {_path} exists!!! idx {i}")
                            continue
                        #print(f"..............list of tuples?? {start_stop_tuples} index: {i} ")
                        cluster_point = ClusterPoint()
                        cluster_point.add_file_path(_path)
                        
                        clusterd_files_queue.put(_path)
                        #print(f"queue size: {len(clusterd_files_queue.queue)} index {i}\n")

                        clock = int(time_counter.value/10)
                        if clock < 3: #don't rerun for first 30 seconds (0 - 10 - 20)

                            speaker_start = 0#start_stop_tuples[clock][0]#get speaker start #diarization_item.iloc[i]['start']
                            speaker_stop = 0#start_stop_tuples[clock][1]#get speaker stop #diarization_item.iloc[i]['stop']
                            
                            cluster_point.add_start(start_stop_tuples[clock][0])
                            cluster_point.add_end(start_stop_tuples[clock][1])
                            cluster_point.add_counter(time_counter.value)#(start_stop_tuples[clock][2])
                            #print(f"queue size: {len(clusterd_files_queue.queue)} index {i}\n")
                            cluster_point_cloud[_path] = cluster_point
                            
                            transcribe_new(all_text_overlaps, speaker_embedding_dict, _path, embedding_model, speaker_start, speaker_stop) #new transcription process
                            #break
                        else:
                            speaker_start = start_stop_tuples[i][0]#get speaker start #diarization_item.iloc[i]['start']
                            speaker_stop = start_stop_tuples[i][1]#get speaker stop #diarization_item.iloc[i]['stop']
                            
                            cluster_point.add_start(start_stop_tuples[i][0])
                            cluster_point.add_end(start_stop_tuples[i][1])
                            cluster_point.add_counter(time_counter.value)#(start_stop_tuples[i][2])
                            cluster_point_cloud[_path] = cluster_point
                            #transcribe(_path, transcription_model, start_time, stop_time, time_counter)
                            
                            transcribe_new(all_text_overlaps, speaker_embedding_dict, _path, embedding_model, speaker_start, speaker_stop) #new transcription process
                            #delete_specific_file(_path)


                    # 30 seconds in (counted from 0 by jumps of 10)
                    if time_counter.value == 20:
                        speaker_embedding_dict = init_embeddings_cluster(SPEAKER_SEGMENT_PATH, embedding_model, cluster_point_cloud)
                    elif (time_counter.value > 20):# and (time_counter.value % 20 == 0):
                        speaker_embedding_dict = init_embeddings_cluster(SPEAKER_SEGMENT_PATH, embedding_model, cluster_point_cloud)

                    # delete enhanced file and add increment time by 10s
                    delete_specific_file(fifo_file_item)
                    time_counter.value += 10                    
                    print("-----------------------------------------------------------------------\n")

            else:
                print("Multi-threading: audio folder is empty. Pausing...")
                condition.wait()
            time.sleep(1)
    speaker_embedding_dict = init_embeddings_cluster(SPEAKER_SEGMENT_PATH, embedding_model, cluster_point_cloud)
    time.sleep(1)
    export_transcription(transcriptions_record, cluster_point_cloud)
    print("\n kill transcription thread!")

def export_transcription(transcriptions_record, cluster_point_cloud):
    print("printing al clusters\n")
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Define the CSV filename with the timestamp
    csv_filename = f"test/transcripts/transcript_{timestamp}.csv"

    export = []
    for _path, cluster_point in cluster_point_cloud.items():
        counter = cluster_point.get_counter()
        idx = int(counter/10)
        #file_path = cluster_point.get_file_path()
        #index = list(clusterd_files_queue.queue).index(file_path)


        start = cluster_point.get_start()
        end = cluster_point.get_end()

        speaker_cluster_label = cluster_point.get_cluster_label()

        print(f"cluster_point: {cluster_point}")
        
        current_speaker = None
        for i, row in transcriptions_record[idx].iterrows():
            entry = {}
            speaker_start = row['speaker_start']
            speaker_end = row['speaker_end']
            text = row['text']
            current_speaker_df = row['speaker'] 

            entry['text'] = text
            entry['speaker_start'] = speaker_start
            entry['speaker_end'] = speaker_end
            entry['start'] = float(start)
            entry['end'] = float(end)
            entry['clock_counter'] = counter
            entry['file_path'] = _path

            # if speaker start and speaker end are the same as
            # recorded start and end for cluster point than we know it's the
            # same person speaking 
            if (float(start) == float(speaker_start)) and (float(end) == float(speaker_end)):
                print(f"start {start} speaker {speaker_start} end {end} speaker_end {speaker_end}")
                current_speaker = current_speaker_df
                entry['speaker'] = f"SPEAKER_0{speaker_cluster_label}"
            #if the same speaker countius but something wrong with time stamp, tag same person
            elif current_speaker == current_speaker_df:
                print(f"same speaker issue with timestamps" )
                current_speaker = current_speaker_df
                entry['speaker'] = f"SPEAKER_0{speaker_cluster_label}"
            else:
                current_speaker = "UNKOWN_SPEAKER"
                entry['speaker'] = "UNKOWN_SPEAKER"
            
            export.append(entry)
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(export)
    
    # Export the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)

        
    
