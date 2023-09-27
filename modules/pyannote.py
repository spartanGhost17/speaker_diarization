from dotenv import load_dotenv
import os
from pyannote.audio import Pipeline
from pyannote.audio import Model
from pyannote.audio import Inference


import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from src.utils.audio_utils import crop_wav
from src.utils.system_utils import delete_specific_file
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def get_pyannote_access_token(dotenv_path):
    '''
        get pyannote from .env file or environment variables
    '''
    load_dotenv(dotenv_path)
    pyannote_token = os.environ.get('PYANNOTE_ACCESS_TOKEN')
    return pyannote_token

def get_pyannote_pipeline(pyannote_access_token):
    '''
         Get pyannote pipeline
    '''
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token=pyannote_access_token)
    return pipeline


def get_embedding_model(pyannote_access_token):
    '''
        Get embeddings model
    '''
    model = Model.from_pretrained("pyannote/embedding", 
                                use_auth_token=pyannote_access_token)

    return model

def get_speaker_embedding_vector(model, file_path):
    '''
        convert voice signal into vector 
    '''
    inference = Inference(model, window="whole")
    speaker_embedding = inference(file_path)
    return speaker_embedding

def embedding_cosine_similarity(model, speaker1_file, speaker2_file):
    embedding1 = get_speaker_embedding_vector(model, speaker1_file)
    embedding2 = get_speaker_embedding_vector(model, speaker2_file)

    distance = cdist(embedding1.reshape(1, -1), embedding2.reshape(1, -1), metric="cosine")[0,0]
    
    return distance


def get_diarization_speaker_info_df(pipeline, audio_path):
    '''
        get speaker diarization info

        pd.Dataframe of speaker diarized info
    '''
    idx = 0
    segments_info_list = []
    # apply the pipeline to an audio file
    diarization = pipeline(audio_path)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_info = {
            'start': np.round(turn.start, decimals=2),
            'stop': np.round(turn.end, decimals=2),
            'speaker': speaker
        }
        
        df = pd.DataFrame.from_dict({idx: segment_info}, orient='index')
        segments_info_list.append(df)

        idx += 1
    diarized_speaker_info_df = pd.concat(segments_info_list, axis=0)

    return diarized_speaker_info_df

def cluster_audio_chunks(diarized_speaker_info_df, SPEAKER_SEGMENT_PATH, normalized_audio_path, 
                        start_stop_tuples, time_counter, cluster_files_list):
    '''
        get a voice sample for each speaker defined in diarized_speaker_info_df
    '''

    speaker = None
    start = None
    stop = None
    CROPING_CONSTANT = 2.5
    start_stops = []
    if not diarized_speaker_info_df.empty:
        start = diarized_speaker_info_df.iloc[0]['speaker_start']
        stop = diarized_speaker_info_df.iloc[0]['speaker_end']
        speaker = diarized_speaker_info_df.iloc[0]['speaker']

    path_items = normalized_audio_path.split('/')
    original_file = path_items[len(path_items)-1]
    for i, row in diarized_speaker_info_df.iterrows():
        folder_path = f"{SPEAKER_SEGMENT_PATH}"
        current_start = row['speaker_start']
        current_stop = row['speaker_end']
        current_speaker = row['speaker']

        if speaker == current_speaker:
            stop = current_stop
        else:
            #speaker changed crop audio of previous speaker
            speaker = current_speaker
            output_file_name = f"{folder_path}crop_normal_{original_file}__{i}_{time.strftime('%Y%m%d_%H%M%S')}.wav"

            if float(stop) - float(start) >= float(CROPING_CONSTANT):
                start_stops.append((start, current_stop, time_counter.value))
                start_stop_tuples.append((start, current_stop, time_counter.value))
                cluster_files_list.append(output_file_name)
                crop_wav(normalized_audio_path, output_file_name, start, float(start) + CROPING_CONSTANT)#, stop)

            start = current_start
            stop = current_stop      

        # crop is last entry
        if (speaker == current_speaker) and (i == len(diarized_speaker_info_df) - 1):
            
            output_file_name = f"{folder_path}crop_at_end_{original_file}_{speaker}__{i}_{time.strftime('%Y%m%d_%H%M%S')}.wav"

            if float(stop) - float(start) >= float(CROPING_CONSTANT):

                print(f"should crop end {(start, current_stop, time_counter.value)}")
                start_stop_tuples.append((start, current_stop, time_counter.value))
                crop_wav(normalized_audio_path, output_file_name, start, float(start)+CROPING_CONSTANT)#, stop)
                cluster_files_list.append(output_file_name)
        

def init_embeddings_cluster(CLUSTER_ROOT, embedding_model, cluster_point_cloud):

    '''
        initialize the embedding cluster dictionary
    '''
    cluster_files = os.listdir(CLUSTER_ROOT) 
    embeddings_dict = {}
    for i, file in enumerate(cluster_files):
        file_path = os.path.join(CLUSTER_ROOT, file)

        if file_path not in embeddings_dict:
            embeddings_dict[file_path] = get_speaker_embedding_vector(embedding_model, file_path)
        

    embedding_arrays = np.array(list(embeddings_dict.values()))
    cosine_similarity_matrix = cosine_similarity(embedding_arrays, embedding_arrays)

    # Compute linkage matrix
    linkage_matrix = linkage(1 - cosine_similarity_matrix, method='ward')

    # Set a threshold or number of clusters based on your visual inspection
    threshold = 1.6  # Adjust this value as needed

    # Cut the dendrogram to obtain cluster labels
    cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')

    # Create a dictionary to associate file names with cluster labels
    clustered_data = {}
    clustered_embeddings = {}
    for i, (file_name, embedding) in enumerate(embeddings_dict.items()):
        cluster_label = cluster_labels[i]
        if cluster_label not in clustered_data:
            clustered_data[cluster_label] = []
            clustered_embeddings[cluster_label] = []
        clustered_data[cluster_label].append(file_name)
        clustered_embeddings[cluster_label].append(embedding)
        
        #reset cluster label for each point
        cluster_point_cloud[file_name].add_cluster_label(cluster_label)

    # Now, clustered_data contains clusters as keys and lists of file names as values
    #for i in clustered_data:
    #    print("cluster: ", i, " ",clustered_data[i],"\n")
    return clustered_embeddings

def get_speaker_key(speaker_embeddings, clustered_embeddings_dict):
    '''
        Compute cosine similarity for each pair of arrays

    '''
    dict_after_clustering = {}

    dict_after_clustering[2] = speaker_embeddings#get_speaker_embedding_vector(embedding_model, file_path)
    similarities = []

    for _, (_, v) in enumerate(dict_after_clustering.items()):
        for _, (_, v_1) in enumerate(clustered_embeddings_dict.items()):
            v = np.array(list(v))
            v_1 = np.array(list(v_1))
            if len(v.shape) < 2:
                v = v.reshape(1, -1)
            if len(v_1.shape) < 2:
                v_1 = v_1.reshape(1, -1)
            
            similarity = cosine_similarity(v, v_1)[0][0]
            similarities.append(similarity)

    # The similarities list now contains the cosine similarities for each pair of arrays
    max_index = np.argmax(similarities)

    # if similarity score is less than 60% this is most likely a new speaker
    if similarities[max_index] < 0.55:#0.6: 
        max_index = -1 

    return max_index, similarities[max_index]
