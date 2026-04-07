import os
import logging
import h5py
import soundfile as sf
import librosa
import audioread
import numpy as np
import pandas as pd
import csv
import datetime
import collections
import pickle
from mido import MidiFile
from scipy.signal import resample

from piano_vad import (note_detection_with_onset_offset_regress, 
    pedal_detection_with_onset_offset_regress, onsets_frames_note_detection, onsets_frames_pedal_detection)
from note_detection import detect_clean_peaks, map_to_midi_velocity_linear_robust, map_to_midi_velocity_sigmoid
import config

piano_notes = ["A0", "A#0", "B0", "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1",
    "A1", "A#1", "B1", "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2",
    "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3",
    "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4",
    "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5",
    "A5", "A#5", "B5", "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6",
    "A6", "A#6", "B6", "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7",
    "A7", "A#7", "B7", "C8"]


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def traverse_folder(folder):
    paths = []
    names = []
    
    for root, dirs, files in os.walk(folder):
        for name in files:
            filepath = os.path.join(root, name)
            names.append(name)
            paths.append(filepath)
            
    return names, paths


def note_to_freq(piano_note):
    return 2 ** ((piano_note - 39) / 12) * 440

    
def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.
    return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)
    

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


def read_metadata(csv_path):
    """Read metadata of MAESTRO dataset from csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict, dict, e.g. {
        'canonical_composer': ['Alban Berg', ...], 
        'canonical_title': ['Sonata Op. 1', ...], 
        'split': ['train', ...], 
        'year': ['2018', ...]
        'midi_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi', ...], 
        'audio_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav', ...],
        'duration': [698.66116031, ...]}
    """

    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter=',')
        lines = list(reader)

    meta_dict = {'canonical_composer': [], 'canonical_title': [], 'split': [], 
        'year': [], 'midi_filename': [], 'audio_filename': [], 'duration': []}

    for n in range(1, len(lines)):
        meta_dict['canonical_composer'].append(lines[n][0])
        meta_dict['canonical_title'].append(lines[n][1])
        meta_dict['split'].append(lines[n][2])
        meta_dict['year'].append(lines[n][3])
        meta_dict['midi_filename'].append(lines[n][4])
        meta_dict['audio_filename'].append(lines[n][5])
        meta_dict['duration'].append(float(lines[n][6]))

    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])
    
    return meta_dict

def read_hackkey_metadata(csv_path):
    """Read metadata of MAESTRO dataset from csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict, dict
    """

    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter=',')
        lines = list(reader)

    meta_dict = {'source': [], 'year': [], 'hackkey_file': [], 'hackkey_file_info': [], 
                'note_json_file': [], 'pedal_file': [], 'pedal_file_info': [],
                'sound_file': [], 'sound_file_info': [], 'hackkey_duration': [],
                'wav_duration': [], 'duration_diff': [], 'two_piano': [],
                'hackkey_diff': [], 'wav_diff': [], 'pedal_diff': [],
                'aligned_duration': [], 'split': []}

    for n in range(1, len(lines)):
        try:
            meta_dict['source'].append(lines[n][0])
            meta_dict['year'].append(lines[n][1])
            meta_dict['hackkey_file'].append(lines[n][2])
            meta_dict['hackkey_file_info'].append(lines[n][3])
            meta_dict['note_json_file'].append(lines[n][4])
            meta_dict['pedal_file'].append(lines[n][5])
            meta_dict['pedal_file_info'].append(lines[n][6])
            meta_dict['sound_file'].append(lines[n][7])
            meta_dict['sound_file_info'].append(lines[n][8])
            meta_dict['hackkey_duration'].append(float(lines[n][9]) if lines[n][9] != '' else 0)
            meta_dict['wav_duration'].append(float(lines[n][10]) if lines[n][10] != '' else 0)
            meta_dict['duration_diff'].append(float(lines[n][11]) if lines[n][11] != '' else 0)
            meta_dict['two_piano'].append(lines[n][12])
            meta_dict['hackkey_diff'].append(float(lines[n][13]) if lines[n][13] != '' else 0)
            meta_dict['wav_diff'].append(float(lines[n][14]) if lines[n][14] != '' else 0)
            meta_dict['pedal_diff'].append(float(lines[n][15]) if lines[n][15] != '' else 0)
            meta_dict['aligned_duration'].append(float(lines[n][16]) if lines[n][16] != '' else 0)
            meta_dict['split'].append(lines[n][17])
        except Exception as e:
            print(n)
            exit(e)


    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])
    
    print(len(meta_dict['source']))
    return meta_dict

def read_hackkey(hackkey_path):
    piano_notes = ["A0", "A#0", "B0", "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1",
    "A1", "A#1", "B1", "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2",
    "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3",
    "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4",
    "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5",
    "A5", "A#5", "B5", "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6",
    "A6", "A#6", "B6", "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7",
    "A7", "A#7", "B7", "C8"]

    df = pd.read_csv(hackkey_path, header=None, names=['time'] + piano_notes)
    # df = pd.read_csv(hackkey_path, header=0)
    return df

def read_pedal(pedal_path):
    '''
    Load pedal data from CSV file
    :param pedal_path: Path to the pedal CSV file
    :return: DataFrame with time and pedal values
    '''
    pedal_df = pd.read_csv(pedal_path, header=None, names=['time', 'pedal_1', 'pedal_2'])
    return pedal_df

def align_hackkey_wav_pedal_and_remove_silence(meta_dict, dataset_dir):
    '''
    Align hackkey, wav, and pedal files based on the provided meta_dict
    :param meta_dict: Dictionary containing paths to hackkey, wav, and pedal files
    :return: Aligned hackkey DataFrame, wav array, and pedal DataFrame
    '''
    
    wav, sr = librosa.load(os.path.join(dataset_dir, meta_dict['sound_file']), sr=config.sample_rate, mono=True)
    hackkey_df = read_hackkey(os.path.join(dataset_dir, meta_dict['hackkey_file']))
    pedal_df = read_pedal(os.path.join(dataset_dir, meta_dict['pedal_file']))
    
    if meta_dict["hackkey_diff"] > 0:
        min_idx = (np.abs(hackkey_df['time'].values - meta_dict["hackkey_diff"])).argmin()
        hackkey_df = hackkey_df[min_idx:]
        hackkey_df.loc[:, "time"] = hackkey_df["time"] - hackkey_df["time"].iloc[0]
        
    if meta_dict["wav_diff"] > 0:
        wav = wav[int(meta_dict["wav_diff"] * sr):]
        
    if meta_dict["pedal_diff"] > 0:
        min_idx = (np.abs(pedal_df['time'].values - meta_dict["pedal_diff"])).argmin()
        pedal_df = pedal_df[min_idx:]
        pedal_df.loc[:, "time"] = pedal_df["time"] - pedal_df["time"].iloc[0]  
    
    
    #NOTE(Jingjing): This shift is designed to deal with the accumulated delay caused by the sensors. 
    # We fit this function and obatin the coeffient roughly to match the expected timing.
    hackkey_times = hackkey_df['time'].values
    shift = 4.52e-5 * (hackkey_times**1.17) 
    hackkey_times = hackkey_times - shift
    hackkey_df['time'] = hackkey_times

    # Cutting Silence
    hackkey_values = (hackkey_df[piano_notes].values > 2).sum(axis=1)
    active_indices = np.where(hackkey_values > 1)[0]
    start = max(0, active_indices[0] - 50)
    end = min(active_indices[-1] + 2000, len(hackkey_values) - 1)
    
    wav_start = int(hackkey_times[start] * sr)
    wav_end = int(hackkey_times[end] * sr)
    wav = wav[wav_start:wav_end]
    # normalize wav
    wav = wav / np.max(np.abs(wav))
    
    hackkey_df = hackkey_df.iloc[start:end]
    hackkey_df.loc[:, "time"] = hackkey_df["time"] - hackkey_df["time"].iloc[0]
    
    pedal_df = pedal_df.iloc[start:end]
    pedal_df.loc[:, "time"] = pedal_df["time"] - pedal_df["time"].iloc[0]
    
    return hackkey_df, wav, pedal_df   

def align_hackkey_wav_and_remove_silence(meta_dict, dataset_dir):
    '''
    Align hackkey, wav, and pedal files based on the provided meta_dict
    :param meta_dict: Dictionary containing paths to hackkey, wav, and pedal files
    :return: Aligned hackkey DataFrame, wav array, and pedal DataFrame
    '''
    try:
        wav, sr = librosa.load(os.path.join(dataset_dir, meta_dict['sound_file']), sr=config.sample_rate, mono=True)
        hackkey_df = read_hackkey(os.path.join(dataset_dir, meta_dict['hackkey_file']))
        
        # Cutting Silence
        hackkey_values = (hackkey_df[piano_notes].values > 1).sum(axis=1)
        active_indices = np.where(hackkey_values > 0)[0]
        start = max(0, active_indices[0] - 50)
        end = min(active_indices[-1] + 50, len(hackkey_values) - 1)

        wav_start = int(hackkey_df['time'].values[start] * sr)
        wav_end = int(hackkey_df['time'].values[end] * sr)
        wav = wav[wav_start:wav_end]
        # normalize wav
        wav = wav / np.max(np.abs(wav))
        
        hackkey_df = hackkey_df.iloc[start:end]
        hackkey_df.loc[:, "time"] = hackkey_df["time"] - hackkey_df["time"].iloc[0]

        return hackkey_df, wav
    except Exception as e:
        print(e)
        return None, None

def read_midi(midi_path):
    """Parse MIDI file.

    Args:
      midi_path: str

    Returns:
      midi_dict: dict, e.g. {
        'midi_event': [
            'program_change channel=0 program=0 time=0', 
            'control_change channel=0 control=64 value=127 time=0', 
            'control_change channel=0 control=64 value=63 time=236', 
            ...],
        'midi_event_time': [0., 0, 0.98307292, ...]}
    """

    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    assert len(midi_file.tracks) == 2
    """The first track contains tempo, time signature. The second track 
    contains piano events."""

    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []

    ticks = 0
    time_in_second = []

    for message in midi_file.tracks[1]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)

    midi_dict = {
        'midi_event': np.array(message_list), 
        'midi_event_time': np.array(time_in_second)}

    return midi_dict

def read_maps_midi(midi_path):
    """Parse MIDI file of MAPS dataset. Not used anymore.

    Args:
      midi_path: str

    Returns:
      midi_dict: dict, e.g. {
        'midi_event': [
            '<meta message set_tempo tempo=439440 time=0>',
            'control_change channel=0 control=64 value=0 time=0',
            'control_change channel=0 control=64 value=0 time=7531',
            ...],
        'midi_event_time': [0., 0.53200309, 0.53200309, ...]}
    """

    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    assert len(midi_file.tracks) == 1

    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []

    ticks = 0
    time_in_second = []

    for message in midi_file.tracks[0]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)

    midi_dict = {
        'midi_event': np.array(message_list), 
        'midi_event_time': np.array(time_in_second)}

    return midi_dict

def insert_interpolated_points(t_array, v_array, max_gap=0.002):
    """
    Inserts interpolated points into the signal if time gaps exceed max_gap.

    Parameters:
    - t_array (array-like): Original time values (must be sorted).
    - v_array (array-like): Corresponding value array.
    - max_gap (float): Maximum allowed time difference between points.

    Returns:
    - new_t (np.ndarray): New time array with interpolated points.
    - new_v (np.ndarray): New value array with interpolated points.
    """
    t_array = np.asarray(t_array)
    v_array = np.asarray(v_array)

    new_t = []
    new_v = []

    for i in range(len(t_array) - 1):
        new_t.append(t_array[i])
        new_v.append(v_array[i])

        dt = t_array[i+1] - t_array[i]

        if dt > max_gap:
            mid_time = (t_array[i] + t_array[i+1]) / 2
            mid_value = (v_array[i] + v_array[i+1]) / 2
            new_t.append(mid_time)
            new_v.append(mid_value)

    new_t.append(t_array[-1])
    new_v.append(v_array[-1])

    return np.array(new_t), np.array(new_v)

class TargetProcessor(object):
    def __init__(self, segment_seconds, frames_per_second, begin_note, 
        classes_num):
        """Class for processing MIDI events to target.

        Args:
          segment_seconds: float
          frames_per_second: int
          begin_note: int, A0 MIDI note of a piano
          classes_num: int
        """
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.begin_note = begin_note
        self.classes_num = classes_num
        self.max_piano_note = self.classes_num - 1

    def process(self, start_time, midi_events_time, midi_events, 
        extend_pedal=True, note_shift=0):
        """Process MIDI events of an audio segment to target for training, 
        includes: 
        1. Parse MIDI events
        2. Prepare note targets
        3. Prepare pedal targets

        Args:
          start_time: float, start time of a segment
          midi_events_time: list of float, times of MIDI events of a recording, 
            e.g. [0, 3.3, 5.1, ...]
          midi_events: list of str, MIDI events of a recording, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]
          extend_pedal, bool, True: Notes will be set to ON until pedal is 
            released. False: Ignore pedal events.

        Returns:
          target_dict: {
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_onset_roll': (frames_num,), 
            'pedal_offset_roll': (frames_num,), 
            'reg_pedal_onset_roll': (frames_num,), 
            'reg_pedal_offset_roll': (frames_num,), 
            'pedal_frame_roll': (frames_num,)}

          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.64, 'offset_time': 697.00, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 697.00, 'offset_time': 697.19, 'velocity': 50}
            ...]

          pedal_events: list of dict, e.g. [
            {'onset_time': 149.37, 'offset_time': 150.35}, 
            {'onset_time': 150.54, 'offset_time': 152.06}, 
            ...]
        """

        # ------ 1. Parse MIDI events ------
        # Search the begin index of a segment
        for bgn_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time:
                break
        """E.g., start_time: 709.0, bgn_idx: 18003, event_time: 709.0146"""

        # Search the end index of a segment
        for fin_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time + self.segment_seconds:
                break
        """E.g., start_time: 709.0, bgn_idx: 18196, event_time: 719.0115"""

        note_events = []
        """E.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]"""

        pedal_events = []
        """E.g. [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]"""

        buffer_dict = {}    # Used to store onset of notes to be paired with offsets
        pedal_dict = {}     # Used to store onset of pedal to be paired with offset of pedal

        # Backtrack bgn_idx to earlier indexes: ex_bgn_idx, which is used for 
        # searching cross segment pedal and note events. E.g.: bgn_idx: 1149, 
        # ex_bgn_idx: 981
        _delta = int((fin_idx - bgn_idx) * 1.)  
        ex_bgn_idx = max(bgn_idx - _delta, 0)
        
        for i in range(ex_bgn_idx, fin_idx):
            # Parse MIDI messiage
            attribute_list = midi_events[i].split(' ')

            # Note
            if attribute_list[0] in ['note_on', 'note_off']:
                """E.g. attribute_list: ['note_on', 'channel=0', 'note=41', 'velocity=0', 'time=10']"""

                midi_note = int(attribute_list[2].split('=')[1])
                velocity = int(attribute_list[3].split('=')[1])

                # Onset
                if attribute_list[0] == 'note_on' and velocity > 0:
                    buffer_dict[midi_note] = {
                        'onset_time': midi_events_time[i], 
                        'velocity': velocity}

                # Offset
                else:
                    if midi_note in buffer_dict.keys():
                        note_events.append({
                            'midi_note': midi_note, 
                            'onset_time': buffer_dict[midi_note]['onset_time'], 
                            'offset_time': midi_events_time[i], 
                            'velocity': buffer_dict[midi_note]['velocity']})
                        del buffer_dict[midi_note]

            # Pedal
            elif attribute_list[0] == 'control_change' and attribute_list[2] == 'control=64':
                """control=64 corresponds to pedal MIDI event. E.g. 
                attribute_list: ['control_change', 'channel=0', 'control=64', 'value=45', 'time=43']"""

                ped_value = int(attribute_list[3].split('=')[1])
                if ped_value >= 64:
                    if 'onset_time' not in pedal_dict:
                        pedal_dict['onset_time'] = midi_events_time[i]
                else:
                    if 'onset_time' in pedal_dict:
                        pedal_events.append({
                            'onset_time': pedal_dict['onset_time'], 
                            'offset_time': midi_events_time[i]})
                        pedal_dict = {}

        # Add unpaired onsets to events
        for midi_note in buffer_dict.keys():
            note_events.append({
                'midi_note': midi_note, 
                'onset_time': buffer_dict[midi_note]['onset_time'], 
                'offset_time': start_time + self.segment_seconds, 
                'velocity': buffer_dict[midi_note]['velocity']})

        # Add unpaired pedal onsets to data
        if 'onset_time' in pedal_dict.keys():
            pedal_events.append({
                'onset_time': pedal_dict['onset_time'], 
                'offset_time': start_time + self.segment_seconds})

        # Set notes to ON until pedal is released
        if extend_pedal:
            note_events = self.extend_pedal(note_events, pedal_events)
        
        # Prepare targets
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        reg_offset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))
        """mask_roll is used for masking out cross segment notes"""

        pedal_onset_roll = np.zeros(frames_num)
        pedal_offset_roll = np.zeros(frames_num)
        reg_pedal_onset_roll = np.ones(frames_num)
        reg_pedal_offset_roll = np.ones(frames_num)
        pedal_frame_roll = np.zeros(frames_num)

        # ------ 2. Get note targets ------
        # Process note events to target
        for note_event in note_events:
            """note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}"""

            piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note) 
            """There are 88 keys on a piano"""

            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))

                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1

                    offset_roll[fin_frame, piano_note] = 1
                    velocity_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = note_event['velocity']

                    # Vector from the center of a frame to ground truth offset
                    reg_offset_roll[fin_frame, piano_note] = \
                        (note_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1

                        # Vector from the center of a frame to ground truth onset
                        reg_onset_roll[bgn_frame, piano_note] = \
                            (note_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)
                
                    # Mask out segment notes
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            """Get regression targets"""
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])

        # Process unpaired onsets to target
        for midi_note in buffer_dict.keys():
            piano_note = np.clip(midi_note - self.begin_note + note_shift, 0, self.max_piano_note)
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((buffer_dict[midi_note]['onset_time'] - start_time) * self.frames_per_second))
                mask_roll[bgn_frame :, piano_note] = 0     

        # ------ 3. Get pedal targets ------
        # Process pedal events to target
        for pedal_event in pedal_events:
            bgn_frame = int(round((pedal_event['onset_time'] - start_time) * self.frames_per_second))
            fin_frame = int(round((pedal_event['offset_time'] - start_time) * self.frames_per_second))

            if fin_frame >= 0:
                pedal_frame_roll[max(bgn_frame, 0) : fin_frame + 1] = 1

                pedal_offset_roll[fin_frame] = 1
                reg_pedal_offset_roll[fin_frame] = \
                    (pedal_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                if bgn_frame >= 0:
                    pedal_onset_roll[bgn_frame] = 1
                    reg_pedal_onset_roll[bgn_frame] = \
                        (pedal_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)

        # Get regresssion padal targets
        reg_pedal_onset_roll = self.get_regression(reg_pedal_onset_roll)
        reg_pedal_offset_roll = self.get_regression(reg_pedal_offset_roll)

        target_dict = {
            'onset_roll': onset_roll, 'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll, 'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll, 'velocity_roll': velocity_roll, 
            'mask_roll': mask_roll, 'reg_pedal_onset_roll': reg_pedal_onset_roll, 
            'pedal_onset_roll': pedal_onset_roll, 'pedal_offset_roll': pedal_offset_roll, 
            'reg_pedal_offset_roll': reg_pedal_offset_roll, 'pedal_frame_roll': pedal_frame_roll
            }

        return target_dict, note_events, pedal_events

    def extend_pedal(self, note_events, pedal_events):
        """Update the offset of all notes until pedal is released.

        Args:
          note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
          pedal_events: list of dict, e.g., [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]

        Returns:
          ex_note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        """
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)
        ex_note_events = []

        idx = 0     # Index of note events
        while pedal_events: # Go through all pedal events
            pedal_event = pedal_events.popleft()
            buffer_dict = {}    # keys: midi notes, value for each key: event index

            while note_events:
                note_event = note_events.popleft()

                # If a note offset is between the onset and offset of a pedal, 
                # Then set the note offset to when the pedal is released.
                if pedal_event['onset_time'] < note_event['offset_time'] < pedal_event['offset_time']:
                    
                    midi_note = note_event['midi_note']

                    if midi_note in buffer_dict.keys():
                        """Multiple same note inside a pedal"""
                        _idx = buffer_dict[midi_note]
                        del buffer_dict[midi_note]
                        ex_note_events[_idx]['offset_time'] = note_event['onset_time']

                    # Set note offset to pedal offset
                    note_event['offset_time'] = pedal_event['offset_time']
                    buffer_dict[midi_note] = idx
                
                ex_note_events.append(note_event)
                idx += 1

                # Break loop and pop next pedal
                if note_event['offset_time'] > pedal_event['offset_time']:
                    break

        while note_events:
            """Append left notes"""
            ex_note_events.append(note_events.popleft())

        return ex_note_events

    def get_regression(self, input):
        """Get regression target. See Fig. 2 of [1] for an example.
        [1] Q. Kong, et al., High-resolution Piano Transcription with Pedals by 
        Regressing Onsets and Offsets Times, 2020.

        input:
          input: (frames_num,)

        Returns: (frames_num,), e.g., [0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, ...]
        """
        step = 1. / self.frames_per_second
        output = np.ones_like(input)
        
        locts = np.where(input < 0.5)[0] 
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0., 0.05) * 20
        output = (1. - output)

        return output


#(NOTE:Jingjing) We modified the TargetProcessor and created HackkeyTargetProcessor for processing Hackkey data. 
# The main difference is that HackkeyTargetProcessor detects note events based on changes of Hackkey values, 
# while TargetProcessor gets note events from MIDI events.
class HackkeyTargetProcessor(object):
    def __init__(self, segment_seconds, frames_per_second, begin_note, 
        classes_num):
        """Class for processing MIDI events to target.

        Args:
          segment_seconds: float
          frames_per_second: int
          begin_note: int, A0 MIDI note of a piano
          classes_num: int
        """
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.begin_note = begin_note
        self.classes_num = classes_num
        self.max_piano_note = self.classes_num - 1

    def process(self, start_time, hackkey_data, extend_pedal=False, note_shift=0):
        """Process Hackkey of an audio segment to target for training,
        includes:
        1. Parse Hackkey
        2. Prepare note targets
        3. Prepare pedal targets

        Args:
          start_time: float, start time of a segment
          midi_events_time: list of float, times of MIDI events of a recording, 
            e.g. [0, 3.3, 5.1, ...]
          midi_events: list of str, MIDI events of a recording, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]
          extend_pedal, bool, True: Notes will be set to ON until pedal is 
            released. False: Ignore pedal events.
        """

        # ------ 1. Parse Hackkey ------
        hackkey_roll = hackkey_data[:, 1:]
        hackkey_roll[hackkey_roll < 1] = 0
        
        hackkey_time = hackkey_data[:, 0]
        end_time = start_time + self.segment_seconds
        
        hackkey_bgn = int(np.argmin(np.abs(hackkey_time - start_time)))
        # hackkey_fin = int(hackkey_bgn + 1000 * self.segment_seconds - 1)
        hackkey_fin = int(np.argmin(np.abs(hackkey_time - end_time)) - 1)

        hackkey_segment = hackkey_roll[hackkey_bgn: hackkey_fin + 1, :]
        # hackkey_segment[hackkey_segment > 9.8] = 9.8
        hackkey_segment_time = hackkey_time[hackkey_bgn: hackkey_fin + 1] - hackkey_time[hackkey_bgn]
        
        hackkey_segment_time, hackkey_segment = insert_interpolated_points(
            hackkey_segment_time, hackkey_segment, max_gap=0.002) 
        
        keys = np.argwhere(np.max(hackkey_segment, axis=0) > 4).flatten()
        notes = []

        for key in keys:
            signal = hackkey_segment[:, key]
            peaks_info, smoothed_signal = detect_clean_peaks(signal, height_threshold=3.0)
            
            for info in peaks_info:
                onset_time = hackkey_segment_time[info['start_index']]
                onset_index = info['start_index']
                onset_value = info['start_value']
                offset_time = hackkey_segment_time[info['end_index']]
                offset_index = info['end_index']
                offset_value = info['end_value']
                peak_time = hackkey_segment_time[info['peak_index']]
                peak_value = info['peak_value']
                peak_index = info['peak_index']
                escapement_time = peak_time - onset_time
                # Estimate velocity values based on Hackkey values
                velocity = (peak_value - onset_value) / escapement_time            
                velocity = min(127, map_to_midi_velocity_sigmoid(velocity))
            
                hackkey_values = signal[info['start_index']:info['end_index'] + 1]
                smoothed_hackkey_values = smoothed_signal[info['start_index']:info['end_index'] + 1]
                
                note = {
                    "key": int(key),
                    "velocity": int(velocity),
                    "onset": onset_time,
                    "onset_index": onset_index,
                    "offset": offset_time,
                    "offset_index": offset_index,
                    "peak_time": peak_time,
                    "peak_index": peak_index,
                    "hackkey_values": hackkey_values.tolist(),
                    "smoothed_hackkey_values": smoothed_hackkey_values.tolist()
                }
            
                notes.append(note)
        
        # Prepare targets
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        reg_offset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))
        hackkey_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))
        """mask_roll is used for masking out cross segment notes"""

        pedal_onset_roll = np.zeros(frames_num)
        pedal_offset_roll = np.zeros(frames_num)
        reg_pedal_onset_roll = np.ones(frames_num)
        reg_pedal_offset_roll = np.ones(frames_num)
        pedal_frame_roll = np.zeros(frames_num)     
        
        note_events = []
                                             

        # ------ 2. Get note targets ------
        # Process note events to target
        for idx, note in enumerate(notes):
            start = note['onset']
            end = note['offset']
            piano_note = note['key']
            
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round(start * self.frames_per_second))
                fin_frame = int(round(end * self.frames_per_second))

                assert fin_frame >= 0
                frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1

                offset_roll[fin_frame, piano_note] = 1
                # Ensure indices are within bounds and shapes match
                target_start = max(bgn_frame, 0) 
                target_end = fin_frame + 1
                
                # Get source values and ensure shape matches target
                target_length = target_end - target_start
                source_values = resample(note['smoothed_hackkey_values'], target_length)
                
                # ------- 2. Normalize Hackkey -------
                source_values[source_values > 12] = 12
                source_values = source_values / 12.0
                    
                hackkey_roll[target_start:target_end, piano_note] = source_values

                # Vector from the center of a frame to ground truth offset
                reg_offset_roll[fin_frame, piano_note] = \
                    end - (fin_frame / self.frames_per_second)

                assert bgn_frame >= 0
                
                onset_roll[bgn_frame, piano_note] = 1
                velocity_roll[bgn_frame, piano_note] = note['velocity']

                # Vector from the center of a frame to ground truth onset
                reg_onset_roll[bgn_frame, piano_note] = \
                    start - (bgn_frame / self.frames_per_second)
                
                #     # Mask out segment notes
                #     else:
                #         mask_roll[: fin_frame + 1, piano_note] = 0
                    
                # else:
                #     mask_roll[bgn_frame :, piano_note] = 0  
                    
        # for idx, (start, end, piano_note) in enumerate(paired_no_off):
            
        #     start /= 1000
        #     end /= 1000
            
        #     bgn_frame = int(round(start * self.frames_per_second))
        #     if 0 <= piano_note <= self.max_piano_note:
        #         if bgn_frame >= 0:
        #             mask_roll[bgn_frame:, piano_note] = 0
        
        # for idx, (start, end, piano_note) in enumerate(paired_no_on):
            
        #     start /= 1000
        #     end /= 1000
            
        #     fin_frame = int(round(end * self.frames_per_second))
        #     if 0 <= piano_note <= self.max_piano_note:
        #         if fin_frame >= 0:
        #             mask_roll[:fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            """Get regression targets"""
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])       

        # ------ 3. Get pedal targets ------
        # Process pedal events to target
        # for pedal_event in pedal_events:
        #     bgn_frame = int(round((pedal_event['onset_time'] - start_time) * self.frames_per_second))
        #     fin_frame = int(round((pedal_event['offset_time'] - start_time) * self.frames_per_second))

        #     if fin_frame >= 0:
        #         pedal_frame_roll[max(bgn_frame, 0) : fin_frame + 1] = 1

        #         pedal_offset_roll[fin_frame] = 1
        #         reg_pedal_offset_roll[fin_frame] = \
        #             (pedal_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

        #         if bgn_frame >= 0:
        #             pedal_onset_roll[bgn_frame] = 1
        #             reg_pedal_onset_roll[bgn_frame] = \
        #                 (pedal_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)

        # # Get regresssion padal targets
        # reg_pedal_onset_roll = self.get_regression(reg_pedal_onset_roll)
        # reg_pedal_offset_roll = self.get_regression(reg_pedal_offset_roll)

        target_dict = {
            'onset_roll': onset_roll, 'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll, 'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll, 'velocity_roll': velocity_roll, 
            'mask_roll': mask_roll, 'hackkey_roll': hackkey_roll,
            # 'reg_pedal_onset_roll': reg_pedal_onset_roll, 
            # 'pedal_onset_roll': pedal_onset_roll, 'pedal_offset_roll': pedal_offset_roll, 
            # 'reg_pedal_offset_roll': reg_pedal_offset_roll, 'pedal_frame_roll': pedal_frame_roll
            }

        return target_dict, hackkey_segment, notes

    def extend_pedal(self, note_events, pedal_events):
        """Update the offset of all notes until pedal is released.

        Args:
          note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
          pedal_events: list of dict, e.g., [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]

        Returns:
          ex_note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        """
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)
        ex_note_events = []

        idx = 0     # Index of note events
        while pedal_events: # Go through all pedal events
            pedal_event = pedal_events.popleft()
            buffer_dict = {}    # keys: midi notes, value for each key: event index

            while note_events:
                note_event = note_events.popleft()

                # If a note offset is between the onset and offset of a pedal, 
                # Then set the note offset to when the pedal is released.
                if pedal_event['onset_time'] < note_event['offset_time'] < pedal_event['offset_time']:
                    
                    midi_note = note_event['midi_note']

                    if midi_note in buffer_dict.keys():
                        """Multiple same note inside a pedal"""
                        _idx = buffer_dict[midi_note]
                        del buffer_dict[midi_note]
                        ex_note_events[_idx]['offset_time'] = note_event['onset_time']

                    # Set note offset to pedal offset
                    note_event['offset_time'] = pedal_event['offset_time']
                    buffer_dict[midi_note] = idx
                
                ex_note_events.append(note_event)
                idx += 1

                # Break loop and pop next pedal
                if note_event['offset_time'] > pedal_event['offset_time']:
                    break

        while note_events:
            """Append left notes"""
            ex_note_events.append(note_events.popleft())

        return ex_note_events

    def get_regression(self, input):
        """Get regression target. See Fig. 2 of [1] for an example.
        [1] Q. Kong, et al., High-resolution Piano Transcription with Pedals by 
        Regressing Onsets and Offsets Times, 2020.

        input:
          input: (frames_num,)

        Returns: (frames_num,), e.g., [0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, ...]
        """
        step = 1. / self.frames_per_second
        output = np.ones_like(input)
        
        locts = np.where(input < 0.5)[0] 
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0., 0.05) * 20
        output = (1. - output)

        return output


def write_events_to_midi(start_time, note_events, pedal_events, midi_path):
    """Write out note events to MIDI file.

    Args:
      start_time: float
      note_events: list of dict, e.g. [
        {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
        {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
        ...]
      midi_path: str
    """
    from mido import Message, MidiFile, MidiTrack, MetaMessage
    
    # This configuration is the same as MIDIs in MAESTRO dataset
    ticks_per_beat = 384
    beats_per_second = 2
    ticks_per_second = ticks_per_beat * beats_per_second
    microseconds_per_beat = int(1e6 // beats_per_second)

    midi_file = MidiFile()
    midi_file.ticks_per_beat = ticks_per_beat

    # Track 0
    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track0.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track0)

    # Track 1
    track1 = MidiTrack()
    
    # Message rolls of MIDI
    message_roll = []

    for note_event in note_events:
        # Onset
        message_roll.append({
            'time': note_event['onset_time'], 
            'midi_note': note_event['midi_note'], 
            'velocity': note_event['velocity']})

        # Offset
        message_roll.append({
            'time': note_event['offset_time'], 
            'midi_note': note_event['midi_note'], 
            'velocity': 0})

    if pedal_events:
        for pedal_event in pedal_events:
            message_roll.append({'time': pedal_event['onset_time'], 'control_change': 64, 'value': 127})
            message_roll.append({'time': pedal_event['offset_time'], 'control_change': 64, 'value': 0})

    # Sort MIDI messages by time
    message_roll.sort(key=lambda note_event: note_event['time'])

    previous_ticks = 0
    for message in message_roll:
        this_ticks = int((message['time'] - start_time) * ticks_per_second)
        if this_ticks >= 0:
            diff_ticks = this_ticks - previous_ticks
            previous_ticks = this_ticks
            if 'midi_note' in message.keys():
                track1.append(Message('note_on', note=message['midi_note'], velocity=message['velocity'], time=diff_ticks))
            elif 'control_change' in message.keys():
                track1.append(Message('control_change', channel=0, control=message['control_change'], value=message['value'], time=diff_ticks))
    track1.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track1)

    midi_file.save(midi_path)


def plot_waveform_midi_targets(data_dict, start_time, note_events):
    """For debugging. Write out waveform, MIDI and plot targets for an 
    audio segment.

    Args:
      data_dict: {
        'waveform': (meta_dicts_num,),
        'onset_roll': (frames_num, classes_num), 
        'offset_roll': (frames_num, classes_num), 
        'reg_onset_roll': (frames_num, classes_num), 
        'reg_offset_roll': (frames_num, classes_num), 
        'frame_roll': (frames_num, classes_num), 
        'velocity_roll': (frames_num, classes_num), 
        'mask_roll':  (frames_num, classes_num), 
        'reg_pedal_onset_roll': (frames_num,),
        'reg_pedal_offset_roll': (frames_num,),
        'pedal_frame_roll': (frames_num,)}
      start_time: float
      note_events: list of dict, e.g. [
        {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
        {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
    """
    import matplotlib.pyplot as plt

    create_folder('debug')
    audio_path = 'debug/debug.wav'
    midi_path = 'debug/debug.mid'
    fig_path = 'debug/debug.pdf'

    librosa.output.write_wav(audio_path, data_dict['waveform'], sr=config.meta_dict_rate)
    write_events_to_midi(start_time, note_events, midi_path)
    x = librosa.core.stft(y=data_dict['waveform'], n_fft=2048, hop_length=160, window='hann', center=True)
    x = np.abs(x) ** 2

    fig, axs = plt.subplots(12, 1, sharex=True, figsize=(30, 30))
    fontsize = 20
    axs[0].matshow(np.log(x), origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(data_dict['onset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[2].matshow(data_dict['offset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[3].matshow(data_dict['reg_onset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[4].matshow(data_dict['reg_offset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[5].matshow(data_dict['frame_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[6].matshow(data_dict['velocity_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[7].matshow(data_dict['hackkey_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[8].matshow(data_dict['mask_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[9].matshow(data_dict['reg_pedal_onset_roll'][:, None].T, origin='lower', aspect='auto', cmap='jet')
    axs[10].matshow(data_dict['reg_pedal_offset_roll'][:, None].T, origin='lower', aspect='auto', cmap='jet')
    axs[11].matshow(data_dict['pedal_frame_roll'][:, None].T, origin='lower', aspect='auto', cmap='jet')
    axs[0].set_title('Log spectrogram', fontsize=fontsize)
    axs[1].set_title('onset_roll', fontsize=fontsize)
    axs[2].set_title('offset_roll', fontsize=fontsize)
    axs[3].set_title('reg_onset_roll', fontsize=fontsize)
    axs[4].set_title('reg_offset_roll', fontsize=fontsize)
    axs[5].set_title('frame_roll', fontsize=fontsize)
    axs[6].set_title('velocity_roll', fontsize=fontsize)
    axs[7].set_title('hackkey_roll', fontsize=fontsize)
    axs[8].set_title('mask_roll', fontsize=fontsize)
    axs[9].set_title('reg_pedal_onset_roll', fontsize=fontsize)
    axs[10].set_title('reg_pedal_offset_roll', fontsize=fontsize)
    axs[11].set_title('pedal_frame_roll', fontsize=fontsize)
    axs[11].set_xlabel('frames')
    axs[11].xaxis.set_label_position('bottom')
    axs[11].xaxis.set_ticks_position('bottom')
    plt.tight_layout(1, 1, 1)
    plt.savefig(fig_path)

    print('Write out to {}, {}, {}!'.format(audio_path, midi_path, fig_path))



#(NOTE:Jingjing) The function below is for plotting the predicteds outputs. 
# The function below is currently only working for debugging the "Replacement" situation. 
# The "Velocity Output" in the modelactually represents the hacckey values predicted by the model.
def plot_waveform_hackkey_targets(data_dict, start_time):
    """For debugging. Write out waveform, MIDI and plot targets for an 
    audio segment.

    Args:
      data_dict: {
        'waveform': (meta_dicts_num,),
        'onset_roll': (frames_num, classes_num), 
        'offset_roll': (frames_num, classes_num), 
        'reg_onset_roll': (frames_num, classes_num), 
        'reg_offset_roll': (frames_num, classes_num), 
        'frame_roll': (frames_num, classes_num), 
        'velocity_roll': (frames_num, classes_num), 
        'mask_roll':  (frames_num, classes_num)}
      start_time: float
    """
    import matplotlib.pyplot as plt

    create_folder('debug')
    audio_path = f'debug/debug_{start_time}.wav'
    fig_path = f'debug/debug_{start_time}.png'

    sf.write(audio_path, data_dict['waveform'], config.sample_rate)
    x = librosa.core.stft(y=data_dict['waveform'], n_fft=2048, hop_length=160, window='hann', center=True)
    # x = librosa.core.stft(y=data_dict['waveform'], n_fft=2048, hop_length=441, window='hann', center=True)
    x = np.abs(x) ** 2

    fig, axs = plt.subplots(8, 1, sharex=True, figsize=(30, 30))
    fontsize = 20
    axs[0].matshow(np.log(x), origin='lower', aspect='auto', cmap='cool')
    axs[1].matshow(data_dict['onset_roll'].T, origin='lower', aspect='auto', cmap='cool')
    axs[2].matshow(data_dict['offset_roll'].T, origin='lower', aspect='auto', cmap='cool')
    axs[3].matshow(data_dict['reg_onset_roll'].T, origin='lower', aspect='auto', cmap='cool')
    axs[4].matshow(data_dict['reg_offset_roll'].T, origin='lower', aspect='auto', cmap='cool')
    axs[5].matshow(data_dict['frame_roll'].T, origin='lower', aspect='auto', cmap='cool')
    axs[6].matshow(data_dict['hackkey_roll'].T, origin='lower', aspect='auto', cmap='cool')
    axs[7].matshow(data_dict['mask_roll'].T, origin='lower', aspect='auto', cmap='cool')
    # axs[8].matshow(data_dict['reg_pedal_onset_roll'][:, None].T, origin='lower', aspect='auto', cmap='cool')
    # axs[9].matshow(data_dict['reg_pedal_offset_roll'][:, None].T, origin='lower', aspect='auto', cmap='cool')
    # axs[10].matshow(data_dict['pedal_frame_roll'][:, None].T, origin='lower', aspect='auto', cmap='cool')
    axs[0].set_title('Log spectrogram', fontsize=fontsize)
    axs[1].set_title('onset_roll', fontsize=fontsize)
    axs[2].set_title('offset_roll', fontsize=fontsize)
    axs[3].set_title('reg_onset_roll', fontsize=fontsize)
    axs[4].set_title('reg_offset_roll', fontsize=fontsize)
    axs[5].set_title('frame_roll', fontsize=fontsize)
    axs[6].set_title('hackkey_roll', fontsize=fontsize)
    axs[7].set_title('mask_roll', fontsize=fontsize)
    # axs[8].set_title('reg_pedal_onset_roll', fontsize=fontsize)
    # axs[9].set_title('reg_pedal_offset_roll', fontsize=fontsize)
    # axs[10].set_title('pedal_frame_roll', fontsize=fontsize)
    axs[7].set_xlabel('frames')
    axs[7].xaxis.set_label_position('bottom')
    axs[7].xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=144, bbox_inches='tight')

    print('Write out to {}, {}!'.format(audio_path, fig_path))


class RegressionPostProcessor(object):
    def __init__(self, frames_per_second, classes_num, onset_threshold, 
        offset_threshold, frame_threshold, pedal_offset_threshold):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          frames_per_second: int
          classes_num: int
          onset_threshold: float
          offset_threshold: float
          frame_threshold: float
          pedal_offset_threshold: float
        """
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.pedal_offset_threshold = pedal_offset_threshold
        self.begin_note = config.begin_note
        self.velocity_scale = config.velocity_scale

    def output_dict_to_midi_events(self, output_dict):
        """Main function. Post process model outputs to MIDI events.

        Args:
          output_dict: {
            'reg_onset_output': (segment_frames, classes_num), 
            'reg_offset_output': (segment_frames, classes_num), 
            'frame_output': (segment_frames, classes_num), 
            'velocity_output': (segment_frames, classes_num), 
            'reg_pedal_onset_output': (segment_frames, 1), 
            'reg_pedal_offset_output': (segment_frames, 1), 
            'pedal_frame_output': (segment_frames, 1)}

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83}, 
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]

          est_pedal_events: list of dict, e.g. [
            {'onset_time': 0.17, 'offset_time': 0.96}, 
            {'osnet_time': 1.17, 'offset_time': 2.65}]
        """

        # Post process piano note outputs to piano note and pedal events information
        (est_on_off_note_vels, est_pedal_on_offs) = \
            self.output_dict_to_note_pedal_arrays(output_dict)
        """est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity], 
        est_pedal_on_offs: (pedal_events_num, 2), the two columns are: [onset_time, offset_time]"""

        # Reformat notes to MIDI events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)

        if est_pedal_on_offs is None:
            est_pedal_events = None
        else:
            est_pedal_events = self.detected_pedals_to_events(est_pedal_on_offs)

        return est_note_events, est_pedal_events

    def output_dict_to_note_pedal_arrays(self, output_dict):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            'frame_output': (frames_num, classes_num), 
            'velocity_output': (frames_num, classes_num), 
            ...}

        Returns:
          est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time, 
            offset_time, piano_note and velocity. E.g. [
             [39.74, 39.87, 27, 0.65], 
             [11.98, 12.11, 33, 0.69], 
             ...]

          est_pedal_on_offs: (pedal_events_num, 2), the 2 columns are onset_time 
            and offset_time. E.g. [
             [0.17, 0.96], 
             [1.17, 2.65], 
             ...]
        """

        # ------ 1. Process regression outputs to binarized outputs ------
        # For example, onset or offset of [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
        # will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

        # Calculate binarized onset output from regression output
        (onset_output, onset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_onset_output'], 
                threshold=self.onset_threshold, neighbour=2)

        output_dict['onset_output'] = onset_output  # Values are 0 or 1
        output_dict['onset_shift_output'] = onset_shift_output  

        # Calculate binarized offset output from regression output
        (offset_output, offset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_offset_output'], 
                threshold=self.offset_threshold, neighbour=4)

        output_dict['offset_output'] = offset_output  # Values are 0 or 1
        output_dict['offset_shift_output'] = offset_shift_output

        if 'reg_pedal_onset_output' in output_dict.keys():
            """Pedal onsets are not used in inference. Instead, frame-wise pedal
            predictions are used to detect onsets. We empirically found this is 
            more accurate to detect pedal onsets."""
            pass

        if 'reg_pedal_offset_output' in output_dict.keys():
            # Calculate binarized pedal offset output from regression output
            (pedal_offset_output, pedal_offset_shift_output) = \
                self.get_binarized_output_from_regression(
                    reg_output=output_dict['reg_pedal_offset_output'], 
                    threshold=self.pedal_offset_threshold, neighbour=4)

            output_dict['pedal_offset_output'] = pedal_offset_output  # Values are 0 or 1
            output_dict['pedal_offset_shift_output'] = pedal_offset_shift_output

        # ------ 2. Process matrices results to event results ------
        # Detect piano notes from output_dict
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)

        if 'reg_pedal_onset_output' in output_dict.keys():
            # Detect piano pedals from output_dict
            est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict)
 
        else:
            est_pedal_on_offs = None    

        return est_on_off_note_vels, est_pedal_on_offs

    def get_binarized_output_from_regression(self, reg_output, threshold, neighbour):
        """Calculate binarized output and shifts of onsets or offsets from the
        regression results.

        Args:
          reg_output: (frames_num, classes_num)
          threshold: float
          neighbour: int

        Returns:
          binary_output: (frames_num, classes_num)
          shift_output: (frames_num, classes_num)
        """
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        (frames_num, classes_num) = reg_output.shape
        
        for k in range(classes_num):
            x = reg_output[:, k]
            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold and self.is_monotonic_neighbour(x, n, neighbour):
                    binary_output[n, k] = 1

                    """See Section III-D in [1] for deduction.
                    [1] Q. Kong, et al., High-resolution Piano Transcription 
                    with Pedals by Regressing Onsets and Offsets Times, 2020."""
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift

        return binary_output, shift_output

    def is_monotonic_neighbour(self, x, n, neighbour):
        """Detect if values are monotonic in both side of x[n].

        Args:
          x: (frames_num,)
          n: int
          neighbour: int

        Returns:
          monotonic: bool
        """
        monotonic = True
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]:
                monotonic = False
            if x[n + i] < x[n + i + 1]:
                monotonic = False

        return monotonic

    def output_dict_to_detected_notes(self, output_dict):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets, 
          MIDI notes and velocities. E.g.,
            [[39.7375, 39.7500, 27., 0.6638],
             [11.9824, 12.5000, 33., 0.6892],
             ...]
        """
        est_tuples = []
        est_midi_notes = []
        classes_num = output_dict['frame_output'].shape[-1]
 
        for piano_note in range(classes_num):
            """Detect piano notes"""
            est_tuples_per_note = note_detection_with_onset_offset_regress(
                frame_output=output_dict['frame_output'][:, piano_note], 
                onset_output=output_dict['onset_output'][:, piano_note], 
                onset_shift_output=output_dict['onset_shift_output'][:, piano_note], 
                offset_output=output_dict['offset_output'][:, piano_note], 
                offset_shift_output=output_dict['offset_shift_output'][:, piano_note], 
                velocity_output=output_dict['velocity_output'][:, piano_note], 
                frame_threshold=self.frame_threshold)
            
            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        est_tuples = np.array(est_tuples)   # (notes, 5)
        """(notes, 5), the five columns are onset, offset, onset_shift, 
        offset_shift and normalized_velocity"""

        est_midi_notes = np.array(est_midi_notes) # (notes,)

        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        velocities = est_tuples[:, 4]
        
        est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
        """(notes, 3), the three columns are onset_times, offset_times and velocity."""

        est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

        return est_on_off_note_vels

    def output_dict_to_detected_pedals(self, output_dict):
        """Postprocess output_dict to piano pedals.

        Args:
          output_dict: dict, e.g. {
            'pedal_frame_output': (frames_num,),
            'pedal_offset_output': (frames_num,),
            'pedal_offset_shift_output': (frames_num,),
            ...}

        Returns:
          est_on_off: (notes, 2), the two columns are pedal onsets and pedal
            offsets. E.g.,
              [[0.1800, 0.9669],
               [1.1400, 2.6458],
               ...]
        """
        frames_num = output_dict['pedal_frame_output'].shape[0]
        
        est_tuples = pedal_detection_with_onset_offset_regress(
            frame_output=output_dict['pedal_frame_output'][:, 0], 
            offset_output=output_dict['pedal_offset_output'][:, 0], 
            offset_shift_output=output_dict['pedal_offset_shift_output'][:, 0], 
            frame_threshold=0.5)

        est_tuples = np.array(est_tuples)
        """(notes, 2), the two columns are pedal onsets and pedal offsets"""
        
        if len(est_tuples) == 0:
            return np.array([])

        else:
            onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
            offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
            est_on_off = np.stack((onset_times, offset_times), axis=-1)
            est_on_off = est_on_off.astype(np.float32)
            return est_on_off

    def detected_notes_to_events(self, est_on_off_note_vels):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 3), the three columns are onset_times, 
            offset_times and velocity. E.g.
            [[32.8376, 35.7700, 0.7932],
             [37.3712, 39.9300, 0.8058],
             ...]
        
        Returns:
          midi_events, list, e.g.,
            [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
             {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
             ...]
        """
        midi_events = []
        for i in range(est_on_off_note_vels.shape[0]):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0], 
                'offset_time': est_on_off_note_vels[i][1], 
                'midi_note': int(est_on_off_note_vels[i][2]), 
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})

        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        """Reformat detected pedal onset and offsets to events.

        Args:
          pedal_on_offs: (notes, 2), the two columns are pedal onsets and pedal
          offsets. E.g., 
            [[0.1800, 0.9669],
             [1.1400, 2.6458],
             ...]

        Returns:
          pedal_events: list of dict, e.g.,
            [{'onset_time': 0.1800, 'offset_time': 0.9669}, 
             {'onset_time': 1.1400, 'offset_time': 2.6458},
             ...]
        """
        pedal_events = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append({
                'onset_time': pedal_on_offs[i, 0], 
                'offset_time': pedal_on_offs[i, 1]})
        
        return pedal_events

class HackkeyRegressionPostProcessor(object):
    def __init__(self, frames_per_second, classes_num, onset_threshold, 
        offset_threshold, frame_threshold, pedal_offset_threshold):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          frames_per_second: int
          classes_num: int
          onset_threshold: float
          offset_threshold: float
          frame_threshold: float
          pedal_offset_threshold: float
        """
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.pedal_offset_threshold = pedal_offset_threshold
        self.begin_note = config.begin_note
        self.velocity_scale = config.velocity_scale
        
    def output_dict_to_note_arrays(self, output_dict):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            'frame_output': (frames_num, classes_num), 
            'velocity_output': (frames_num, classes_num), 
            ...}

        Returns:
          est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time, 
            offset_time, piano_note and velocity. E.g. [
             [39.74, 39.87, 27, 0.65], 
             [11.98, 12.11, 33, 0.69], 
             ...]

          est_pedal_on_offs: (pedal_events_num, 2), the 2 columns are onset_time 
            and offset_time. E.g. [
             [0.17, 0.96], 
             [1.17, 2.65], 
             ...]
        """

        # ------ 1. Process regression outputs to binarized outputs ------
        # For example, onset or offset of [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
        # will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

        # Calculate binarized onset output from regression output
        (onset_output, onset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_onset_output'], 
                threshold=self.onset_threshold, neighbour=2)

        output_dict['onset_output'] = onset_output  # Values are 0 or 1
        output_dict['onset_shift_output'] = onset_shift_output  

        # Calculate binarized offset output from regression output
        (offset_output, offset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_offset_output'], 
                threshold=self.offset_threshold, neighbour=4)

        output_dict['offset_output'] = offset_output  # Values are 0 or 1
        output_dict['offset_shift_output'] = offset_shift_output

        # if 'reg_pedal_onset_output' in output_dict.keys():
        #     """Pedal onsets are not used in inference. Instead, frame-wise pedal
        #     predictions are used to detect onsets. We empirically found this is 
        #     more accurate to detect pedal onsets."""
        #     pass

        # if 'reg_pedal_offset_output' in output_dict.keys():
        #     # Calculate binarized pedal offset output from regression output
        #     (pedal_offset_output, pedal_offset_shift_output) = \
        #         self.get_binarized_output_from_regression(
        #             reg_output=output_dict['reg_pedal_offset_output'], 
        #             threshold=self.pedal_offset_threshold, neighbour=4)

        #     output_dict['pedal_offset_output'] = pedal_offset_output  # Values are 0 or 1
        #     output_dict['pedal_offset_shift_output'] = pedal_offset_shift_output

        # ------ 2. Process matrices results to event results ------
        # Detect piano notes from output_dict
        est_on_off_note_vels, hackkeys = self.output_dict_to_detected_notes(output_dict)

        # if 'reg_pedal_onset_output' in output_dict.keys():
        #     # Detect piano pedals from output_dict
        #     est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict)
 
        # else:
        #     est_pedal_on_offs = None    

        return est_on_off_note_vels, hackkeys

    def output_dict_to_hackkey(self, output_dict):
        """Main function. Post process model outputs to MIDI events.

        Args:
          output_dict: {
            'reg_onset_output': (segment_frames, classes_num), 
            'reg_offset_output': (segment_frames, classes_num), 
            'frame_output': (segment_frames, classes_num), 
            'hackkey_output': (segment_frames, classes_num), 
            'reg_pedal_onset_output': (segment_frames, 1), 
            'reg_pedal_offset_output': (segment_frames, 1), 
            'pedal_frame_output': (segment_frames, 1)}

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83}, 
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]

          est_pedal_events: list of dict, e.g. [
            {'onset_time': 0.17, 'offset_time': 0.96}, 
            {'osnet_time': 1.17, 'offset_time': 2.65}]
        """

        # Post process piano note outputs to piano note and pedal events information
        # (est_on_off_note_vels, est_pedal_on_offs) = \
        #     self.output_dict_to_note_pedal_arrays(output_dict)
        """est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity], 
        est_pedal_on_offs: (pedal_events_num, 2), the two columns are: [onset_time, offset_time]"""
        est_on_off_note_vels, hackkeys = self.output_dict_to_note_arrays(output_dict)
        
        # Reformat notes to MIDI events
        est_hackkey_events = self.detected_notes_to_hackkey(est_on_off_note_vels, hackkeys)
        return est_hackkey_events

    def get_binarized_output_from_regression(self, reg_output, threshold, neighbour):
        """Calculate binarized output and shifts of onsets or offsets from the
        regression results.

        Args:
          reg_output: (frames_num, classes_num)
          threshold: float
          neighbour: int

        Returns:
          binary_output: (frames_num, classes_num)
          shift_output: (frames_num, classes_num)
        """
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        (frames_num, classes_num) = reg_output.shape
        
        for k in range(classes_num):
            x = reg_output[:, k]
            for n in range(1, neighbour):
                if x[n] > threshold and self.is_monotonic_neighbour_beginning(x, n, neighbour):
                    binary_output[n, k] = 1

                    """See Section III-D in [1] for deduction.
                    [1] Q. Kong, et al., High-resolution Piano Transcription 
                    with Pedals by Regressing Onsets and Offsets Times, 2020."""
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift
        
            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold and self.is_monotonic_neighbour(x, n, neighbour):
                    binary_output[n, k] = 1

                    """See Section III-D in [1] for deduction.
                    [1] Q. Kong, et al., High-resolution Piano Transcription 
                    with Pedals by Regressing Onsets and Offsets Times, 2020."""
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift

        return binary_output, shift_output
    
    def is_monotonic_neighbour_beginning(self, x, n, neighbour):
        """Detect if values are monotonic in one side of x[n] for the beginning
        of the sequence.

        Args:
          x: (frames_num,)
          n: int
          neighbour: int

        Returns:
          monotonic: bool
        """
        monotonic = True
        for i in range(neighbour):
            if x[n + i] < x[n + i + 1]:
                monotonic = False

        return monotonic

    def is_monotonic_neighbour(self, x, n, neighbour):
        """Detect if values are monotonic in both side of x[n].

        Args:
          x: (frames_num,)
          n: int
          neighbour: int

        Returns:
          monotonic: bool
        """
        monotonic = True
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]:
                monotonic = False
            if x[n + i] < x[n + i + 1]:
                monotonic = False

        return monotonic

    def output_dict_to_detected_notes(self, output_dict):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets, 
          MIDI notes and velocities. E.g.,
            [[39.7375, 39.7500, 27., 0.6638],
             [11.9824, 12.5000, 33., 0.6892],
             ...]
        """
        est_tuples = []
        est_hackkey_notes = []
        classes_num = output_dict['frame_output'].shape[-1]
 
        for piano_note in range(classes_num):
            """Detect piano notes"""
            est_tuples_per_note = note_detection_with_onset_offset_regress(
                frame_output=output_dict['frame_output'][:, piano_note], 
                onset_output=output_dict['onset_output'][:, piano_note], 
                onset_shift_output=output_dict['onset_shift_output'][:, piano_note], 
                offset_output=output_dict['offset_output'][:, piano_note], 
                offset_shift_output=output_dict['offset_shift_output'][:, piano_note], 
                # This works for the "Replacement" condition (replaced velocity head by hackkey)
                velocity_output=output_dict['hackkey_output'][:, piano_note], # Since no velocity head is used in the "Replacement" condition, we use hackkey output to temporarily replace velocity here.
                hackkey_output=output_dict['hackkey_output'][:, piano_note], 
                # The following settings for the "Additional" condition (added hackkey head as additional head)
                # velocity_output=output_dict['velocity_output'][:, piano_note], 
                # hackkey_output=output_dict['hackkey_output'][:, piano_note], 
                frame_threshold=self.frame_threshold)
            
            # if est_tuples_per_note != []:
            # est_tuples += [i[0:4] for i in est_tuples_per_note]
            est_tuples += est_tuples_per_note
            est_hackkey_notes += [piano_note] * len(est_tuples_per_note)
            
            # if est_tuples != []:
            #     print(est_tuples)
            #     raise ValueError
        est_tuples_tmp = [i[0:5] for i in est_tuples]
        est_hackkeys = [i[5] for i in est_tuples]
        
        est_tuples = np.asarray(est_tuples_tmp)   # (notes, 5)
        """(notes, 5), the five columns are onset, offset, onset_shift, 
        offset_shift and normalized_velocity"""

        est_hackkey_notes = np.array(est_hackkey_notes) # (notes,)

        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        velocities = est_tuples[:, 4]
                
        est_on_off_note_vels = np.stack((onset_times, offset_times, est_hackkey_notes, velocities), axis=-1)
        """(notes, 4), the four columns are onset_times, offset_times, hackkey_notes and velocities."""

        est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

        return est_on_off_note_vels, est_hackkeys

    def detected_notes_to_hackkey(self, est_on_off_note_vels, hackkeys):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 4), the four columns are onset_times, 
            offset_times, hackkey_notes and velocities. E.g.
            [[32.8376, 35.7700, 27., 0.7932],
             [37.3712, 39.9300, 33., 0.8058],
             ...]
        
        Returns:
          midi_events, list, e.g.,
            [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
             {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
             ...]
        """
        hackkey_events = []
        for i in range(est_on_off_note_vels.shape[0]):
            hackkey_events.append({
                'onset_time': est_on_off_note_vels[i][0], 
                'offset_time': est_on_off_note_vels[i][1], 
                'midi_note': int(est_on_off_note_vels[i][2]), 
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale),
                'hackkey': np.array(hackkeys[i] * 12)}) # Recover values to the original scale.

        return hackkey_events

class OnsetsFramesPostProcessor(object):
    def __init__(self, frames_per_second, classes_num):
        """Postprocess the Googl's onsets and frames system output. Only used
        for comparison.

        Args:
          frames_per_second: int
          classes_num: int
        """
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.begin_note = config.begin_note
        self.velocity_scale = config.velocity_scale
        
        self.frame_threshold = 0.5
        self.onset_threshold = 0.1
        self.offset_threshold = 0.3

    def output_dict_to_midi_events(self, output_dict):
        """Main function. Post process model outputs to MIDI events.

        Args:
          output_dict: {
            'reg_onset_output': (segment_frames, classes_num), 
            'reg_offset_output': (segment_frames, classes_num), 
            'frame_output': (segment_frames, classes_num), 
            'velocity_output': (segment_frames, classes_num), 
            'reg_pedal_onset_output': (segment_frames, 1), 
            'reg_pedal_offset_output': (segment_frames, 1), 
            'pedal_frame_output': (segment_frames, 1)}

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83}, 
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]

          est_pedal_events: list of dict, e.g. [
            {'onset_time': 0.17, 'offset_time': 0.96}, 
            {'osnet_time': 1.17, 'offset_time': 2.65}]
        """

        # Post process piano note outputs to piano note and pedal events information
        (est_on_off_note_vels, est_pedal_on_offs) = \
            self.output_dict_to_note_pedal_arrays(output_dict)
        """est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity], 
        est_pedal_on_offs: (pedal_events_num, 2), the two columns are: [onset_time, offset_time]"""
        
        # Reformat notes to MIDI events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)

        if est_pedal_on_offs is None:
            est_pedal_events = None
        else:
            est_pedal_events = self.detected_pedals_to_events(est_pedal_on_offs)

        return est_note_events, est_pedal_events

    def output_dict_to_note_pedal_arrays(self, output_dict):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            'frame_output': (frames_num, classes_num), 
            'velocity_output': (frames_num, classes_num), 
            ...}

        Returns:
          est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time, 
            offset_time, piano_note and velocity. E.g. [
             [39.74, 39.87, 27, 0.65], 
             [11.98, 12.11, 33, 0.69], 
             ...]

          est_pedal_on_offs: (pedal_events_num, 2), the 2 columns are onset_time 
            and offset_time. E.g. [
             [0.17, 0.96], 
             [1.17, 2.65], 
             ...]
        """

        # Sharp onsets and offsets
        output_dict = self.sharp_output_dict(
            output_dict, onset_threshold=self.onset_threshold, 
            offset_threshold=self.offset_threshold)

        # Post process output_dict to piano notes
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict, 
            frame_threshold=self.frame_threshold)

        if 'reg_pedal_onset_output' in output_dict.keys():
            # Detect piano pedals from output_dict
            est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict)
 
        else:
            est_pedal_on_offs = None    

        return est_on_off_note_vels, est_pedal_on_offs

    def sharp_output_dict(self, output_dict, onset_threshold, offset_threshold):
        """Sharp onsets and offsets. E.g. when threshold=0.3, for a note, 
        [0, 0.1, 0.4, 0.7, 0, 0] will be sharped to [0, 0, 0, 1, 0, 0]
        [0., 0., 1., 0., 0., 0.]

        Args:
          output_dict: {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            ...}
          onset_threshold: float
          offset_threshold: float

        Returns:
          output_dict: {
            'onset_output': (frames_num, classes_num), 
            'offset_output': (frames_num, classes_num)}
        """
        if 'reg_onset_output' in output_dict.keys():
            output_dict['onset_output'] = self.sharp_output(
                output_dict['reg_onset_output'], 
                threshold=onset_threshold)

        if 'reg_offset_output' in output_dict.keys():
            output_dict['offset_output'] = self.sharp_output(
                output_dict['reg_offset_output'], 
                threshold=offset_threshold)

        return output_dict

    def sharp_output(self, input, threshold=0.3):
        """Used for sharping onset or offset. E.g. when threshold=0.3, for a note, 
        [0, 0.1, 0.4, 0.7, 0, 0] will be sharped to [0, 0, 0, 1, 0, 0]

        Args:
          input: (frames_num, classes_num)

        Returns:
          output: (frames_num, classes_num)
        """
        (frames_num, classes_num) = input.shape
        output = np.zeros_like(input)

        for piano_note in range(classes_num):
            loct = None
            for i in range(1, frames_num - 1):
                if input[i, piano_note] > threshold and input[i, piano_note] > input[i - 1, piano_note] and input[i, piano_note] > input[i + 1, piano_note]:
                    loct = i
                else:
                    if loct is not None:
                        output[loct, piano_note] = 1
                        loct = None

        return output

    def output_dict_to_detected_notes(self, output_dict, frame_threshold):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets, 
          MIDI notes and velocities. E.g.,
            [[39.7375, 39.7500, 27., 0.6638],
             [11.9824, 12.5000, 33., 0.6892],
             ...]
        """

        est_tuples = []
        est_midi_notes = []

        for piano_note in range(self.classes_num):
            
            est_tuples_per_note = onsets_frames_note_detection(
                frame_output=output_dict['frame_output'][:, piano_note], 
                onset_output=output_dict['onset_output'][:, piano_note], 
                offset_output=output_dict['offset_output'][:, piano_note], 
                velocity_output=output_dict['velocity_output'][:, piano_note], 
                threshold=frame_threshold)

            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        est_tuples = np.array(est_tuples)   # (notes, 3)
        """(notes, 5), the five columns are onset, offset, onset_shift, 
        offset_shift and normalized_velocity"""

        est_midi_notes = np.array(est_midi_notes) # (notes,)
        
        if len(est_midi_notes) == 0:
            return []
        else:
            onset_times = est_tuples[:, 0] / self.frames_per_second
            offset_times = est_tuples[:, 1] / self.frames_per_second
            velocities = est_tuples[:, 2]
        
            est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
            """(notes, 3), the three columns are onset_times, offset_times and velocity."""

            est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

            return est_on_off_note_vels

    def output_dict_to_detected_pedals(self, output_dict):
        """Postprocess output_dict to piano pedals.

        Args:
          output_dict: dict, e.g. {
            'pedal_frame_output': (frames_num,),
            'pedal_offset_output': (frames_num,),
            'pedal_offset_shift_output': (frames_num,),
            ...}

        Returns:
          est_on_off: (notes, 2), the two columns are pedal onsets and pedal
            offsets. E.g.,
              [[0.1800, 0.9669],
               [1.1400, 2.6458],
               ...]
        """

        frames_num = output_dict['pedal_frame_output'].shape[0]
        
        est_tuples = onsets_frames_pedal_detection(
            frame_output=output_dict['pedal_frame_output'][:, 0], 
            offset_output=output_dict['reg_pedal_offset_output'][:, 0], 
            frame_threshold=0.5)

        est_tuples = np.array(est_tuples)
        """(notes, 2), the two columns are pedal onsets and pedal offsets"""
        
        if len(est_tuples) == 0:
            return np.array([])

        else:
            onset_times = est_tuples[:, 0] / self.frames_per_second
            offset_times = est_tuples[:, 1] / self.frames_per_second
            est_on_off = np.stack((onset_times, offset_times), axis=-1)
            est_on_off = est_on_off.astype(np.float32)
            return est_on_off

    def detected_notes_to_events(self, est_on_off_note_vels):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 3), the three columns are onset_times, 
            offset_times and velocity. E.g.
            [[32.8376, 35.7700, 0.7932],
             [37.3712, 39.9300, 0.8058],
             ...]
        
        Returns:
          midi_events, list, e.g.,
            [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
             {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
             ...]
        """
        midi_events = []
        for i in range(len(est_on_off_note_vels)):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0], 
                'offset_time': est_on_off_note_vels[i][1], 
                'midi_note': int(est_on_off_note_vels[i][2]), 
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})

        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        """Reformat detected pedal onset and offsets to events.

        Args:
          pedal_on_offs: (notes, 2), the two columns are pedal onsets and pedal
          offsets. E.g., 
            [[0.1800, 0.9669],
             [1.1400, 2.6458],
             ...]

        Returns:
          pedal_events: list of dict, e.g.,
            [{'onset_time': 0.1800, 'offset_time': 0.9669}, 
             {'onset_time': 1.1400, 'offset_time': 2.6458},
             ...]
        """
        pedal_events = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append({
                'onset_time': pedal_on_offs[i, 0], 
                'offset_time': pedal_on_offs[i, 1]})
        
        return pedal_events


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        """Contain statistics of different training iterations.
        """
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pkl'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'train': [], 'validation': [], 'test': []}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))
        logging.info('    Dump statistics to {}'.format(self.backup_statistics_path))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = pickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'train': [], 'validation': [], 'test': []}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict


def load_audio(path, sr=22050, mono=True, offset=0.0, duration=None,
    dtype=np.float32, res_type='kaiser_best', 
    backends=[audioread.ffdec.FFmpegAudioFile]):
    """Load audio. Copied from librosa.core.load() except that ffmpeg backend is 
    always used in this function."""

    y = []
    with audioread.audio_open(os.path.realpath(path), backends=backends) as input_file:
        sr_native = input_file.meta_dictrate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))
                               * n_channels)

        n = 0

        for frame in input_file:
            frame = librosa.core.audio.util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)

        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
            if mono:
                y = librosa.core.audio.to_mono(y)

        if sr is not None:
            y = librosa.core.audio.remeta_dict(y, sr_native, sr, res_type=res_type)

        else:
            sr = sr_native

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)