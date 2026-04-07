import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pretty_midi import PrettyMIDI, Note, Instrument
from scipy.signal import find_peaks

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

piano_notes = [
    "A0", "A#0", "B0", "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1",
    "A1", "A#1", "B1", "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2",
    "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3",
    "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4",
    "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5",
    "A5", "A#5", "B5", "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6",
    "A6", "A#6", "B6", "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7",
    "A7", "A#7", "B7", "C8"
]

def find_local_min(signal, center, direction, max_look=3000, next_peak=None, previous_end=None, dis=40):
    """
    Find the local minimum from center in the given direction.
    direction: -1 (left), +1 (right)
    """
    end = max(0, center - max_look) if direction == -1 else min(len(signal) - 1, center + max_look)
    idx_range = range(center, end, direction)
    
    local_min_idx = center
    local_min_val = signal[center]
    for i in idx_range:
        # Check if current index satisfies boundary conditions
        if direction == -1 and previous_end is not None and i < previous_end:
            break
        if direction == 1 and next_peak is not None and i > next_peak:
            break
        
        if signal[i] < local_min_val and signal[i] >= 1:  # Only consider values >= 1
            local_min_val = signal[i]
            local_min_idx = i
        elif signal[i] <= 1:
            # If we hit a value of 1, we can stop searching
            local_min_idx = i
            local_min_val = 1
            break
        else: 
            if local_min_val < 7:
                if direction == 1:
                    if i + dis < len(signal):
                        count = 0# Ensure we have enough points ahead
                        for j in range(1, dis):
                            count += 1 if signal[i + j - 1] < signal[i + j] else 0
                        if count >= dis * 0.95: 
                            break# Check if enough points are increasing
                else:
                    if i - dis > 0:  # Ensure we have enough points ahead
                        count = 0
                        for j in range(1, dis):
                            count += 1 if signal[i - j + 1] < signal[i - j] else 0
                        if count >= dis * 0.95:
                            break
        
    
    return local_min_idx

def detect_clean_peaks(signal, height_threshold=5.5, distance=60, prominence=0.5, smoothing_sigma=2, max_look=3000):
    # Smooth the signal to reduce noise
    smoothed = gaussian_filter1d(signal, sigma=smoothing_sigma)
    # smoothed = savgol_filter(signal, window_length=20, polyorder=2)
    # smoothed[smoothed < 2] = 0  # Ensure no negative values after smoothing

    # Find peaks
    peaks, properties = find_peaks(
        smoothed,
        height=height_threshold,
        distance=distance,
        prominence=prominence
    )
    
    # Remove peaks where all values between consecutive peaks are larger than 7.5
    i = 0
    while i < len(peaks) - 1:
        if len(smoothed[peaks[i]:peaks[i+1]]) > 0:
            if np.all(smoothed[peaks[i]:peaks[i+1]] > 7):
                peaks = np.delete(peaks, i+1)
                continue
        i += 1
    

    results = []
    start_idx= 0
    end_idx = 0
    for i in range(len(peaks)):
        start_idx = find_local_min(smoothed, peaks[i], direction=-1, max_look=max_look, previous_end=end_idx)

        if i < len(peaks) - 1:
            end_idx = find_local_min(smoothed, peaks[i], direction=1, max_look=max_look, next_peak=peaks[i + 1])
        else:
            end_idx = find_local_min(smoothed, peaks[i], direction=1, max_look=max_look, next_peak=len(smoothed) - 1)
        
        #correct peak index
        earlier_peak, properties = find_peaks(
            smoothed[start_idx:peaks[i]],
            height=signal[peaks[i]] - 1,
            distance=distance,
            prominence=0.5
        )
        
        if len(earlier_peak) > 0:
            peaks[i] = start_idx + earlier_peak[0]  # Use the last peak found in the range

        results.append({
            "peak_index": peaks[i],
            "peak_value": signal[peaks[i]],
            "start_index": start_idx,
            "start_value": signal[start_idx],
            "end_index": end_idx,
            "end_value": signal[end_idx]
        })

    return results, smoothed

def map_to_midi_velocity_sigmoid(vel, mean=180, std=91):
    """
    Sigmoid-based mapping to compress extremes.
    """
    # Standardize
    x = (vel - mean) / std
    # Sigmoid mapping
    midi_vel = 127 / (1 + np.exp(-x))
    return int(round(midi_vel))

def map_to_midi_velocity_linear_robust(vel, v_min=0, v_max=362.13):
    """
    Linearly map key press velocity to MIDI velocity (0-127), clipping outliers.
    """
    vel_clipped = max(min(vel, v_max), v_min)
    midi_vel = (vel_clipped - v_min) / (v_max - v_min) * 127
    return int(round(midi_vel))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect peaks in piano notes signal.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file containing piano notes data.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output json file with detected peaks.")
    parser.add_argument("--output_midi", type=str, required=True, help="Path to save the output MIDI file with detected peaks.")
    parser.add_argument("--onoff_threshold", type=float, default=1.0, help="Threshold for on/off detection.")
    parser.add_argument("--height_threshold", type=float, default=5.0, help="Threshold for peak height detection.")
    args = parser.parse_args()

    data = pd.read_csv(args.input_file, header=None, names=['time'] + piano_notes)
    
    hackkey_times = data['time'].values
    # shift = 4.52e-5 * (hackkey_times**1.17)
    # hackkey_times = hackkey_times - shift
    
    data = data.loc[:, piano_notes].values
    data[data < args.onoff_threshold] = 0 

    keys = np.argwhere(np.max(data, axis=0) > 4).flatten()
    notes = []
    midi_notes = []
    
    vel = []
    midi_vel = []
    escap = []
    
    for key in keys:
        signal = data[:, key]
        peaks_info, smoothed_signal = detect_clean_peaks(signal, height_threshold=args.height_threshold)

        for info in peaks_info:
            onset_time = hackkey_times[info['start_index']]
            onset_value = info['start_value']
            offset_time = hackkey_times[info['end_index']]
            peak_time = hackkey_times[info['peak_index']]
            peak_value = info['peak_value']
            escapement_time = peak_time - onset_time
            escap.append(escapement_time)
            
            if escapement_time < 0:
                print(onset_time, offset_time, key)
                exit()
            
            velocity = (peak_value - onset_value) / escapement_time
            vel.append(velocity)
            try:
                velocity = min(127, map_to_midi_velocity_linear_robust(velocity))
                midi_vel.append(velocity)
            except:
                print(onset_time, offset_time, key, peak_value, onset_value, escapement_time)
                exit()
            hackkey_values = signal[info['start_index']:info['end_index'] + 1]
            smoothed_hackkey_values = smoothed_signal[info['start_index']:info['end_index'] + 1]
            
            note = {
                "onset": onset_time,
                "offset": offset_time,
                "key": int(key),
                "velocity": velocity,
                "peak_time": peak_time,
                "escapement_time": escapement_time,
                "hackkey_values": hackkey_values.tolist(),
                "smoothed_hackkey_values": smoothed_hackkey_values.tolist()
            }
            
            notes.append(note)
            
            midi_notes.append(Note(
                velocity=velocity,
                pitch=key + 21,  # MIDI pitch starts at 21 for A0
                start=onset_time,
                end=offset_time
            ))

    print(np.mean(escap), np.std(escap), np.max(escap), np.min(escap))
    print(np.mean(vel), np.std(vel), np.max(vel), np.min(vel))
    print(np.mean(midi_vel), np.std(midi_vel), np.max(midi_vel), np.min(midi_vel))
    
    # Save notes to JSON file
    with open(args.output_file, "w") as f:
        json.dump(notes, f, indent=4)
        
    # Create MIDI file
    midi = PrettyMIDI()
    piano = Instrument(program=0, name='Acoustic Grand Piano')
    piano.notes.extend(midi_notes)
    midi.instruments.append(piano)
    midi.write(args.output_midi)
