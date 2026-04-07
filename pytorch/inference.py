import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import pandas as pd
import argparse
import h5py
import math
import time
import librosa
import soundfile as sf
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
 
from utilities import (create_folder, get_filename, RegressionPostProcessor, HackkeyRegressionPostProcessor,
    OnsetsFramesPostProcessor, write_events_to_midi, load_audio, align_hackkey_wav_pedal_and_remove_silence, 
    piano_notes, align_hackkey_wav_and_remove_silence)
from models import Regress_onset_offset_frame_velocity_CRNN, Regress_onset_offset_frame_velocity_hackkey_CRNN
from pytorch_utils import forward
import config
from scipy.signal import resample


class PianoTranscription(object):
    def __init__(self, model_type, checkpoint_path=None, 
        segment_samples=16000*10, device=torch.device('cuda'), 
        post_processor_type='regression'):
        """Class for transcribing piano solo recording.

        Args:
          model_type: str
          checkpoint_path: str
          segment_samples: int
          device: 'cuda' | 'cpu'
        """

        if 'cuda' in str(device) and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.segment_samples = segment_samples
        self.post_processor_type = post_processor_type
        self.frames_per_second = config.frames_per_second
        self.classes_num = config.classes_num
        self.onset_threshold = 0.05
        self.offset_threshold = 0.3
        self.frame_threshold = 0.3
        self.pedal_offset_threshold = 0.3

        # Build model
        Model = eval(model_type)
        self.model = Model(frames_per_second=self.frames_per_second, 
            classes_num=self.classes_num)

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'], strict=False)

        # Parallel
        if 'cuda' in str(self.device):
            self.model.to(self.device)
            print('GPU number: {}'.format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            print('Using CPU.')

    def transcribe(self, audio):
        """Transcribe an audio recording.

        Args:
          audio: (audio_samples,)
          midi_path: str, path to write out the transcribed MIDI.

        Returns:
          transcribed_dict, dict: {'output_dict':, ..., 'est_note_events': ..., 
            'est_pedal_events': ...}
        """

        audio = audio[None, :]  # (1, audio_samples)

        # Pad audio to be evenly divided by segment_samples
        audio_len = audio.shape[1]
        pad_len = int(np.ceil(audio_len / self.segment_samples)) \
            * self.segment_samples - audio_len

        audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)

        # Enframe to segments
        segments = self.enframe(audio, self.segment_samples)
        """(N, segment_samples)"""

        # Forward
        output_dict = forward(self.model, segments, batch_size=1)
        """{'reg_onset_output': (N, segment_frames, classes_num), ...}"""

        # Deframe to original length
        for key in output_dict.keys():
            if key != 'hiddens':
                output_dict[key] = self.deframe(output_dict[key])[0 : audio_len]
        """output_dict: {
          'reg_onset_output': (segment_frames, classes_num), 
          'reg_offset_output': (segment_frames, classes_num), 
          'frame_output': (segment_frames, classes_num), 
          'hackkey_output': (segment_frames, classes_num), 
          'reg_pedal_onset_output': (segment_frames, 1), 
          'reg_pedal_offset_output': (segment_frames, 1), 
          'pedal_frame_output': (segment_frames, 1)}"""

        # Post processor
        if self.post_processor_type == 'regression':
            """Proposed high-resolution regression post processing algorithm."""
            post_processor = RegressionPostProcessor(self.frames_per_second, 
                classes_num=self.classes_num, onset_threshold=self.onset_threshold, 
                offset_threshold=self.offset_threshold, 
                frame_threshold=self.frame_threshold, 
                pedal_offset_threshold=self.pedal_offset_threshold)
        
        if self.post_processor_type == 'regression_hackkey':
            """Proposed high-resolution regression post processing algorithm."""
            post_processor = HackkeyRegressionPostProcessor(self.frames_per_second, 
                classes_num=self.classes_num, onset_threshold=self.onset_threshold, 
                offset_threshold=self.offset_threshold, 
                frame_threshold=self.frame_threshold, 
                pedal_offset_threshold=self.pedal_offset_threshold)

        elif self.post_processor_type == 'onsets_frames':
            """Google's onsets and frames post processing algorithm. Only used 
            for comparison."""
            post_processor = OnsetsFramesPostProcessor(self.frames_per_second, 
                self.classes_num)

        
        if self.post_processor_type == 'regression_hackkey':
            est_hackkey_events = \
                post_processor.output_dict_to_hackkey(output_dict)
        else:
            # Post process output_dict to MIDI events
            (est_note_events, est_pedal_events) = \
                post_processor.output_dict_to_midi_events(output_dict)

        # Write MIDI events to file
        # if midi_path:
        #     write_events_to_midi(start_time=0, note_events=est_note_events, 
        #         pedal_events=est_pedal_events, midi_path=midi_path)
        #     print('Write out to {}'.format(midi_path))

        # transcribed_dict = {
        #     'output_dict': output_dict, 
        #     'est_note_events': est_note_events,
        #     'est_pedal_events': est_pedal_events}
        
        transcribed_dict = {
            'output_dict': output_dict, 
            'est_hackkey_events': est_hackkey_events}

        return transcribed_dict

    def enframe(self, x, segment_samples):
        """Enframe long sequence to short segments.

        Args:
          x: (1, audio_samples)
          segment_samples: int

        Returns:
          batch: (N, segment_samples)
        """
        assert x.shape[1] % segment_samples == 0
        batch = []

        pointer = 0
        while pointer + segment_samples <= x.shape[1]:
            batch.append(x[:, pointer : pointer + segment_samples])
            pointer += segment_samples // 2

        batch = np.concatenate(batch, axis=0)
        return batch

    def deframe(self, x):
        """Deframe predicted segments to original sequence.

        Args:
          x: (N, segment_frames, classes_num)

        Returns:
          y: (audio_frames, classes_num)
        """
        if x.shape[0] == 1:
            return x[0]

        else:
            x = x[:, 0 : -1, :]
            """Remove an extra frame in the end of each segment caused by the
            'center=True' argument when calculating spectrogram."""
            (N, segment_samples, classes_num) = x.shape
            assert segment_samples % 4 == 0

            y = []
            y.append(x[0, 0 : int(segment_samples * 0.75)])
            for i in range(1, N - 1):
                y.append(x[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)])
            y.append(x[-1, int(segment_samples * 0.25) :])
            y = np.concatenate(y, axis=0)
            return y

def inference(args):
    """Inference template.

    Args:
      model_type: str
      checkpoint_path: str
      post_processor_type: 'regression' | 'onsets_frames'. High-resolution 
        system should use 'regression'. 'onsets_frames' is only used to compare
        with Googl's onsets and frames system.
      audio_path: str
      cuda: bool
    """

    # Arugments & parameters
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    post_processor_type = args.post_processor_type
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    # Transcriptor
    sample_rate = config.sample_rate
    segment_samples = sample_rate * 10  
    transcriptor = PianoTranscription(model_type, device=device, 
        checkpoint_path=checkpoint_path, segment_samples=segment_samples, 
        post_processor_type=post_processor_type)
    
    audio_paths = args.audio_paths
    
    for audio_path in tqdm(audio_paths):
        
        
        ext = audio_path.split('.')[-1]
        audio = librosa.load(audio_path, sr=sample_rate, mono=True)[0]
        duration = audio.shape[0] / sample_rate
        
        # Transcribe and write out to MIDI file
        transcribe_time = time.time()
        transcribed_dict = transcriptor.transcribe(audio)

        print('Transcribe time: {:.3f} s'.format(time.time() - transcribe_time))
        # print(transcribed_dict['output_dict']['hiddens'].shape)
        
        hackkey_events = transcribed_dict['est_hackkey_events']
        hackkey_roll = np.zeros((88, int(duration * 1000)))
        time_index = np.arange(0, duration, 0.001)[:int(duration * 1000)]
        for i in hackkey_events:
            start = max(0, int(i['onset_time'] * 1000))
            end = min(int(i['offset_time'] * 1000), int(duration * 1000))
            key = i['midi_note']
            vel = i['velocity']
            hackkey = i['hackkey']
            
            # Calculate target length and create interpolated indices
            target_len = end - start
            
            if target_len > 0:
                interpolated_hackkey = resample(hackkey, target_len)
                try:
                # Fill the roll with interpolated values
                    hackkey_roll[key, start:end] = interpolated_hackkey
                except Exception as e:
                    print(f"Error filling hackkey roll for key {key}, start {start}, end {end}: {e}")
                    continue
            
                
        df = pd.DataFrame(hackkey_roll.T, columns=piano_notes)
        df.index = time_index
        df.to_csv(audio_path.replace(f'.{ext}', '_keypress_motion.csv'), index=True)
        # np.save(audio_path.replace(f'.{ext}', '_hiddens.npy'), transcribed_dict['output_dict']['hiddens'])

def visualize(args):
    """Visualize the transcribed hackkey"""
    # Arugments & parameters
    audio_paths = args.audio_paths
    output_paths = [audio_path.replace(f'.{audio_path.split(".")[-1]}', '_keypress_motion.csv') for audio_path in audio_paths]
    
    for output_path in output_paths:
        basename = os.path.basename(output_path).split('.')[0].split("_")[0]
        hackkey_output = pd.read_csv(output_path, header=0)    
        start = args.start * 1000
        end = args.end * 1000
        
        hackkey_roll = hackkey_output.values[start:end, 1:].T
        
        keys = np.argwhere(np.max(hackkey_roll, axis=1) > 4).flatten()
        fig, axe = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot full hackkey roll
        axe.imshow(hackkey_roll, origin='lower', aspect='auto', cmap='gray')
        axe.set_title('Hackkey Roll Prediction')
        axe.set_xlabel('Time (ms)')
        plt.savefig(f"results/{basename}_roll_prediction.png", dpi=144, bbox_inches='tight')
        
        # Plot key-wise predictions
        fig2, axs = plt.subplots(len(keys), 1, figsize=(20, len(keys) * 5))
        if len(keys) == 1:
            axs = [axs]
        
        for idx, key in enumerate(keys):
            axs[idx].plot(hackkey_roll[key, :], label='Hackkey_PRED', linewidth=2)
            axs[idx].legend(fontsize=10)
            axs[idx].set_title(f"Key {key}", fontsize=10)
            axs[idx].set_xlabel('Time (ms)')
            axs[idx].set_ylabel('Key Press Depth (mm)')

        plt.tight_layout()
        plt.savefig(f"results/{basename}_note_wise_prediction.png", dpi=144, bbox_inches='tight')
        plt.close(fig)
        plt.close(fig2)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--post_processor_type', type=str, default='regression', choices=['onsets_frames', 'regression', 'regression_hackkey'])
    parser.add_argument('--audio_paths', type=str, nargs='+', required=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--visualize', '-v', action='store_true', default=False, help='Visualize the transcribed hackkey')
    parser.add_argument('--output_path', type=str, default='results/hackkey_roll.csv', help='Path to save the transcribed hackkey roll')
    parser.add_argument('--start', type=int, default=0, help='Start time in seconds for visualization')
    parser.add_argument('--end', type=int, default=10, help='End time in seconds for visualization')

    args = parser.parse_args()

    # Run inference
    print('Running inference...')
    print('Model type: {}'.format(args.model_type))
    print('Checkpoint path: {}'.format(args.checkpoint_path))
    print('Post processor type: {}'.format(args.post_processor_type))
    print('Audio paths: {}'.format(args.audio_paths))
    print('Using device: {}'.format('cuda' if args.cuda and torch.cuda.is_available() else 'cpu'))
    
    # Inference
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {args.checkpoint_path} does not exist.")
    inference(args)
    
    if args.visualize:
        visualize(args)
