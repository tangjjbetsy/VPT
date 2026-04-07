#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 12
#$ -l h_rt=1:0:0
#$ -l gpu=1
#$ -l cluster=andrena
#$ -l h_vmem=7.5G
#$ -m bae
#$ -l rocky
#$ -N train_hackkey

source ~/.bashrc
mamba activate unet

DATASET_DIR="/data/scratch/acw555/Hackkey_Sony"

# Modify to your workspace
WORKSPACE="/data/scratch/acw555/Hackkey_Sony"

# # Pack audio files to hdf5 format for training
# python3 utils/features.py pack_hackkey_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# --- 1. Train note transcription system ---
# python3 pytorch/main.py train --workspace=$WORKSPACE \
#     --model_type='Regress_onset_offset_frame_velocity_hackkey_CRNN' \
#     --loss_type='regress_onset_offset_frame_velocity_hackkey' \
#     --augmentation='none' \
#     --max_note_shift=0 \
#     --batch_size=8 \
#     --learning_rate=5e-4 \
#     --reduce_iteration=10000 \
#     --resume_iteration=0 \
#     --early_stop=200000 \
#     --datadir=$DATASET_DIR \
#     --cuda
    
# --- 2. Inference note transcription system ---
# python3 pytorch/inference.py \
#     --model_type='Regress_onset_offset_frame_velocity_CRNN' \
#     --checkpoint_path="/data/scratch/acw555/Hackkey_Sony/checkpoints/main/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_mae/augmentation=none/max_note_shift=0/batch_size=16/time=2025-05-29_16-51-21/12000_iterations.pth" \
#     --post_processor_type="regression_hackkey" \
#     --audio_path="SkillCheck/20240407-arpeggio-sakuraba/20240407-16-36-30/raw/sound/core_output_sound.wav" \
#     --meta_path="/data/scratch/acw555/Hackkey_Sony/hackkey_meta_concert_clean_upload_realign.csv" \
#     --output_path="results/hackkey_roll_skillcheck.csv" \
#     --data_dir=$DATASET_DIR \
#     --start=25 \
#     --end=30 \
#     --cuda


# --- 3. Calculate metrics for Hackkey ---
CHECKPOINT_PATH="/data/scratch/acw555/Hackkey_Sony/checkpoints/main/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_mae/augmentation=none/max_note_shift=0/batch_size=16/time=2025-05-30_07-45-39/60000_iterations.pth"

python3 pytorch/calculate_score_for_paper.py infer_prob \
    --workspace=$WORKSPACE \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --checkpoint_path=$CHECKPOINT_PATH \
    --augmentation='none' \
    --dataset='hackkey' \
    --split='test' \
    --cuda

python3 pytorch/calculate_score_for_paper.py calculate_metrics \
    --workspace=$WORKSPACE \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --augmentation='none' \
    --dataset='hackkey' \
    --split='test'

