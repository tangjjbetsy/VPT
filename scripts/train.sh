conda activate vpt

# Modify to your dataset directory
DATASET_DIR="YOUR_DATASET_DIR"

# Modify to your workspace
WORKSPACE="YOUR_WORKSPACE_DIR"

# # Pack audio files to hdf5 format for training
python3 utils/features.py pack_hackkey_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Note that we use the bytedance released checkpoints to initialize the model
# "Replacement" Mode, the velocity head was replaced by hackkey values
# --- 1. Train note transcription system ---
python3 pytorch/main.py train --workspace=$WORKSPACE \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --loss_type='regress_onset_offset_frame_hackkey_mae' \
    --max_note_shift=0 \
    --batch_size=8 \
    --learning_rate=5e-4 \
    --reduce_iteration=10000 \
    --resume_iteration=0 \
    --early_stop=200000 \
    --datadir=$DATASET_DIR \
    --checkpoint_path="checkpoints/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth" \
    --cuda

# "Additional" Mode, a new head was added to predict hackkey values, some 
# modifications were expected when using this mode, please refer to the comments in the codebase for details.

# python3 pytorch/main.py train --workspace=$WORKSPACE \
#     --model_type='Regress_onset_offset_frame_velocity_CRNN' \
#     --loss_type='regress_onset_offset_frame_hackkey_mae' \
#     --max_note_shift=0 \
#     --batch_size=8 \
#     --learning_rate=5e-4 \
#     --reduce_iteration=10000 \
#     --resume_iteration=0 \
#     --early_stop=200000 \
#     --datadir=$DATASET_DIR \
#     --checkpoint_path="checkpoints/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth" \
#     --cuda
    
