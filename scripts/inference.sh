# conda activate vpt

python3 pytorch/inference.py  -v \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --checkpoint_path="checkpoints/60000_iterations.pth" \
    --post_processor_type="regression_hackkey" \
    --start=0 \
    --end=10 \
    --audio_paths results/sample1.wav results/sample2.wav results/sample3.wav \
    # --cuda # Uncomment this line if you want to use GPU for inference
