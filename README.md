# Visualising Pianists' Touch (VPT): Transcribing Expressive Piano Performance from Audio to Piano Key Motion
[![Paper](https://img.shields.io/badge/Paper-CHI%202026-green)](https://pseudo.link/paper)
[![Pre-trained Models](https://img.shields.io/badge/Models-Zenodo-9cf?logo=zenodo)](https://pseudo.link/models)
[![Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://pseudo.link/code)
[![Project Page](https://img.shields.io/badge/Demo-Project%20Page-blueviolet)](https://github.com/tangjjbetsy/VPT)
[![Colab Demo](https://img.shields.io/badge/Colab-Demo%20Notebook-orange?logo=googlecolab)](https://colab.research.google.com/drive/1tYuwfdin8RwlGGs7UAH1RkaeAFaHtak2?usp=share_link)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://pseudo.link/license)

This repository contains the official implementation of our CHI 2026 paper:

**"Visualising Pianists’ Touch: Transcribing Expressive Piano Performance from Audio to Piano Key Motion"**

by Jingjing Tang, Shinichi Furuya, Hayato Nishioka, Momoko Shioki, Geraint Wiggins, György Fazekas, and Vincent K.M. Cheung.

## Overview
Understanding expressive piano performance requires not only *what* notes are played, but also *how* they are physically produced. MIDI provides a discrete event-based representation (pitch, onset, duration, velocity), but it cannot capture continuous touch gestures such as fine-grained key press depth trajectories.

VPT introduces an audio-to-key-motion transcription technique that predicts **continuous piano key motion trajectories directly from performance audio**, making motor-grounded performance information accessible without requiring specialised sensors.

## Details
This repository was developed based on the [Piano Transcription](https://github.com/bytedance/piano_transcription) model by ByteDance. Therefore, we followed the similar structure and codebase of the original repository, and we have made modifications to the model architecture, training procedure, and evaluation metrics to accommodate the task of transcribing continuous piano key motion.

### Training
Due to commercial sensitivity of the training data, we are currently unable to release the dataset. However, we provided the full codebased for training the model with the key motion signals. You can refer to the `pytorch/main.py` file for details on how to train the model. We also provided a sample training script in `scripts/train.sh` that you can modify according to your needs. The checkpoint of our best performance model is released on our [Zenodo page](https://pseudo.link/models) for inference and further research. 

### Inference
We provide a sample inference script in `scripts/inference.sh` that demonstrates how to use the trained model to transcribe piano key motion from audio. You can modify the script to specify your own audio files. The inference script will generate CSV files containing the predicted key press depth trajectories, as well as visualizations of the predictions. To run the inference script, simply execute the following command in your terminal:

```bash
bash scripts/inference.sh
```
The script could process one or multiple audio files, and the output CSV files and visualizations will be saved in the `results` directory.

### Demo
We have also created a demo video showcasing the capabilities of our VPT model. You can watch the demo video on our [project page](https://pseudo.link/project-page). We recommend the Sonic Visualiser software for visualising the transcribed key motion alongside the audio performance. You can download Sonic Visualiser from [here](https://www.sonicvisualiser.org/).

## Acknowledgement
This work was supported by the UKRI Centre for Doctoral Training in Artificial Intelligence and Music [grant number EP/S022694/1], JST CREST [grant number JPMJCR20D4], JST CRONOS [grant number JPMJCS24N8], and Sony Computer Science Laboratories, Inc.. J. Tang is a research student jointly funded by the China Scholarship Council [grant number 202008440382] and Queen Mary University of London. G. Wiggins received funding from the Flemish Government under the "Onderzoeksprogramma Artificiële Intelligentie (AI) Vlaanderen". We thank the reviewers for their valuable feedback, which helped improve the quality of this work.

## Contact
Jingjing Tang: `jingjing.tang@qmul.ac.uk`

## Copyright and License
This project is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
