# Few-Shot Synthetic Speech Detection

[![Paper](https://img.shields.io/badge/IEEE-ASRU%202025-00629B.svg)](https://ieeexplore.ieee.org/document/11434594)
[![arXiv](https://img.shields.io/badge/arXiv-2508.13320-b31b1b.svg)](https://arxiv.org/abs/2508.13320)

Official implementation of *["Rapidly Adapting to New Voice Spoofing: Few-Shot
Detection of Synthesized Speech Under Distribution
Shifts"](https://ieeexplore.ieee.org/document/11434594)*.

## Abstract

> We address the challenge of detecting synthesized speech under distribution
> shifts—arising from unseen synthesis methods, speakers, languages, or audio
> conditions—relative to the training data. Few-shot learning methods are a
> promising way to tackle distribution shifts by rapidly adapting on the basis
> of a few in-distribution samples. We propose a self-attentive prototypical
> network to enable more robust few-shot adaptation. To evaluate our approach,
> we systematically compare the performance of traditional zero-shot detectors
> and the proposed few-shot detectors, carefully controlling training
> conditions to introduce distribution shifts at evaluation time. In conditions
> where distribution shifts hamper the zero-shot performance, our proposed
> few-shot adaptation technique can quickly adapt using as few as 10
> in-distribution samples—achieving up to 32% relative EER reduction on
> deepfakes in the Japanese language and 20% relative reduction on the ASVspoof
> 2021 Deepfake dataset.

## Model Variants

| `--model` | Training labels | Prototype aggregation |
| --- | --- | --- |
| `binary_mean` | bonafide / spoof | arithmetic mean |
| `binary_attention` | bonafide / spoof | self-attention |
| `multiclass_attention` | bonafide / attack type | self-attention |



## Quick Start

### 1. Create the environment

Python 3.10 is recommended. Create and activate an environment from the
repository root, then install the pinned dependencies:

```bash
conda create -n fewshot-spoof python=3.10 -y
conda activate fewshot-spoof
pip install -r requirements.txt
```

If Fairseq is installed from a local checkout instead of `requirements.txt`,
make it importable before running the code:

```bash
export FAIRSEQ_ROOT=/path/to/fairseq
export PYTHONPATH="$FAIRSEQ_ROOT:${PYTHONPATH}"
```

### 2. Download XLS-R 300M

The SSL-AASIST front end requires the Fairseq **XLS-R 300M** checkpoint.

```bash
mkdir -p checkpoints/xlsr
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt \
  -O checkpoints/xlsr/xlsr2_300m.pt
```

The checkpoint is also linked from the official
[Fairseq XLS-R release](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr).
Pass its path to every training or evaluation command with `--ssl-checkpoint`.

### 3. Download the released artifacts

Install the Hugging Face Hub client and download the checkpoints, configs, and
paper manifests into the repository:

```bash
pip install huggingface_hub
hf download ash56/few_shot_spoof_detection \
  --local-dir checkpoints/released
```

The release does not include XLS-R 300M. Keep using the separately downloaded
checkpoint at `checkpoints/xlsr/xlsr2_300m.pt` when running SSL-AASIST models.

### 4. Prepare the data lists

The paper trains on ASVspoof 2019 LA and evaluates under multiple distribution
shifts, including ASVspoof 2021, ShiftySpeech, In-the-Wild, and CodecFake.
Download the paper manifests from the
[`manifests/` directory on Hugging Face](https://huggingface.co/ash56/few_shot_spoof_detection/tree/main/manifests),
then obtain the audio from the corresponding dataset provider:

| Dataset | Download |
| --- | --- |
| ASVspoof 2019 LA | [University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336) |
| ASVspoof 2021 LA and DF | [ASVspoof 2021](https://www.asvspoof.org/index2021.html) |
| ShiftySpeech | [Hugging Face](https://huggingface.co/datasets/ash56/ShiftySpeech) |
| In-the-Wild | [Deepfake Total](https://deepfake-total.com/in_the_wild) |
| CodecFake | [CodecFake](https://github.com/roger-tseng/CodecFake) |
| VoxCeleb1 | [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) |

Each plain-text manifest contains one utterance per line. Binary training and
all evaluation manifests use:

```text
ASVspoof2019_LA_train/flac/LA_T_1000137.flac bonafide
ASVspoof2019_LA_train/flac/LA_T_1004644.flac spoof
```

Multi-class training manifests preserve the ASVspoof 2019 LA attack labels:

```text
ASVspoof2019_LA_train/flac/LA_T_1000137.flac bonafide
ASVspoof2019_LA_train/flac/LA_T_1004644.flac A01
ASVspoof2019_LA_train/flac/LA_T_1004645.flac A02
```

Supported spoof labels are `A01` through `A06`. Relative paths are resolved
against the manifest directory by default. Use `--audio-root /path/to/datasets`
when paths are relative to a common dataset root. Audio is loaded as mono,
resampled to 16 kHz when needed, and repeated or truncated to 64,600 samples by
the dataset loader.

The released training and validation lists are disjoint. The loaders do not
create dataset splits: they sample support and query examples from the supplied
lists at runtime. Audio remains subject to each source dataset's license.

### 5. Train a model

The paper's few-shot protocol uses 5 support and 15 query examples per class,
100 episodes per epoch, Adam with a learning rate of `1e-3`, and StepLR decay by
`0.5` every 20 epochs. These values can be overridden through the CLI.

Binary mean-prototype baseline:

```bash
python src/train.py --model binary_mean --cuda \
  --ssl-checkpoint checkpoints/xlsr/xlsr2_300m.pt \
  --train-list checkpoints/released/manifests/train/asvspoof2019_la_binary.txt \
  --valid-list checkpoints/released/manifests/validation/asvspoof2019_la_binary.txt \
  --audio-root /path/to/ASVspoof2019/LA \
  --experiment-root runs/binary_mean \
  --iterations 100
```

Binary self-attentive ProtoNet:

```bash
python src/train.py --model binary_attention --cuda \
  --ssl-checkpoint checkpoints/xlsr/xlsr2_300m.pt \
  --train-list checkpoints/released/manifests/train/asvspoof2019_la_binary.txt \
  --valid-list checkpoints/released/manifests/validation/asvspoof2019_la_binary.txt \
  --audio-root /path/to/ASVspoof2019/LA \
  --experiment-root runs/binary_attention
```

Multi-class self-attentive ProtoNet:

```bash
python src/train.py --model multiclass_attention --cuda \
  --ssl-checkpoint checkpoints/xlsr/xlsr2_300m.pt \
  --train-list checkpoints/released/manifests/train/asvspoof2019_la_multiclass.txt \
  --valid-list checkpoints/released/manifests/validation/asvspoof2019_la_multiclass.txt \
  --audio-root /path/to/ASVspoof2019/LA \
  --experiment-root runs/multiclass_attention
```

Each run writes `best_model.pth`, `last_model.pth`, per-epoch checkpoints,
training curves, and loss/accuracy logs to `--experiment-root`.

### 6. Evaluate a checkpoint

Evaluation repeatedly samples bonafide and spoof support sets, builds one
prototype per class, and reports the mean EER. The paper reports the mean over
100 support-set draws. The checkpoints used below are available from the
[Hugging Face release](https://huggingface.co/ash56/few_shot_spoof_detection/tree/main/checkpoints).

Binary mean-prototype checkpoint:

```bash
python src/evaluate.py --model binary_mean \
  --ssl-checkpoint checkpoints/xlsr/xlsr2_300m.pt \
  --checkpoint checkpoints/released/checkpoints/binary_mean_5shot/best_model.pth \
  --eval-list checkpoints/released/manifests/evaluation/shiftyspeech/ja/apnet2.txt \
  --audio-root /path/to/ShiftySpeech \
  --output-dir runs/eval_binary_mean \
  --support-sizes 5 10 100 \
  --num-iters 100 \
  --seed 7 \
  --device cuda:0
```

Binary attention checkpoint:

```bash
python src/evaluate.py --model binary_attention \
  --ssl-checkpoint checkpoints/xlsr/xlsr2_300m.pt \
  --checkpoint checkpoints/released/checkpoints/binary_attn_5shot/best_model.pth \
  --eval-list checkpoints/released/manifests/evaluation/shiftyspeech/ja/apnet2.txt \
  --audio-root /path/to/ShiftySpeech \
  --output-dir runs/eval_binary \
  --support-sizes 5 10 100 \
  --num-iters 100 \
  --seed 7 \
  --device cuda:0
```

Multi-class-trained attention checkpoint:

```bash
python src/evaluate.py --model multiclass_attention \
  --ssl-checkpoint checkpoints/xlsr/xlsr2_300m.pt \
  --checkpoint checkpoints/released/checkpoints/multiclass_attn_5shot/best_model.pth \
  --eval-list checkpoints/released/manifests/evaluation/shiftyspeech/ja/apnet2.txt \
  --audio-root /path/to/ShiftySpeech \
  --output-dir runs/eval_multiclass \
  --support-sizes 5 10 100 \
  --num-iters 100 \
  --seed 7 \
  --device cuda:0
```

For each support size, the evaluator writes all iteration-level EERs and the
mean EER to `<support>_support_<tag>-eer.txt`.

## Zero-Shot Baselines

The binary zero-shot baseline uses the original two-class SSL-AASIST model. Its
checkpoint is included in the Hugging Face release and can be evaluated with:

```bash
python src/zero_shot.py evaluate --model binary \
  --ssl-checkpoint checkpoints/xlsr/xlsr2_300m.pt \
  --checkpoint checkpoints/released/checkpoints/zero_shot_ssl_aasist_binary/model.pth \
  --eval-list checkpoints/released/manifests/evaluation/asvspoof/asvspoof2021_df.txt \
  --audio-root /path/to/datasets \
  --scores runs/zero_shot_binary/scores.txt \
  --summary runs/zero_shot_binary/eer.txt \
  --device cuda:0
```

The AASIST zero-shot baseline is maintained by its original authors and is not
redistributed with this release. Clone the official repository; its pretrained
ASVspoof 2019 LA checkpoint is included at
`models/weights/AASIST-L.pth`:

```bash
git clone https://github.com/clovaai/aasist.git
cd aasist
pip install -r requirements.txt
```

Set `database_path` in `config/AASIST-L.conf` to the extracted ASVspoof 2019
LA directory, then run the upstream evaluator:

```bash
python main.py --eval --config ./config/AASIST-L.conf
```

The paper reports this external baseline as `AASIST`. Refer to the
[official AASIST repository](https://github.com/clovaai/aasist) for its model
implementation, data preparation, and license.

The seven-class zero-shot model was trained on ASVspoof 2019 LA with one
bonafide class and six attack classes (`A01` through `A06`). The default
training configuration follows the paper: RawBoost algorithm 5, Adam with a
learning rate of `1e-5` and weight decay of `1e-4`, batch size 64, and 100
epochs.

```bash
python src/zero_shot.py train-multiclass \
  --ssl-checkpoint checkpoints/xlsr/xlsr2_300m.pt \
  --train-list checkpoints/released/manifests/train/asvspoof2019_la_multiclass.txt \
  --valid-list checkpoints/released/manifests/validation/asvspoof2019_la_multiclass.txt \
  --audio-root /path/to/ASVspoof2019/LA \
  --output-dir runs/zero_shot_multiclass \
  --device cuda:0
```

Training saves a checkpoint after each epoch. Select a checkpoint using
validation performance; the published model corresponds to epoch 33. At
evaluation time, the spoof score is the mean of the six spoof-class logits:

```bash
python src/zero_shot.py evaluate --model multiclass \
  --ssl-checkpoint checkpoints/xlsr/xlsr2_300m.pt \
  --checkpoint checkpoints/released/checkpoints/zero_shot_ssl_aasist_multiclass/model.pth \
  --eval-list checkpoints/released/manifests/evaluation/asvspoof/asvspoof2021_df.txt \
  --audio-root /path/to/datasets \
  --scores runs/zero_shot_multiclass/scores.txt \
  --summary runs/zero_shot_multiclass/eer.txt \
  --device cuda:0
```

## Mahalanobis Baseline

The paper's one-class baseline uses the pretrained SpeechBrain x-vector model
to extract bonafide embeddings, fits one Gaussian with Ledoit-Wolf covariance
estimation, and uses Mahalanobis distance as the spoof score. 

Create a manifest containing the 2,580 bonafide utterances from the ASVspoof
2019 LA training partition, then fit the Gaussian statistics:

```bash
python src/mahalanobis.py fit \
  --train-list checkpoints/released/manifests/train/asvspoof2019_la_bonafide.txt \
  --audio-root /path/to/ASVspoof2019/LA \
  --output-stats runs/mahalanobis/asvspoof2019_bona_xvec.pkl \
  --device cuda:0
```

Score an evaluation set and compute EER:

```bash
python src/mahalanobis.py evaluate \
  --eval-list checkpoints/released/manifests/evaluation/asvspoof/asvspoof2021_df.txt \
  --audio-root /path/to/datasets \
  --stats checkpoints/released/checkpoints/one_class_mahalanobis/gaussian_stats.pkl \
  --scores runs/mahalanobis/asv2021_df_scores.txt \
  --summary runs/mahalanobis/asv2021_df_eer.txt \
  --device cuda:0
```

The first run downloads `speechbrain/spkrec-xvect-voxceleb` into
`checkpoints/spkrec-xvect-voxceleb` unless `--encoder-cache` is overridden.

## Data Manifests

Train, validation, and evaluation manifests use paths relative to their
corresponding dataset root. Provide the local dataset location through
`--audio-root`. This keeps manifests portable without redistributing audio,
which remains subject to each source dataset's license.

The release contains binary and multi-class ASVspoof 2019 LA train/validation
lists and evaluation lists for ASVspoof 2019 LA, ASVspoof 2021 LA/DF,
In-the-Wild, CodecFake C1-C7, and the six ShiftySpeech vocoder domains. Extract
ShiftySpeech archives while preserving the repository layout so paths such as
`Vocoders/apnet2/jsut_flac/BASIC5000_0958.flac` resolve directly beneath the
ShiftySpeech root.

## Released Checkpoints

Paper checkpoints and their matching configuration files are distributed
separately through
[Hugging Face](https://huggingface.co/ash56/few_shot_spoof_detection).

## Credits

This implementation builds upon the following repositories:

- [Prototypical Networks for Few-shot Learning in PyTorch](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch)
- [SSL Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)
- [AASIST](https://github.com/clovaai/aasist)

## License

This project is released under the [MIT License](LICENSE).

## Citation

Please cite this work if you find it useful:

```bibtex
@inproceedings{garg2025rapidly,
  title={Rapidly Adapting to New Voice Spoofing: Few-Shot Detection of Synthesized Speech Under Distribution Shifts},
  author={Garg, Ashi and Cai, Zexin and Xinyuan, Henry Li and Garc{\'\i}a-Perera, Leibny Paola and Khudanpur, Kevin Duh Sanjeev and Wiesner, Matthew and Andrews, Nicholas},
  booktitle={2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  pages={1--7},
  year={2025},
  organization={IEEE}
}
```
