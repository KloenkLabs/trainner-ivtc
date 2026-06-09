# trainner-ivtc Research Log

## Current MVP Findings

The first useful baseline is a global luma cadence classifier trained only on synthetic data. It is fast enough to train on an RTX 4090 at native SD-like resolution and produces JSONL metadata suitable for downstream VapourSynth experiments.

Validation is used during training after every epoch to compute metrics and choose `best.pt`. It is not used for gradient updates and there is no early stopping yet.

The data pipeline now supports online synthetic generation. This should simplify training and avoid generated sample-folder I/O, but training speed should be re-measured because CPU generation and PNG decode throughput may become the bottleneck.

## Source Data

The first larger source sequence is `datasets/voy_intro_progressive_png`.

Observed source properties:

- 2154 RGB PNG frames.
- Alphabetical filenames from `00000001.png` to `00002154.png`.
- Native resolution 760x480.
- Progressive source frames used to synthesize interlaced/telecined training samples.

The smaller plumbing dataset is `datasets/gt_test1`.

Observed source properties:

- Progressive PNG frames.
- Native resolution 640x400.
- Used for quick synthetic generation, training-loop, and inference smoke testing.

## Voyager Intro Baseline

Config: `configs/voy_intro_luma_v1.yaml`

Dataset: `datasets/synthetic_voy_intro_luma_v1`

Training output: `experiments/voy_intro_luma_v1`

Key settings:

- Train samples: 2048
- Validation samples: 256
- Window frames: 11
- Input resolution: native 760x480
- Batch size: 8
- Epochs: 12
- Model base channels: 24
- Channel multipliers: `[1, 2, 4, 4]`

Result:

- Total iterations: 3072
- Training time: 7:11
- Average speed: 7.136 it/s
- Best epoch: 12
- Validation accuracy: 0.8242
- Validation macro F1: 0.8087

## Hyperparameter Sweep

The current full sweep tested all combinations of:

- Batch sizes: 4, 8, 16
- Epochs: 8, 12, 16
- Window frames: 7, 11, 15

Results are stored at:

- `experiments/sweeps/voy_intro_luma_v1/results.json`
- `experiments/sweeps/voy_intro_luma_v1/results.csv`

Best run:

- Window frames: 7
- Batch size: 8
- Epochs: 16
- Best epoch: 16
- Validation accuracy: 0.87890625
- Validation macro F1: 0.8737535901012606
- Checkpoint: `experiments/sweeps/voy_intro_luma_v1/wf7_bs8_ep16/checkpoints/best.pt`

Top 5 runs by validation macro F1:

| Window frames | Batch size | Epochs | Val macro F1 | Val accuracy |
| --- | --- | --- | --- | --- |
| 7 | 8 | 16 | 0.8738 | 0.8789 |
| 7 | 16 | 16 | 0.8616 | 0.8516 |
| 7 | 8 | 12 | 0.8292 | 0.8359 |
| 7 | 16 | 12 | 0.8080 | 0.7969 |
| 7 | 4 | 16 | 0.7772 | 0.7656 |

Aggregate observations:

| Group | Average macro F1 | Best macro F1 |
| --- | --- | --- |
| `window_frames=7` | 0.7641 | 0.8738 |
| `window_frames=11` | 0.5775 | 0.7617 |
| `window_frames=15` | 0.5052 | 0.6455 |
| `batch_size=4` | 0.5769 | 0.7772 |
| `batch_size=8` | 0.6541 | 0.8738 |
| `batch_size=16` | 0.6158 | 0.8616 |
| `epochs=8` | 0.5343 | 0.7359 |
| `epochs=12` | 0.6169 | 0.8292 |
| `epochs=16` | 0.6956 | 0.8738 |

Current interpretation:

- The 7-frame window unexpectedly beats 11 and 15 frames on the current synthetic validation set.
- More epochs helped consistently within this small sweep.
- Batch size 8 produced the best single result, while batch size 16 was close.
- The result may reflect the current synthetic data distribution and simple global architecture, not a general rule for real DVD footage.

## Small Test Checkpoint

Config: `configs/global_luma_v1_test.yaml`

Dataset: `datasets/synthetic_global_luma_v1_test`

Training output: `experiments/global_luma_v1_test`

The quick test run used 32 train samples and 8 validation samples at native 640x400. It produced usable checkpoints for plumbing only:

- `best.pt` was from epoch 1.
- `last.pt` was from epoch 2.
- Validation accuracy was 0.125.
- Macro F1 was 0.0247.

This checkpoint can test inference and JSONL emission, but it is not useful for cadence quality.

## Inference Smoke Finding

The small test checkpoint successfully ran inference on `datasets/gt_test1` and emitted 27 JSONL prediction records. This confirms that checkpoint loading, frame-sequence reading, luma conversion, field splitting, batching, and JSONL serialization are wired together.

## Data Generation Caveats

Some generated samples look like synthetic noise or moving procedural patterns with no visible source-frame content. This is intentional for the current `video`, `scene_cut`, and `unknown` classes. Film phase samples and blend samples use source frames.

This is not dropout. Dropout is model-side regularization and only affects hidden activations during training.

Current data risks:

- Procedural negative classes may be too visually different from real DVD video, credits, and VFX shots.
- `scene_cut` and `unknown` should probably gain source-based variants.
- True-video examples should eventually come from real or better synthesized 29.97i/29.97p material.
- The validation set is synthetic and may overstate usefulness on real mixed-content DVD footage.
- Previous sweep results used materialized synthetic datasets; online-mode performance and best hyperparameters need a new benchmark.
