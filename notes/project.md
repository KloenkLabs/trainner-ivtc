# trainner-ivtc Project Notes

## Project Goal

`trainner-ivtc` is a synthetic-first neural cadence classifier for downstream IVTC workflows. The first MVP is a global luma classifier: it looks at a short temporal window of interlaced frames, split into luma fields, and predicts cadence metadata with confidence scores.

The model does not reconstruct progressive frames and does not perform IVTC itself. Inference writes JSONL metadata intended for later VapourSynth processing.

The v1 scope is intentionally narrow:

- Input: progressive PNG source frames for online synthetic training or materialized synthetic generation, plus pre-extracted frame sequences for inference.
- Internal model input: luma fields shaped as `[B, window_frames * 2, H / 2, W]`.
- Output: one global cadence class per target frame/window.
- Deferred: patch-grid cadence maps, local field matching, RGB/chroma features, ONNX export, VapourSynth integration, and mixed-cadence region reconstruction.

## Current Architecture

The implementation lives in a new root package, `trainner_ivtc`. `legacy-trainner` remains unchanged and is treated as reference code only.

The classifier is a lightweight fully convolutional 2D CNN over stacked luma fields:

- Luma frame window is loaded from PNGs.
- Each interlaced frame is split into top and bottom fields.
- Fields are stacked along the channel axis.
- Residual/downsampling blocks extract temporal-spatial features from the stacked field tensor.
- Global average pooling collapses the spatial dimensions.
- A linear classifier head emits 9 logits.

For the current Voyager intro config, an 11-frame 760x480 sample becomes a model tensor of `[22, 240, 760]`. The tested model defaults are `base_channels: 24`, `channel_mult: [1, 2, 4, 4]`, and `dropout: 0.1`.

## Classes

The model predicts these 9 classes:

| ID | Class name |
| --- | --- |
| `0` | `film_phase_0` |
| `1` | `film_phase_1` |
| `2` | `film_phase_2` |
| `3` | `film_phase_3` |
| `4` | `film_phase_4` |
| `V` | `video` |
| `B` | `blend` |
| `C` | `scene_cut` |
| `U` | `unknown` |

Inference derives `film_confidence`, `video_confidence`, class probabilities, and a `recommended_action` for each JSONL record.

## Synthetic Data

Synthetic data is generated from alphabetically sorted progressive PNG frame sequences. No naming scheme is enforced; file order is based on filename sorting.

Training now defaults to online synthetic generation, where the PyTorch dataset creates each sample in memory and returns the field tensor directly. The materialized `make_synthetic` path remains available for debugging and writes inspectable PNG frame folders plus JSONL manifests. Each manifest row points to a sample directory and records its class label, field order, frame list, and target frame index.

Dataset split and sizing are source-frame based:

- `train_samples_pct` and `val_samples_pct` split each alphabetically sorted source sequence into train and validation ranges.
- Defaults are `90` and `10`.
- Source-based train and validation samples do not draw from the same source-frame range.
- If both `height` and `width` are `0`, the config resolves native dimensions from the source images.
- A bounded per-worker LRU source-frame cache is controlled by `source_cache_size`.

The generator currently applies these data variations:

- Random telecine phase for `film_phase_0..4`.
- Configurable field order, currently `tff` or `bff`.
- 3:2 pulldown field pairing using the current film phase.
- Source-frame blend samples by blending two telecined windows with a random alpha.
- Procedural true-video samples with moving texture and simple moving patterns.
- Procedural scene-cut samples using two separate generated clips with a random cut position.
- Procedural unknown samples with static or low-motion content.
- Optional Gaussian noise through `noise_std`.

Important caveat: procedural `video`, `scene_cut`, and `unknown` samples may look like noise or moving synthetic patterns and may contain no recognizable source image. That is data generation behavior, not model dropout. Dropout is only applied inside the model during training.

## Scripts

### `trainner_ivtc.data.make_synthetic`

Creates synthetic train/validation datasets from progressive source PNG frames.

Example:

```powershell
python -m trainner_ivtc.data.make_synthetic --config configs/voy_intro_luma_v1.yaml --overwrite
```

Optional worker override:

```powershell
python -m trainner_ivtc.data.make_synthetic --config configs/voy_intro_luma_v1.yaml --workers 8
```

By default, `data.num_workers: auto` uses the system CPU core count. Generation is multithreaded and keeps manifest order deterministic. `make_synthetic` uses the same percentage split and native dimension resolution as online training.

### `trainner_ivtc.train`

Trains the global classifier. With `data.dataset_mode: online`, training reads source PNGs and generates synthetic samples on the fly. With `data.dataset_mode: manifest`, training reads a generated dataset manifest.

Example:

```powershell
python -m trainner_ivtc.train --config configs/voy_intro_luma_v1.yaml
```

Training logs compact progress to the terminal. JSON metrics and detailed messages are written to `train.log` in the experiment output directory. The trainer validates after each epoch, writes `last.pt`, and updates `best.pt` when validation macro F1 improves. Online training resamples train samples each epoch while keeping validation fixed.

### `trainner_ivtc.infer`

Runs inference on a pre-extracted frame sequence and writes JSONL predictions.

Example:

```powershell
python -m trainner_ivtc.infer --checkpoint experiments/sweeps/voy_intro_luma_v1/wf7_bs8_ep16/checkpoints/best.pt --input datasets/gt_test1 --output predictions.jsonl
```

Useful options include `--field-order`, `--batch-size`, and `--device`.

### `trainner_ivtc.sweep`

Runs a configured hyperparameter sweep over batch size, epoch count, and temporal window length.

Example:

```powershell
python -m trainner_ivtc.sweep --base-config configs/voy_intro_luma_v1.yaml --sweep-dir experiments/sweeps/voy_intro_luma_v1
```

The sweep creates per-window datasets, per-combination configs, per-run experiment folders, and summary files at `results.json` and `results.csv`.
