# trainner-ivtc Project Notes

## Project Goal

`trainner-ivtc` is a synthetic-first neural cadence classifier for downstream IVTC workflows. The first MVP is a global luma classifier: it looks at a short temporal window of interlaced frames, split into luma fields, and predicts cadence metadata with confidence scores.

The model does not reconstruct progressive frames and does not perform IVTC itself. Inference writes JSONL metadata intended for later VapourSynth processing.

The v1 scope is intentionally narrow:

- Input: progressive PNG source frames for online synthetic training or materialized synthetic generation, plus pre-extracted frame sequences for inference.
- Internal model input: luma fields shaped as `[B, window_frames * 2, H / 2, W]`.
- Output: one global cadence class per target frame/window.
- Deferred: patch-grid cadence maps, local field matching, RGB/chroma features, ONNX export, and mixed-cadence region reconstruction.

## Current Architecture

The implementation lives in a new root package, `trainner_ivtc`. `legacy-trainner` remains unchanged and is treated as reference code only.

The classifier is a lightweight fully convolutional 2D CNN over stacked luma fields:

- Luma frame window is loaded from PNGs.
- Each interlaced frame is split into top and bottom fields.
- Fields are stacked along the channel axis.
- Residual/downsampling blocks extract temporal-spatial features from the stacked field tensor.
- Global average pooling collapses the spatial dimensions.
- A linear classifier head emits 8 logits.

For the current Voyager intro config, an 11-frame 760x480 sample becomes a model tensor of `[22, 240, 760]`. The tested model defaults are `base_channels: 24`, `channel_mult: [1, 2, 4, 4]`, and `dropout: 0.1`.

## Classes

The model predicts these 8 classes:

| ID | Class name |
| --- | --- |
| `0` | `film_phase_0` |
| `1` | `film_phase_1` |
| `2` | `film_phase_2` |
| `3` | `film_phase_3` |
| `4` | `film_phase_4` |
| `V` | `video` |
| `B` | `blend` |
| `U` | `unknown` |

Inference emits compact JSONL records with `idx`, `class_id`, `class_name`, `conf`, `film_conf`, `video_conf`, and `probs`. Output class names use `pd_0..pd_4` for pulldown phases. Float values are rounded to 6 decimals.

## Synthetic Data

Synthetic data is generated from alphabetically sorted progressive PNG frame sequences. No naming scheme is enforced; file order is based on filename sorting.

Training now defaults to online synthetic generation, where the PyTorch dataset creates each sample in memory and returns the field tensor directly. The materialized `make_synthetic` path remains available for debugging and writes inspectable PNG frame folders plus JSONL manifests. Each manifest row points to a sample directory and records its class label, field order, frame list, and target frame index.

Dataset split and sizing are source-frame based:

- `train_samples_pct` and `val_samples_pct` split each alphabetically sorted source sequence into train and validation ranges.
- Defaults are `90` and `10`.
- Source-based train and validation samples do not draw from the same source-frame range.
- `dataset_repeats` multiplies each online split length so small source sets can have longer epochs.
- `source_cache_mode` controls source frame caching: `shared_ram` preloads luma frames once for DataLoader workers, `lru` keeps the previous bounded per-worker cache, and `none` disables caching.
- If both `height` and `width` are `0`, the config resolves native dimensions from the source images.
- A bounded per-worker LRU source-frame cache is controlled by `source_cache_size`.
- Online training can random-crop each generated sample with `crop_height`, `crop_width`, and `crop_modulo`.
- Cropping is applied once per temporal window before field splitting, so every frame in the sample uses the same crop.
- Crop bounds are deterministic from the sample seed and epoch, and train samples use new crops each epoch when `resample_train_each_epoch` is enabled.
- Training-only augmentations are configured under `data.augmentations`, with independent per-window `chance` values.

The generator currently applies these data variations:

- Random telecine phase for `film_phase_0..4`.
- Configurable field order, currently `tff` or `bff`.
- 3:2 pulldown field pairing using the current film phase.
- Source-frame blend samples by blending two telecined windows with a random alpha.
- Progressive/PsF `video` samples that keep source frames unchanged.
- Procedural unknown samples with static or low-motion content.
- Optional Gaussian noise with a configurable `std_range`.
- Optional under-exposure by multiplying all frames in a window by a sampled darkening factor.
- Optional online-only random crop with modulo-aligned crop bounds.

Important caveat: procedural `unknown` samples may look like noise or moving synthetic patterns and may contain no recognizable source image. That is data generation behavior, not model dropout. Dropout is only applied inside the model during training.

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

Training logs compact progress to the terminal. JSON metrics and detailed messages are written to `train.log` in the experiment output directory. The trainer validates after each epoch, writes `last.pt`, and updates `best.pt` when validation macro F1 improves. Macro F1 averages only classes with `class_distribution` weight at least `0.001`. Online training resamples train samples each epoch while keeping validation fixed.

### `trainner_ivtc.infer`

Runs inference on a pre-extracted frame sequence and writes JSONL predictions.

Example:

```powershell
python -m trainner_ivtc.infer --checkpoint experiments/sweeps/voy_intro_luma_v1/wf7_bs8_ep16/checkpoints/best.pt --input datasets/gt_test1 --output predictions.jsonl
```

Useful options include `--field-order`, `--batch-size`, and `--device`.

Example record:

```json
{"idx":123,"class_id":"2","class_name":"pd_2","conf":0.97,"film_conf":0.98,"video_conf":0.01,"probs":{"pd_0":0.01,"pd_1":0.0,"pd_2":0.97,"pd_3":0.0,"pd_4":0.0,"video":0.01,"blend":0.0,"unknown":0.01}}
```

### `trainner_ivtc.sweep`

Runs a configured hyperparameter sweep over batch size, epoch count, and temporal window length.

Example:

```powershell
python -m trainner_ivtc.sweep --base-config configs/voy_intro_luma_v1.yaml --sweep-dir experiments/sweeps/voy_intro_luma_v1
```

The sweep creates per-window datasets, per-combination configs, per-run experiment folders, and summary files at `results.json` and `results.csv`.
