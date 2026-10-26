# Deep Learning-Based Lung Sound Event Detection
### An evaluation of class-imbalance-aware loss functions — MSc Artificial Intelligence dissertation, Queen Mary University of London (2026)

A frame-level **CNN-BiGRU** detector for breath phases and adventitious lung sounds on the public **HF_Lung_V1** dataset (9,765 recordings), replicating and extending [Hsu et al. (2021)](https://arxiv.org/abs/2102.03049). The study asks a practical question: **does reweighting the loss actually help when abnormal sounds are rare, or does the gain disappear once you control for training noise?**

## Tasks
| Task | What is detected |
|---|---|
| Inhalation / Exhalation | Breath phases |
| CAS | Continuous adventitious sounds (wheeze, stridor, rhonchi) |
| DAS | Discontinuous adventitious sounds (crackles) |

## Pipeline
```
.wav (15 s, 4 kHz)
  → 80 Hz high-pass Butterworth filter (zero-phase)
  → STFT (n_fft=256, hop=64) → 938 frames
  → per frame: 129 log-spectrogram bins + 60 MFCC features (20 + Δ + ΔΔ) + 4 sub-band energies = 193 channels
  → Conv1d(193→256) + ReLU + MaxPool + Dropout
  → BiGRU (256 per direction)
  → Linear → one logit per frame
```

## Loss configurations compared (CAS and DAS)
1. **Standard BCE**
2. **Raw class-weighted BCE** (`pos_weight` = negative/positive frame ratio)
3. **Capped class-weighted BCE** (`pos_weight` capped below the raw ratio)
4. **Focal loss** (α from training-set class balance, γ = 2; neither tuned)

## Evaluation protocol
- **Leakage-safe split:** recordings from the same date are never split across train and validation.
- **Test set touched once:** early stopping, checkpoint selection, cap selection and decision thresholds all use the validation split; the predefined test partition is read only for final evaluation.
- **Three levels of scoring:** frame, segment and event level (events matched by temporal Jaccard Index ≥ 0.5, following Hsu et al.).
- **Multi-seed robustness check:** the split is fixed and only initialisation/batch order vary, so differences reflect training noise, not a different validation set. An outlier seed was detected as an incomplete checkpoint and retrained (Section 8b).
- **Duration-stratified error analysis:** recall on short, medium and long events.

## Results (held-out test set, frame-level)
| Task | Model | F1 | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|
| Inhalation | Baseline | **76.0%** | 78.5% | 92.3% | 0.941 |
| Exhalation | Baseline | 44.8% | 50.3% | 92.5% | 0.849 |
| CAS | Standard BCE | 50.2% | 64.0% | 79.1% | 0.802 |
| CAS | Raw weighted | 51.2% | 66.5% | 78.5% | 0.808 |
| CAS | Capped (2.5) | 47.5% | 57.1% | 80.8% | 0.799 |
| CAS | Focal | 47.9% | 72.7% | 69.8% | 0.773 |
| DAS | Standard BCE | **70.0%** | 84.6% | 76.0% | 0.875 |
| DAS | Raw weighted | 69.5% | 82.9% | 76.6% | 0.872 |
| DAS | Capped (1.5) | 69.5% | 87.4% | 73.0% | 0.873 |
| DAS | Focal | 68.5% | 84.7% | 73.7% | 0.869 |

### Multi-seed check (test F1, mean ± sd across seeds)
| Task | Standard BCE | Raw weighted | Capped |
|---|---|---|---|
| CAS | 47.2 ± 1.3 | 50.6 ± 1.0 | — |
| DAS | 70.2 ± 0.7 (2 seeds) | 66.4 ± 1.9 | 69.8 ± 0.6 |

**In these runs:** raw class weighting gave a small, consistent gain for CAS, while for DAS no reweighting strategy beat standard BCE. Focal loss did not improve either task. See Section 9 of the notebook for which event durations each loss helps or hurts.

## How to run
The notebook was developed in Google Colab on an NVIDIA A100 GPU.
1. Download HF_Lung_V1 (see Hsu et al., 2021) and place it in Google Drive as `MyDrive/HF_Lung_V1-master/` with `train/train/` and `test/test/` folders.
2. Run Sections 1–2 (mount Drive, definitions). Features are cached to `processed/`.
3. Training functions reload existing checkpoints unless `force_retrain=True`, so results can be reproduced without retraining once checkpoints exist.
4. Sections 7, 8 and 9 produce the results tables above.

The dataset and model checkpoints are not included in this repository.

## Tech stack
Python · PyTorch · librosa · SciPy · scikit-learn · pandas

## Reference
Hsu, F.-S. et al. (2021). *Benchmarking of eight recurrent neural network variants for breath phase and adventitious sound detection on a self-developed open-access lung sound database — HF_Lung_V1.* PLOS ONE. [arXiv:2102.03049](https://arxiv.org/abs/2102.03049)

---
**Author:** Sude Kumbasar · MSc dissertation supervised at Queen Mary University of London
