<div align="center">

# Soft Augmentation and Adversarial Robustness

**Does teaching a model to say "I'm not sure" also make it harder to fool?**

A preliminary empirical study on CIFAR-10 with ResNet-18, FGSM and PGD.

[Open the notebook in Colab](https://colab.research.google.com/drive/1ByMjPK5--Klpmt5EvQvppoqQtwJ1Q0Gw)

Status: preliminary results (1 training seed) | Dataset: CIFAR-10 | Model: ResNet-18

</div>

---

## Overview

Soft Augmentation ([Liu et al., CVPR 2023](https://arxiv.org/abs/2211.04625)) lowers the confidence of a training label in proportion to how much of an image has been cropped away. The original paper reports gains in accuracy, calibration and occlusion robustness, but it does not evaluate adversarial attacks.

This project asks one question:

> Does training with Soft Augmentation make a model harder to fool with adversarial attacks than training with standard hard-label augmentation?

## Key Result

In all **8 attack settings** tested (FGSM and PGD at epsilon = 1, 2, 4, 8 out of 255), the Soft Augmentation model retained more accuracy than the baseline, by **3.4 to 19.0 percentage points**. On clean images the difference was only 0.58 points.

Neither model is robust in absolute terms. The accurate claim is "more resistant than the baseline," not "robust." This is a single-seed result; see [Limitations](#limitations).

## Results

Test accuracy (%) on the CIFAR-10 test set. Higher is better.

| Setting | Baseline | Soft Augmentation | Gap (points) |
|---|---:|---:|---:|
| Clean (no attack) | 94.19 | 94.77 | +0.58 |
| FGSM 1/255 | 61.19 | 64.56 | +3.37 |
| PGD 1/255 | 40.71 | 44.17 | +3.46 |
| FGSM 2/255 | 45.29 | 57.24 | +11.95 |
| PGD 2/255 | 11.19 | 27.76 | +16.57 |
| FGSM 4/255 | 34.94 | 52.61 | +17.67 |
| PGD 4/255 | 1.69 | 15.90 | +14.21 |
| FGSM 8/255 | 28.73 | 47.69 | +18.96 |
| PGD 8/255 | 0.05 | 7.44 | +7.39 |

**What the results show**

- The Soft Augmentation model is ahead in every attack setting, under both the weaker (FGSM) and the stronger (PGD) attack.
- Sanity checks hold: accuracy falls as epsilon grows, and PGD is always lower than FGSM at the same epsilon.
- The advantage under attack (up to about 19 points) is much larger than the clean-accuracy difference (0.58 points).
- Absolute robustness is low: under PGD, the soft model keeps 15.90% at 4/255 and 7.44% at 8/255.

## Method

### Soft Augmentation

Let `v` be the fraction of the original image still visible, `C` the number of classes, and `k` the crop exponent. The target probability for the true class is:

```
p = 1 - (1 - 1/C) * (1 - v)^k
```

The remaining probability `1 - p` is shared equally among the other classes, and the loss is scaled by `p`, so heavily damaged images teach less.

Example: with `C = 10`, `k = 2`, `v = 0.3`, the true class gets `p = 0.559,` and each wrong class gets `0.049`.

### Attacks

| Attack | Description |
|---|---|
| FGSM | One gradient-sign step of size epsilon |
| PGD | 10 steps, step size epsilon/4, projected back to the epsilon-ball |

Epsilon values are 1, 2, 4, and 8, each divided by 255, with pixel values in the range [0, 1].

## Experimental Setup

| Item | Setting |
|---|---|
| Dataset | CIFAR-10 (50,000 train, 10,000 test) |
| Model | ResNet-18, using the authors' code |
| Training | 50 epochs, SGD, seed 0, Colab T4 GPU |
| Baseline | Standard crop and flip, cross-entropy loss |
| Soft Augmentation | Soft crop, soft-target loss |
| Metric | Test accuracy (%) on attacked images |

Both models used identical settings, seed and training length. The only differences are the augmentation and loss flags.

## Reproducing the Results

**1. Get the training code.** Training uses the authors' repository: [youngleox/soft_augmentation](https://github.com/youngleox/soft_augmentation).

**2. Keep the final checkpoint.** The repository saves a checkpoint only when the epoch is a multiple of `SAVE_EPOCH`. Change it from 100 to 25 in `conf/global_settings.py` so the epoch-50 model is saved.

**3. Train both models.**

```bash
# Baseline
python train.py --task cifar10 --net resnet18 --da 0 --loss ce --sch 0.25 --ft 5 --n 2 --seed 0

# Soft Augmentation
python train.py --task cifar10 --net resnet18 --da 2 --loss st --pcrop 2 --crop 12 --sch 0.25 --ft 5 --n 2 --seed 0
```

**4. Run the attacks.** Open the notebook and run it on both checkpoints:

[Open the attack notebook in Colab](https://colab.research.google.com/drive/1ByMjPK5--Klpmt5EvQvppoqQtwJ1Q0Gw).

## Limitations

- **Single seed.** Each model was trained once. The original paper averages 3 runs. A previously retrained soft model reached 94.65% clean accuracy versus 94.77% here, so small clean-accuracy gaps are within run-to-run variation.
- **Shorter training.** 50 epochs, not the 500 used in the original paper.
- **Attack strength.** PGD with 10 steps is not the strongest attack. A stronger or adaptive attack could reduce the advantage, and it has not yet been ruled out that part of the gain comes from harder-to-attack gradients rather than genuine robustness.
- **No explanation tested.** Why soft labels help here is a hypothesis (for example, less overconfident outputs), not a result.
- **Scope.** One dataset and one architecture (CIFAR-10, ResNet-18).

## Repository Contents

```
.
├── README.md
├── soft_augmentation_attacks.ipynb      # attack evaluation notebook
└── Experiment_Report.pdf                # full experiment report
```

## Read More

- Write-up: [Substack post](https://frankenstein7.substack.com)
- Full report: [`Experiment_Report.pdf`](Experiment_Report.pdf)

## Credits and References

The training code and the Soft Augmentation method belong to the original authors. The adversarial evaluation in this repository is my own addition.

1. Liu, Y., Yan, S., Leal-Taixe, L., Hays, J., Ramanan, D. *Soft Augmentation for Image Classification.* CVPR 2023. [arXiv:2211.04625](https://arxiv.org/abs/2211.04625). Code: [youngleox/soft_augmentation](https://github.com/youngleox/soft_augmentation).
2. Goodfellow, I. J., Shlens, J., Szegedy, C. *Explaining and Harnessing Adversarial Examples.* ICLR 2015. (FGSM)
3. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., Vladu, A. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018. (PGD)

## Author

**Malaika** — self-taught AI student documenting research in machine learning and security.
[GitHub](https://github.com/malaika-nadeem) · [Substack](https://frankenstein7.substack.com)
