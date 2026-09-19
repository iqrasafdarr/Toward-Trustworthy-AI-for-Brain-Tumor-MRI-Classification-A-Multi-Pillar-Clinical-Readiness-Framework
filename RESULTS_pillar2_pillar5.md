\# Pillar 2 \& Pillar 5 Results — Seed 42



\*\*Date:\*\* 2026-09-20

\*\*Data:\*\* predictions\_{mobilenetv2,resnet50,efficientnetb0}\_seed42.npz (1,600 test images, 4 classes)

\*\*Script:\*\* quick\_start\_pillar2\_pillar5.py (temperature scaling + APS conformal triage + CRI v2)

\*\*Branch:\*\* feature/clinical-readiness-v2



\## Pillar 5 — Clinical Readiness Index (CRI v2)



\### Pareto frontier (default IX weights)

All three architectures are \*\*non-dominated\*\*: MobileNetV2, ResNet50, EfficientNetB0.

=\&gt; No single architecture dominates on all CRI axes at default weights; final

&#x20;  ranking requires seed-averaged metrics (see Caveats).



\### Decision curve analysis (tumor vs no-tumor)

| Model | Beats treat-all/none over thresholds | Mean net-benefit advantage (10–50% band) |

|---|---|---|

| MobileNetV2 | 4% – 99% | 0.1026 |

| ResNet50 | 2% – 99% | 0.1055 |

| EfficientNetB0 | 2% – 99% | 0.1107 |



EfficientNetB0 is the strongest on decision-curve net benefit; all three models

are clinically net-beneficial across essentially the entire threshold range.



\## Pillar 2 — Conformal triage (APS)



Temperature scaling fitted on a 10% calibration carve-out:

MobileNetV2 T=1.65, ResNet50 T=2.25, EfficientNetB0 T=1.40.



| Model | α | Target cov. | q\_hat | Actual cov. | Mean set size | Referral rate |

|---|---|---|---|---|---|---|

| MobileNetV2 | 0.10 | 90% | 1.0000 | 99.9% | 3.38/4 | 88.9% |

| MobileNetV2 | 0.15 | 85% | 1.0000 | 99.7% | 3.10/4 | 82.3% |

| MobileNetV2 | 0.20 | 80% | 1.0000 | 99.6% | 2.93/4 | 78.1% |

| ResNet50 | 0.10 | 90% | 1.0000 | 99.9% | 3.64/4 | 93.9% |

| ResNet50 | 0.15 | 85% | 1.0000 | 99.9% | 3.37/4 | 89.0% |

| ResNet50 | 0.20 | 80% | 0.9998 | 99.8% | 3.11/4 | 83.3% |

| EfficientNetB0 | 0.10 | 90% | 1.0000 | 100.0% | 3.71/4 | 93.5% |

| EfficientNetB0 | 0.15 | 85% | 1.0000 | 99.9% | 3.54/4 | 89.6% |

| EfficientNetB0 | 0.20 | 80% | 0.9998 | 99.9% | 3.26/4 | 85.3% |



\### Interpretation (important — read before citing)

\- Coverage is achieved (≈99–100%) but \*\*q\_hat ≈ 1.0 even after temperature

&#x20; scaling\*\*, meaning APS needs nearly the full class ranking to guarantee

&#x20; coverage: mean prediction sets contain \~3–3.7 of 4 classes.

\- Referral rates of 78–94% mean the conformal triage layer flags almost every

&#x20; scan as uncertain — \*\*not clinically useful as-is\*\*. This is an honest finding:

&#x20; seed-42 models are too miscalibrated/uncertain for selective prediction.

\- Possible causes: saturated logits, small test set (1,600), calibration

&#x20; carve-out from the same test set (double-dipping, see Caveats), or genuinely

&#x20; hard classes (meningioma vs pituitary overlap in this dataset is known).



\## Caveats

1\. Calibration set = 10% carve-out of the SAME seed-42 test set

&#x20;  (double-dipping). Final paper numbers must use a separate validation npz

&#x20;  (will be produced by the Kaggle retraining run, Track B).

2\. Single seed (42). Paper claims need seed-averaged results (5 seeds).

3\. Label order assumed \[glioma, meningioma, notumor, pituitary]; confirm

&#x20;  against training script before publication.



\## Next steps

\- \[ ] Track B: Kaggle retraining (baseline + Pillar 1 APL) → new npz with logits + .keras checkpoints

\- \[ ] Re-run Pillar 2/5 with proper validation calibration split

\- \[ ] Pillar 1 APL fine-tune + Pillars 3/4 via run\_all\_pillars.py

