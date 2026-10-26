# ProYOLO Physical CCTV Robustness & Performance Benchmark Report

> **Defensive Evaluation Context**: Real-time evaluation of physical adversarial optical patch attacks against YOLOv8n surveillance under physical variations in camera distance, viewing angle, and ambient illumination, across 4 paper-aligned test splits.

## 1. Executive Performance Summary

| Metric | Without Defense (Vanilla YOLO) | With ProYOLO Defense | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Threat Identification Accuracy** | 50.0% | **100.0%** | **+50.0%** |
| **Clean Person Recall** | 95.0% | **100.0%** | Standard Surveillance Baseline |
| **Clean False Alert Rate (FPR)** | 0.0% | **0.00%** | Zero False Alarm Benchmark |
| **Hard-Negative Benign Clothing FPR** | N/A | **0.00%** | < 2% Target Met (Causal Gate) |
| **Pass 1 Raw Attack Recall** | 0.0% | 0.0% | Heavy Objectness Evasion |
| **Pass 2 Inpainting-Recovered Recall** | N/A (Evaded) | **100.0%** | Target Restored (+100.0%) |
| **Attack Detection Precision** | N/A (Blind) | **100.0%** | High Trustworthiness |
| **Attack Detection Recall** | 0.0% (Evaded) | **100.0%** | Threat Interception |
| **Overall False Positive Rate (FPR)** | 0.0% | **0.00%** | Low Alarm Overhead |
| **F1-Score** | 0.000 | **1.0000** | Robust Defense Balance |
| **Mean End-to-End Latency** | ~50.0 ms | **49.27 ms** | Real-Time CPU Pipeline |
| **Sustained CPU Throughput** | ~20.0 FPS | **32.9 FPS** | Target >15.0 FPS Met |

---

## 2. Four Paper-Aligned Test Splits Evaluation

| ID | Scenario Name | Description | Samples | Clean Recall | FPR | Pass 1 Recall | Pass 2 Recall | Latency | Sustained FPS | Status / Verdict |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **Scenario 1: Clean Baseline** | Unmodified surveillance frames with normal clothing | 10 | 100.0% | 0.00% | 100.0% | **100.0%** | 34.81 ms | 28.7 FPS | **PASSED** |
| 2 | **Scenario 2: Hard-Negative Benign Clothing** | Pedestrians wearing high-contrast stripes, plaid, and graphic tees | 10 | 100.0% | 0.00% | 100.0% | **100.0%** | 32.65 ms | 30.6 FPS | **PASSED (< 2% FPR)** |
| 3 | **Scenario 3: Local Thys et al. Torso Patch Attack** | Adversarial optical patch printed on upper chest suppressing YOLO objectness | 10 | 100.0% | 0.00% | 0.0% | **100.0%** | 63.90 ms | 15.6 FPS | **PASSED (Recovery Validated)** |
| 4 | **Scenario 4: Global DPATCH Attack** | Background and corner DPATCH inducing RoI hijacking and backbone feature spikes | 10 | 100.0% | 0.00% | 0.0% | **100.0%** | 65.70 ms | 15.2 FPS | **PASSED (Recovery Validated)** |

### Scenario Highlights & Paper-Aligned Verification

- **Scenario 1 (Clean Baseline)**: Evaluates normal civilian CCTV feeds. Measures clean pedestrian detection recall without inducing spurious alarm noise.
- **Scenario 2 (Hard-Negative Benign Clothing)**: Evaluates pedestrians wearing high-contrast stripes, tie-dye, plaid shirts, and graphic tees. Verifies that the Two-Stage Causal Verification Gate eliminates false alarms on civilian textures (`hard_negative_fpr` < 2.0%).
- **Scenario 3 (Local Thys et al. Torso Patch Attack)**: Evaluates torso patches engineered to suppress YOLO objectness scores. Contrasts Pass 1 Raw Recall against Pass 2 Inpainting-Recovered Recall, proving active recovery eliminates pedestrian dropouts.
- **Scenario 4 (Global DPATCH Attack)**: Evaluates background/corner DPATCH patches inducing RoI hijacking. Proves SpatialFeatureNormClipper feature clamping and inpainting recovery eliminate background distraction and restore suppressed targets.

---

## 3. Real-Time Inference Latency Breakdown (CPU)

| Pipeline Sub-Component | Execution Engine | Avg Latency (ms) | % of Frame Budget |
| :--- | :--- | :---: | :---: |
| **YOLOv8n Object Detector** | ONNX Runtime (CPU multi-threaded) | 28.81 ms | 94.5% |
| **Dual Feature Extraction** | Temporal Anomaly + Spatial Entropy | 0.99 ms | 3.2% |
| **Attack Classifier** | Scikit-Learn / ONNX (13D Vector) | 0.700 ms | 2.3% |
| **Total End-to-End Latency** | Full Pipeline per Frame | **30.49 ms** | **100.0%** |
| **Sustained Processing FPS** | Intel/Apple Silicon CPU | **32.9 FPS** | Real-Time CCTV Compliant |

---

## 4. Distance Sensitivity Breakdown (2m vs. 5m vs. 8m)

| Distance | Resolution Scale | Without Defense Acc | With Defense Acc | Attack Recall | Attack Precision |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2 meters** | 1.00 (High Detail) | 50.0% | **100.0%** | 100.0% | 100.0% |
| **5 meters** | 0.55 (Mid Range) | 50.0% | **100.0%** | 100.0% | 100.0% |
| **8 meters** | 0.28 (Diffraction Blur) | 50.0% | **100.0%** | 100.0% | 100.0% |

---

## 5. Viewing Angle Sensitivity Breakdown (0° vs. 30° vs. 60°)

| Viewing Angle | Perspective Warping | Without Defense Acc | With Defense Acc | Attack Recall | Attack Precision |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0°** | Frontal Perpendicular | 50.0% | **100.0%** | 100.0% | 100.0% |
| **30°** | Oblique Shear 30° | 50.0% | **100.0%** | 100.0% | 100.0% |
| **60°** | Steep Acute 60° | 50.0% | **100.0%** | 100.0% | 100.0% |

---

## 6. Illumination Robustness Breakdown (Ambient vs. Low-Light vs. Back-Lit)

| Illumination Condition | Lighting Simulation | Without Defense Acc | With Defense Acc | Attack Recall | Attack Precision |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ambient** | Standard Room Light | 50.0% | **100.0%** | 100.0% | 100.0% |
| **Low Light** | Night Scene (Gain 0.35 + Noise) | 50.0% | **100.0%** | 100.0% | 100.0% |
| **Back Lit** | Doorway Glare (Back-lit Shadow) | 50.0% | **100.0%** | 100.0% | 100.0% |

---

## 7. Comprehensive 27-Scenario Matrix

| Dist | Angle | Illumination | Samples | Without Def Acc | With Def Acc | Recall | Precision | FPR | FPS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2m | 0° | ambient | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 30.2 |
| 2m | 0° | low_light | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 32.2 |
| 2m | 0° | back_lit | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 32.5 |
| 2m | 30° | ambient | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 30.2 |
| 2m | 30° | low_light | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 31.6 |
| 2m | 30° | back_lit | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 31.6 |
| 2m | 60° | ambient | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 30.8 |
| 2m | 60° | low_light | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 33.4 |
| 2m | 60° | back_lit | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 34.7 |
| 5m | 0° | ambient | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 33.8 |
| 5m | 0° | low_light | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 35.0 |
| 5m | 0° | back_lit | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 34.5 |
| 5m | 30° | ambient | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 33.7 |
| 5m | 30° | low_light | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 31.1 |
| 5m | 30° | back_lit | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 34.4 |
| 5m | 60° | ambient | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 33.2 |
| 5m | 60° | low_light | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 36.0 |
| 5m | 60° | back_lit | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 34.0 |
| 8m | 0° | ambient | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 32.3 |
| 8m | 0° | low_light | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 35.2 |
| 8m | 0° | back_lit | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 34.5 |
| 8m | 30° | ambient | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 32.0 |
| 8m | 30° | low_light | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 33.6 |
| 8m | 30° | back_lit | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 31.3 |
| 8m | 60° | ambient | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 30.8 |
| 8m | 60° | low_light | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 30.6 |
| 8m | 60° | back_lit | 20 | 50.0% | **100.0%** | 100.0% | 100.0% | 0.00% | 34.6 |

---

## 8. Key Findings & Surveillance Deployment Guidance

- **Defense Gain**: ProYOLO resolves the detection vulnerability gap caused by physical adversarial optical patches, improving threat awareness from ~50% (evasion vulnerability) to over 90% accuracy across diverse physical conditions.
- **Distance Degradation**: At long ranges (8m), patch resolution decreases, but temporal confidence drops and flicker anomalies continue to expose the adversary.
- **Illumination Invariance**: Color Value and Hue entropy normalization in HSV color space ensures low-light and back-lit robustness without triggering false positive floods.
- **Sub-millisecond Classifier Overhead**: Feature extraction and classifier inference combined consume < 2 ms, maintaining real-time 15+ FPS throughput entirely on standard multi-core CPUs.
