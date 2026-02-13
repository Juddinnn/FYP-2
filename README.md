📌 Overview

This project implements a deep learning framework for detecting Fast Flux DNS attacks using:
LSTM, GRU, BiLSTM (RNN-based models)
Spiking Neural Network (SNN)
Feature-level Ablation Studies

DNS behavioral features are extracted from raw dig logs and used to classify:
0 → Benign
1 → Fast Flux

📂 Project Structure
```
fast-flux-detection/
│
├── fast_flux_detection_rnn_snn.py
├── ablation_test_no_features.py
├── ablation_test_feature_groups_only.py
├── dataset/
│   ├── benign/
│   └── ff/
└── README.md
```
🧠 Main Model: RNN & SNN Pipeline
File: fast_flux_detection_rnn_snn.py
Includes:
-DNS feature extraction
-Shannon entropy calculation
-Data normalization (StandardScaler)
-Sequence generation (SEQ_LEN = 5)
-LSTM, GRU, BiLSTM
-Spiking Neural Network (LIF neurons via snntorch)
-Early stopping
-Classification report & confusion matrix

🧪 Ablation Test 1 – Feature Removal
File: ablation_test_no_features.py

Experiments:
Baseline (All Features)
No TTL Features
No IP Diversity Features
No DNS Structure Features
Purpose:
To measure performance impact when specific feature groups are removed.

🧪 Ablation Test 2 – Single Feature Groups
File: ablation_test_feature_groups_only.py
Experiments:
TTL Only
IP Diversity Only
DNS Structure Only
Purpose:
To evaluate how well each feature group performs independently.

🔍 Feature Groups
-TTL Features
-ttl_min, ttl_max, ttl_avg, ttl_stddev
-IP Diversity Features
-num_A_records, ip_entropy, num_unique_subnets
-DNS Structure Features
-num_CNAME_records, num_NS_records

📊 Evaluation
-Accuracy
-Precision
-Recall
-F1-score
-Confusion Matrix

Dataset split:
-70% Training
-15% Validation
-15% Testing

🚀 Technologies
Python
PyTorch
snntorch
scikit-learn
pandas / numpy
matplotlib / seaborn
