# PL-FCDM
## 📊 Optimized Hyperparameters by NNI

| Dataset  | Batch Size | lr1      | wd1      | lr2      | wd2      | Stepsize | Gamma | Dropout Rate (d) |
|-----------|-------------|----------|----------|----------|----------|-----------|--------|------------------|
| **ABIDE I**   | 16  | 0.00007 | 0.00006 | 0.00005 | 0.00030 | 60  | 0.50 | 0.30 |
| **ABIDE II**  | 64  | 0.00040 | 0.00060 | 0.00010 | 0.00015 | 30  | 0.75 | 0.10 |
| **ADHD-200**  | 32  | 0.00060 | 0.00030 | 0.00600 | 0.00030 | 30  | 0.50 | 0.10 |
| **MDD**       | 128 | 0.00007 | 0.00060 | 0.00080 | 0.00030 | 90  | 0.40 | 0.10 |


The project structure is organized as follows:


## 📂 Directory Description

- **imports/** — Contains dataset construction files.  
- **net/** — Includes neural network model implementations.  
- Files starting with **“01”** — Pre-trained model implementation.  
- Files starting with **“02”** — Pseudo-label prediction and functional connectivity (FC) reconstruction.  
- Files starting with **“03”** — OE-DSCNN model training and classification.

