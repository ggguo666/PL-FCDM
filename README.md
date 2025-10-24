# PL-FCDM
hyperparameters
| dataset    | Batch Size | lr1   | weightdecay1 | lr2   | Weightdecay2 |
|-----------|------------|-----------|-----------|-----------|-----------|
| ABIDEI    | 16         | 0.00007   | 0.00006   | 0.00005   | 0.0003    |
| ABIDE II  | 64         | 0.0004    | 0.0006    | 0.0001    | 0.00015   |
| ADHD-200  | 32         | 0.0006    | 0.0003    | 0.006     | 0.0003    |
| MDD       | 128        | 0.00007   | 0.0006    | 0.0008    | 0.0003    |

The project structure is organized as follows:


## 📂 Directory Description

- **imports/** — Contains dataset construction files.  
- **net/** — Includes neural network model implementations.  
  - Files starting with **“01”** — Pre-trained model implementation.  
  - Files starting with **“02”** — Pseudo-label prediction and functional connectivity (FC) reconstruction.  
  - Files starting with **“03”** — OE-DSCNN model training and classification.

