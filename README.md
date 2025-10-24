# PL-FCDM

## 📦 Dependencies and Versions

The following Python packages and versions were used in this project:

| Package               | Version      |
|-----------------------|-------------|
| Python                | 3.8+        |
| PyTorch               | 1.12+       |
| torch.nn.functional   | Included in PyTorch |
| torchvision           | 0.13+       |
| tensorboardX          | 2.5+        |
| scikit-learn          | 1.0+        |
| pandas                | 1.4+        |
| matplotlib            | 3.5+        |
| numpy                 | 1.22+       |
| nni                   | 2.5+        |

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


## 🧩 Dynamic Functional Connectivity (dFC) Computation

The following code demonstrates how dynamic functional connectivity (dFC) matrices are computed using a sliding window approach with **nilearn**:

```python
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, input_data, plotting
import os

# Define atlas file
atlas_dir = 'dos160_roi_atlas.nii'
masker = input_data.NiftiLabelsMasker(
    labels_img=atlas_dir,
    standardize=True,
    memory='nilearn_cache',
    n_jobs=-1
)

# Folder containing functional images
folder_path = ''

# Define sliding window parameters
window_size = 50
step_size = 1

start_index = 获取文件夹名字.file_names_without_extension.index('')

for file_name in 获取文件夹名字.file_names_without_extension[start_index:]:
    print(file_name)
    func_filename = ''  # Specify functional image file path
    if os.path.exists(func_filename):
        # Extract time series for each ROI
        time_series = masker.fit_transform(func_filename)
        time_series1 = time_series[:, :-1]

        # Initialize connectivity measure
        connectivity_measure = ConnectivityMeasure(kind="correlation")

        # Initialize sliding window indices
        start_idx = 0
        end_idx = window_size

        # Create output folder
        folder_path1 = os.path.join('', file_name)
        os.makedirs(folder_path1, exist_ok=True)

        # Compute dynamic functional connectivity
        while end_idx <= len(time_series1):
            window_time_series = time_series1[start_idx:end_idx, :]
            connectivity_matrix = connectivity_measure.fit_transform([window_time_series])[0]

            # Save connectivity matrix
            np.save(folder_path1 + '\\' + str(start_idx) + '.npy', connectivity_matrix)
            print(start_idx)

            # Update sliding window
            start_idx += step_size
            end_idx += step_size
    else:
        print("File not found, skipping:", file_name)
        continue
