HoVer-Net Nuclei Segmentation Toolkit
A PyTorch implementation of HoVer-Net for nuclei segmentation and classification in histopathology images. This toolkit supports both tile-based inference (for small images) and WSI (Whole Slide Image) processing (for large tissue sections).
Overview
This project provides a ready-to-use pipeline for:
Nuclei instance segmentation (identifying individual nuclei)
Optional nuclei type classification (if using a type-enabled model)
Support for both small image tiles (e.g., 256×256px) and large WSI formats (e.g., .svs, .tif)
Visualization of input-output comparisons for result validation
Based on the original HoVer-Net paper with adaptations for PyTorch compatibility.
Features
⚡ Fast inference with configurable batch size and parallel processing
🎨 Automatic result visualization (input vs. segmentation overlay)
🧩 Support for multiple model checkpoints (original/fast mode, with/without type prediction)
📁 QuPath-compatible output (for downstream 病理分析)
🖥️ GPU acceleration (configurable for multi-GPU setups)
Installation
Prerequisites
Python 3.8+
PyTorch 1.7+ (with CUDA support recommended)
OpenCV, NumPy, Matplotlib, OpenSlide (for WSI processing)

Clone the Repository
bash
git clone https://github.com/your-username/hovernet-nuclei-segmentation.git
cd hovernet-nuclei-segmentation
Environment Setup
This project recommends using Anaconda/Miniconda to manage Python environments for consistent dependency versions. Follow these steps for setup:

Step 1: Install Anaconda/Miniconda
If you haven’t installed a package manager, download and install from official sources:
Anaconda (full-featured, includes pre-installed scientific computing packages): https://www.anaconda.com/products/individual
Miniconda (lightweight, with only core dependencies): https://docs.conda.io/en/latest/miniconda.html

Step 2: Create a Virtual Environment
Open a terminal and run this command to create a virtual environment named hover_net_v2 (customize the name if needed):
bash
conda create -n hover_net_v2 python=3.9
Note: The project supports Python 3.8+. Python 3.9 is used as an example here.

Step 3: Activate the Virtual Environment
After creation, activate the environment:
Linux/macOS:
bash
conda activate hover_net_v2
Windows:
cmd
activate hover_net_v2

Step 4: Install PyTorch (GPU-Accelerated Version, Optional)
The project uses PyTorch for model inference. For GPU acceleration, install a version matching your system’s CUDA. Example for PyTorch 2.8.0 + CUDA 12.1:
bash
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
Additional notes:
For CPU-only usage, install the CPU version:
bash
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
If your CUDA version differs (e.g., CUDA 11.8), replace cu121 with the corresponding tag (e.g., cu118).

Step 5: Install Other Core Dependencies
Install remaining dependencies via pip (matched to key package versions in your environment):
bash
pip install \
    opencv-python==4.5.5.62 \
    numpy==1.23.5 \
    matplotlib==3.6.2 \
    openslide-python==1.4.2 \
    scikit-image==0.19.3 \
    tensorflow-gpu==2.10.0 \
    tqdm==4.64.1
Verify the Environment
After installation, check key package versions with:
bash
pip list | grep -E "torch|opencv-python|numpy|matplotlib|openslide-python|tensorflow-gpu"
If output matches your environment (e.g., below), the setup is successful:
plaintext
torch                2.8.0
opencv-python        4.5.5.62
numpy                1.23.5
matplotlib           3.6.2
openslide-python     1.4.2
tensorflow-gpu       2.10.0
tqdm                 4.64.1
This ensures the dependency environment matches your testing setup.

  Download Model Weights
HoVer-Net requires pre-trained weights. Download from the official release and place them in the model_weights/ directory.
Example structure:
plaintext
model_weights/
├── hovernet_original_consep_notype_tf2pytorch.tar
├── hovernet_fast_pannuke_type_pytorch.tar
└── ...
Note: Ensure the weight file matches your model mode (e.g., original weights for model_mode="original").
Usage
Basic Workflow
Prepare your input images (tile mode: .png/.jpg; WSI mode: .svs/.tif)
Configure parameters (see "Configuration" section)
Run inference
Check outputs in the specified directory
Run Tile Mode (for small images)
Process standard-sized images (e.g., 256×256px):
python
运行
# Run the main script
python infer.py
When prompted, input:
Input directory (where your tile images are stored)
Output directory (where results will be saved)
Run WSI Mode (for large slides)
Process large whole-slide images (requires OpenSlide):
Modify sub_cmd = "wsi" in the script
Adjust WSI-specific parameters (e.g., proc_mag, tile_shape)
Run the script and input paths as above
Configuration
Key parameters in infer.py (modify before running):
Parameter	Description
model_mode	Model structure: "original" (high accuracy) or "fast" (faster speed)
nr_types	Number of nuclei types to predict (0 = no type prediction, use with notype weights; >0 = type prediction, use with type weights)
model_path	Path to pre-trained weights (must match model_mode)
batch_size	Inference batch size (adjust based on GPU memory; 8-32 recommended)
gpu	GPU ID(s) to use (e.g., "0" for single GPU, "0,1" for multi-GPU)
Output
Results are saved in the specified output_dir with the following structure:
plaintext
output_dir/
├── overlay/          # Visualization: input images with segmentation overlay
├── qupath/           # QuPath-compatible annotations (.txt)
└── raw/              # Raw predictions (if `save_raw_map=True`)
Example Results
Input Image	Segmentation Result (Overlay)

Input
image

Output
image
Troubleshooting
Weight loading errors (e.g., "Missing key(s) in state_dict"): Ensure model_mode matches the weight file (e.g., original weights ↔ model_mode="original"), and nr_types matches the weight type (0 for notype weights, >0 for type weights).
GPU out-of-memory: Reduce batch_size or use a smaller model_mode (e.g., "fast").
WSI processing slow: Increase nr_inference_workers or reduce tile_shape.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments
Based on the original HoVer-Net implementation: vqdang/hover_net
Uses weights trained on public histopathology datasets (CoNSeP, PanNuke, etc.)
