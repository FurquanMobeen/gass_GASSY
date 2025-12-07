# 🧠 Gas Bottle Detection & Tracking using YOLO

## 📘 Project Context

This project is part of an AI-based monitoring system designed to **detect and track gas bottles moving on a conveyor belt** in real-time.  
Using a fine-tuned YOLO model from [Ultralytics](https://github.com/ultralytics/ultralytics), the solution provides automated detection, labeling, and tracking directly from live video streams or stored footage.

The system supports **on-site deployment** and aims to enhance:
- Quality control  
- Production efficiency  
- Safety monitoring  
- Inventory tracking  

## 🚀 Getting Started

Follow these steps to set up the project on your local machine.

### 1️⃣ Prerequisites
- **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **Visual Studio Code** (Recommended): For editing and running code.

### 2️⃣ Installation

1. **Clone the repository**
   Open your terminal (Command Prompt or Terminal) and run:
   ```bash
   git clone https://github.com/FurquanMobeen/gass_GASSY.git
   cd gass_GASSY
   ```

2. **Create a Virtual Environment (Recommended)**
   This keeps your project libraries separate from other projects.
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   Install all required AI and vision libraries:
   ```bash
   pip install ultralytics opencv-python easyocr torch torchvision
   ```
   *(Note: If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org/) for faster performance.)*

### 3️⃣ Project Structure
Ensure your folders are organized so the scripts can find the models and videos:

```text
gass_GASSY/
├── models/                  # 📂 Place your AI models here
│   ├── yolo11n_bottles.pt
│   ├── bottle_classifier_fold_2.pth
│   └── new_yolo11s_extract_tarra_weights.pt
├── videos/                  # 📂 Place your input videos here
│   └── 14_55_front_cropped.mp4
├── track-gas-bottle.py      # 🐍 Main tracking script
└── README.md
```

---

## 🏃‍♂️ How to Run

To simply detect and track gas bottles in the video:

For Windows:
```bash
python track-gas-bottle.py
```

For macOS:
```bash
python3 track-gas-bottle.py
```

### ⚙️ Configuration
Make sure to fill in all the models and videos and bytetrack paths in `config.py` file.

All the models can be downloaded here: <br> https://ucll-my.sharepoint.com/:f:/r/personal/r0975382_ucll_be/Documents/AI%20Applications%20-%20Team%20Gassy/Models?csf=1&web=1&e=jxcnzj

## Main algorithms
- `track-gas-bottle.py` : Gas bottle tracking from one source video.


## 🧩 Customer Requirements

The customer requested the following capabilities:

- ✅ Real-time **detection and tracking** of gas bottles on conveyor belts.  
- ✅ Clear **bounding boxes and confidence levels** displayed on live video.  
- ✅ Reliable performance under **different lighting and motion conditions**.  
- ✅ Easy-to-run Python scripts for local computers.  
- ✅ Extendable dataset and training workflow for future updates.

## ⚙️ General Algorithm Architecture

### 🤖 Gas Bottle Inspection and Tracking Algorithm

This algorithm is a comprehensive computer vision pipeline designed to **track, inspect, and log data** from gas bottles moving in a video feed. It acts as a **smart automated inspector** that watches the video, spots the bottles, gives them unique IDs, checks if they're safe (OK/NOT OK), and tries to read their tarra weight and recertification year.

### 🛠️ Core Components and Technologies

The algorithm relies on several advanced machine learning models and libraries:

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Object Detection** | **YOLOv8** (from `ultralytics`) | Detects the **location** of the gas bottles (the object) and the **location** of the stamped text (tarra weight/year). |
| **Tracking** | Integrated Tracker (e.g., BOT-SORT/ByteTrack) | Assigns a stable, consistent **ID** to each bottle as it moves across frames. |
| **Classification** | **ConvNeXtV2** (via `timm`) | A trained model for **binary classification** (OK/NOT OK) of the bottle's visual condition. |
| **Text Reading** | **EasyOCR** | An Optical Character Recognition library used to **read** the numbers/text (weight and year) from the detected text regions. |

---

### ⚙️ Key Operational Functions

#### 1. Tracking & Identification 🎯

* **Goal:** To give each individual gas bottle a unique ID and follow it throughout the video.
* **Process:** The `yolo.track()` function assigns an internal `track_id`. The code maps this to a clean, consecutive `display_id` for reporting (`id_mapping`).

#### 2. Condition Classification ✅/❌

* **Goal:** Determine the safety status of the bottle (OK or NOT OK).
* **Process:** The `classify_gas_bottle` function extracts the bottle's image (Region of Interest or **ROI**) and feeds it to the **ConvNeXtV2 classifier**.
* **Data Storage:** The highest confidence classification and label (`ok` or `nok`) are stored persistently in the `id_ocr_data` dictionary for that bottle's ID.

#### 3. Reading Stamped Data (OCR) 🔢

* **Goal:** Extract the **Tarra weight** (e.g., 10.8 kg) and **Recertification Year** (e.g., 2025) stamped on the bottle.
* **Function:** `extract_text_with_ocr`
    * Uses **`text_yolo`** to precisely locate the tiny stamped text regions.
    * Applies **image preprocessing** (CLAHE, rotations) to enhance the often faint text.
    * Uses **`EasyOCR`** to read the characters.
    * **Stabilization:** Stores multiple OCR results per bottle (`ocr_memory`) and uses a **majority vote** (`Counter`) to select the most *stable* reading over time, increasing reliability.
    * **Validation:** Uses **Regular Expressions** to confirm the extracted values are valid weights ($\text{X.X}$) and years ($\text{20XX}$).
    * **Automatic Flagging:** If the extracted recertification year is older than the `CURRENT_YEAR`, the bottle is automatically flagged as **"nok (expired)"**.

#### 4. Output and Reporting 📊

* **Video Output:** Draws the bounding boxes, `ID`, classification, Tarra, and Year directly onto the video frames before saving to the `output_path` (e.g., `videos/output/...mp4`).
* **CSV Log:** The `save_csv_data` function writes all the final, stabilized data for every tracked bottle into a structured `.csv` file (`_tracking_data.csv`).
* **Performance Analysis (Optional):** If a **Ground Truth** file is found, the algorithm:
    * Compares its classifications against the known labels.
    * Calculates a **Confusion Matrix** and **Classification Report**.
    * Generates a heatmap image of the Confusion Matrix to visually represent the model's accuracy.

---

## 🏆 Latest Performance Results
### (New) YOLO11s Model Object detection:
The model achieved excellent detection accuracy on photo-based validation datasets.
| Metric | Value | Description |
|:-------|:------:|:------------|
| 🧩 **mAP@0.5** | **0.9932**  | Mean Average Precision at IoU 0.5 |
| 🎯 **Precision** | **0.9811**  | Correct detections among predicted positives |
| 🔍 **Recall** | **0.9832** | True detections among actual positives |
| ⚖️ **F1 Score** | **0.9821** | Best balance between precision and recall |

### EfficientNet_B0 classification:
- Early stopping successfully prevented overfitting across folds and reduced unnecessary computation. 
- Validation performance was consistently high across folds, demonstrating the model’s stability and ability to generalize.. 
- Fold 2 achieved perfect validation accuracy, showing that the dataset is well-learned by the model.
- Overall, the model demonstrates strong generalization, with an average validation accuracy of 98.63% across 5 folds.
- The loss and accuracy curves provide a clear visual understanding of the learning process and model stability. 

### 📊 Performance Visualization
The chart below summarizes the algorithm’s performance metrics:
- Precision–Recall Curve: mAP@0.5 = 0.992
- F1–Confidence Curve: Peak F1 = 0.96 at confidence 0.79
- Precision–Confidence & Recall–Confidence Curves: Stable up to ~0.8 confidence

🧪 Performance evaluation was performed on fine-tuned YOLO11x trained with real site photos.

---