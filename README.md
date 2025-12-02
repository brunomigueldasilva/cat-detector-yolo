# 🐱 Cat Detector

A project for exploring **YOLO** and **OpenCV** libraries for real-time cat detection and identification in video.

## About the Project

This project was developed as a hands-on way to learn and experiment with computer vision technologies, specifically:

- **YOLO (You Only Look Once)** - A family of real-time object detection models known for their speed and accuracy
- **OpenCV** - A widely-used library for image and video processing in computer vision applications

The main goal was to explore how these tools can be combined to create a system capable of detecting and identifying cats in real-time, either from video files or a webcam feed.

## Features

- **Cat detection** in video or webcam using YOLO models
- **Individual identification** of cats through colour analysis (distinguishes between black cats and others)
- **Segmentation** with precise masks for each detected cat
- **Multiple YOLO models** available (from lightweight to most accurate)
- **CUDA/GPU support** for accelerated processing
- **Frame skipping** optimisation for better performance
- **Video recording** of processed output with detections marked
- **Debug mode** for HSV channel and colour mask visualisation

## Project Structure

```
cat-detector/
├── main.py           # Entry point and mode selection
├── tracker.py        # Tracker implementations (Simple, Color, Segmentation)
├── config.py         # Configuration, constants, and interactive menu
├── requirements.txt  # Project dependencies
├── models/           # Directory for YOLO models (created automatically)
└── videos/           # Directory for video files
```

## Operating Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `detection` | Simple cat detection without individual identification | Basic tracking |
| `identification` | Identification through colour analysis (DALI vs INDY) | Distinguish cats by colour |
| `identification-segmentation` | Identification with precise segmentation masks | Higher precision in delimitation |

## Available Models

The project supports several models from the YOLO11 family, allowing you to choose the balance between speed and accuracy:

| Model | Size | Speed | Accuracy | Recommendation |
|-------|------|-------|----------|----------------|
| `yolo11n` | Nano | ⚡⚡⚡⚡⚡ | ⭐⭐ | Resource-limited devices |
| `yolo11s` | Small | ⚡⚡⚡⚡ | ⭐⭐⭐ | Good balance |
| `yolo11m` | Medium | ⚡⚡⚡ | ⭐⭐⭐⭐ | **Recommended** |
| `yolo11l` | Large | ⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy |
| `yolo11x` | Extra Large | ⚡ | ⭐⭐⭐⭐⭐ | Maximum accuracy |

For segmentation mode, `-seg` variants of each model are available (e.g., `yolo11m-seg.pt`).

## Installation

### Requirements

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) NVIDIA GPU with CUDA for acceleration

### Steps

1. **Clone or download the project**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python main.py
   ```
   The program will automatically check if all dependencies are installed.

## Usage

### Interactive Mode

```bash
python main.py
```

The program will present an interactive menu where you can configure all options.

### Command Line

```bash
python main.py detection                    # Detection only
python main.py identification               # Colour-based identification
python main.py identification-segmentation  # Identification with segmentation
```

## Configuration Parameters

### Main Configuration (`Config` class)

The `Config` dataclass in `config.py` contains all the main settings for the tracker:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | `'yolo11m.pt'` | Path to the YOLO model file. Automatically downloaded if not present. |
| `skip_frames` | `int` | `1` | Number of frames to skip between detections. `0` processes every frame, `1` processes every 2nd frame, etc. Higher values improve performance but reduce tracking smoothness. |
| `confidence_threshold` | `float` | `0.25` | Minimum confidence score (0.0-1.0) for a detection to be considered valid. Lower values detect more cats but may include false positives. |
| `use_cuda` | `bool` | `False` | Whether to use GPU acceleration. Requires NVIDIA GPU with CUDA installed. |
| `debug_mode` | `bool` | `False` | Enables debug visualisation showing HSV channels and colour detection masks. Useful for tuning colour thresholds. |
| `video_source` | `any` | `0` | Video input source. Use `0` for default webcam, or a file path string for video files. |
| `output_path` | `str` or `None` | `None` | Path to save the processed video. If `None`, video is not saved. |
| `show_live` | `bool` | `True` | Whether to display the video during processing. Disable for headless processing. |
| `processing_resolution` | `tuple` or `None` | `None` | Resolution for processing as `(width, height)`. `None` uses original resolution. Lower resolutions improve performance. |

### Colour Detection Thresholds (`ColorThresholds` class)

The colour-based identification uses HSV (Hue, Saturation, Value) colour space analysis. These thresholds can be fine-tuned in `config.py`:

#### Black Cat Detection (DALI)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `black_brightness_max` | `75` | Maximum V (brightness) value to consider a pixel as black. Range: 0-255. |
| `black_saturation_max` | `80` | Maximum S (saturation) value for black pixels. Low saturation indicates grey/black tones. Range: 0-255. |
| `black_percentage_threshold` | `20` | Minimum percentage of pixels that must be classified as black to identify a cat as black. Range: 0-100. |

#### Brown/Orange Cat Detection (INDY)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `brown_hue_min` | `20` | Minimum H (hue) value for brown/orange detection. Range: 0-179 in OpenCV. |
| `brown_hue_max` | `44` | Maximum H (hue) value for brown/orange detection. This range covers orange to brown tones. |
| `brown_saturation_min` | `10` | Minimum S (saturation) for brown detection. Filters out grey tones. |
| `brown_saturation_max` | `200` | Maximum S (saturation) for brown detection. |
| `brown_brightness_min` | `80` | Minimum V (brightness) for brown detection. Filters out very dark pixels. |
| `brown_brightness_max` | `180` | Maximum V (brightness) for brown detection. Filters out very bright pixels. |
| `brown_percentage_threshold` | `20` | Minimum percentage of pixels that must be brown to identify a cat as INDY. |

#### General Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_detections_required` | `3` | Minimum number of consistent detections before assigning an identity. Helps prevent flickering between identities. |

### Display Colours

The following BGR colour constants are defined for visualisation:

| Constant | Value (BGR) | Usage |
|----------|-------------|-------|
| `COLOR_DEFAULT` | `(255, 0, 0)` | Blue - Default bounding box colour |
| `COLOR_DALI` | `(0, 0, 0)` | Black - Bounding box for DALI (black cat) |
| `COLOR_INDY` | `(0, 165, 255)` | Orange - Bounding box for INDY |
| `COLOR_TEXT` | `(255, 255, 255)` | White - Text labels |
| `MASK_COLOR_DALI` | `(50, 50, 50)` | Dark grey - Segmentation mask for DALI |
| `MASK_COLOR_INDY` | `(0, 140, 255)` | Orange - Segmentation mask for INDY |
| `MASK_ALPHA` | `0.4` | Transparency of segmentation masks |

### Processing Resolution Presets

The interactive menu offers the following resolution presets:

| Option | Resolution | Aspect Ratio | Orientation |
|--------|------------|--------------|-------------|
| Original | Native | - | - |
| Full HD Portrait | 1080×1920 | 9:16 | Portrait |
| HD Portrait | 720×1280 | 9:16 | Portrait ⭐ |
| qHD Portrait | 540×960 | 9:16 | Portrait |
| Full HD Landscape | 1920×1080 | 16:9 | Landscape |
| HD Landscape | 1280×720 | 16:9 | Landscape |
| Custom | User-defined | - | - |

## Controls During Execution

| Key | Action | Notes |
|-----|--------|-------|
| `q` | Quit the program | **Important:** The OpenCV video window must have focus (be the active window) for this to work. Click on the video window first, then press 'q'. |
| `r` / `R` | Reset colour thresholds | Only in debug mode |
| `p` / `P` | Print current thresholds | Only in debug mode |
| `Ctrl+C` | Force quit | Works from the terminal |

> ⚠️ **Note:** If pressing 'q' doesn't quit the program, make sure you've clicked on the video window to give it focus. Pressing 'q' in the terminal window won't work - the key must be pressed while the OpenCV video window is active.

## Technical Notes

- YOLO models are automatically downloaded on first run
- The cat class in the COCO dataset corresponds to ID 15
- Display colours are in BGR format (OpenCV standard)
- The integrated timing system allows performance analysis of each operation
- The `TimingStats` class collects detailed timing statistics that are printed at the end of processing

## Dependencies

```
opencv-python>=4.8.0
ultralytics>=8.3.0
numpy>=1.24.0
```

PyTorch is automatically installed as a dependency of Ultralytics.

## Troubleshooting

### Common Issues

1. **Pressing 'q' doesn't quit** - The OpenCV video window must have focus. Click on the video window first, then press 'q'. Alternatively, use `Ctrl+C` in the terminal to force quit.

2. **"CUDA not available"** - Ensure you have an NVIDIA GPU and CUDA toolkit installed. Alternatively, use CPU mode.

2. **Low FPS** - Try:
   - Increasing `skip_frames`
   - Using a smaller model (e.g., `yolo11n`)
   - Reducing `processing_resolution`
   - Enabling CUDA if you have a compatible GPU

3. **Wrong cat identification** - Adjust the colour thresholds in `ColorThresholds` class. Enable `debug_mode` to visualise the colour detection.

4. **Model download fails** - Ensure you have internet connectivity. Models are downloaded from Ultralytics servers.

---

*Project developed for educational purposes and computer vision technology exploration.*
