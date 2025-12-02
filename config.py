"""Configuration, constants, menu, and utilities for cat detector."""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from collections import defaultdict
import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

CAT_CLASS_ID = 15  # COCO dataset cat class

# Display colors (BGR format)
COLOR_DEFAULT = (255, 0, 0)
COLOR_DALI = (0, 0, 0)
COLOR_INDY = (0, 165, 255)
COLOR_TEXT = (255, 255, 255)

# Segmentation mask colors
MASK_COLOR_DALI = (50, 50, 50)
MASK_COLOR_INDY = (0, 140, 255)
MASK_ALPHA = 0.4

# Cat names
NAME_BLACK_CAT = "DALI"
NAME_OTHER_CAT = "INDY"

# Models
DETECTION_MODELS = {
    '1': 'yolo11n.pt',
    '2': 'yolo11s.pt',
    '3': 'yolo11m.pt',
    '4': 'yolo11l.pt',
    '5': 'yolo11x.pt'
}

SEGMENTATION_MODELS = {
    '1': 'yolo11n-seg.pt',
    '2': 'yolo11s-seg.pt',
    '3': 'yolo11m-seg.pt',
    '4': 'yolo11l-seg.pt',
    '5': 'yolo11x-seg.pt'
}


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class ColorThresholds:
    """Thresholds for color detection."""
    black_brightness_max: int = 75
    black_saturation_max: int = 80
    black_percentage_threshold: int = 20
    brown_hue_min: int = 20
    brown_hue_max: int = 44
    brown_saturation_min: int = 10
    brown_saturation_max: int = 200
    brown_brightness_min: int = 80
    brown_brightness_max: int = 180
    brown_percentage_threshold: int = 20
    min_detections_required: int = 3


@dataclass
class Config:
    """Complete configuration for the tracker."""
    # Tracker settings
    model_path: str = 'yolo11m.pt'
    skip_frames: int = 1
    confidence_threshold: float = 0.25
    use_cuda: bool = False
    debug_mode: bool = False
    color_thresholds: ColorThresholds = field(default_factory=ColorThresholds)

    # Video settings
    video_source: any = 0
    output_path: Optional[str] = None
    show_live: bool = True
    processing_resolution: Optional[Tuple[int, int]] = None


# =============================================================================
# TIMING UTILITIES
# =============================================================================

class TimingStats:
    """Collects timing statistics for operations."""

    def __init__(self):
        self._times: Dict[str, List[float]] = defaultdict(list)

    def record(self, operation: str, duration: float):
        self._times[operation].append(duration)

    def get_averages_ms(self) -> Dict[str, float]:
        return {op: np.mean(times) * 1000 if times else 0.0
                for op, times in self._times.items()}

    def print_summary(self):
        print("\n⏱️  Detailed Timing Statistics:")
        print("-" * 60)
        for op, times in self._times.items():
            if times:
                avg = np.mean(times) * 1000
                min_t = np.min(times) * 1000
                max_t = np.max(times) * 1000
                std = np.std(times) * 1000
                print(
                    f"{
                        op:20s}: avg={
                        avg:7.3f}ms  min={
                        min_t:7.3f}ms  max={
                        max_t:7.3f}ms  std={
                        std:7.3f}ms")
            else:
                print(f"{op:20s}: No data")
        print("-" * 60)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, stats: TimingStats, operation: str):
        self.stats = stats
        self.operation = operation

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.stats.record(self.operation, time.perf_counter() - self.start)


# =============================================================================
# INSTALLATION CHECK
# =============================================================================

def check_installation() -> bool:
    """Check if all dependencies are installed."""
    print("\n" + "=" * 60)
    print("🔍 CHECKING INSTALLATION")
    print("=" * 60 + "\n")

    all_ok = True

    # Check OpenCV
    try:
        import cv2
        print(f"✅ OpenCV installed (version {cv2.__version__})")
    except ImportError:
        print("❌ OpenCV not found")
        print("   Run: pip install opencv-python")
        all_ok = False

    # Check NumPy
    try:
        import numpy as np
        print(f"✅ NumPy installed (version {np.__version__})")
    except ImportError:
        print("❌ NumPy not found")
        print("   Run: pip install numpy")
        all_ok = False

    # Check Ultralytics
    try:
        import ultralytics
        print(f"✅ Ultralytics installed (version {ultralytics.__version__})")
    except ImportError:
        print("❌ Ultralytics not found")
        print("   Run: pip install ultralytics")
        all_ok = False

    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch installed (version {torch.__version__})")
        if torch.cuda.is_available():
            print(f"   🎮 CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️  CUDA not available (using CPU)")
    except ImportError:
        print("❌ PyTorch not found")
        print("   Run: pip install torch")
        all_ok = False

    # Check directories
    for d in ['models', 'videos']:
        if not os.path.exists(d):
            print(f"\n📁 Creating '{d}' directory...")
            os.makedirs(d, exist_ok=True)
            print(f"✅ '{d}' directory created")
        else:
            print(f"\n✅ '{d}' directory exists")

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ SYSTEM READY TO USE!")
    else:
        print("❌ INSTALLATION HAS PROBLEMS")
        print("\nInstall missing dependencies and try again.")
    print("=" * 60 + "\n")

    return all_ok


# =============================================================================
# INTERACTIVE MENU
# =============================================================================

def show_menu(mode: str = 'color') -> Config:
    """Show interactive menu and return configuration."""
    config = Config()
    use_seg = mode == 'identification-segmentation'
    models = SEGMENTATION_MODELS if use_seg else DETECTION_MODELS

    titles = {
        'detection': 'CAT DETECTOR - TRACKING ONLY',
        'identification': 'CAT DETECTOR - DALI vs INDY',
        'identification-segmentation': 'CAT DETECTOR - SEGMENTATION VERSION'
    }

    print("\n" + "=" * 60)
    print(f"🐱 {titles.get(mode, titles['identification'])}")
    print("=" * 60 + "\n")

    # Video source
    print("📹 VIDEO SOURCE:")
    print("  1. Webcam")
    print("  2. Video file")

    choice = input("\nChoice (1-2) [2]: ").strip() or "2"

    if choice == "1":
        config.video_source = 0
        print("✅ Selected: Webcam\n")
    else:
        print("\n📁 Available files in 'videos/':")
        if os.path.exists('videos'):
            video_files = [f for f in os.listdir('videos')
                           if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if video_files:
                for i, f in enumerate(video_files, 1):
                    print(f"  {i}. {f}")

                file_choice = input(
                    f"\nChoose a file (1-{len(video_files)}) or enter full path: ").strip()

                try:
                    idx = int(file_choice) - 1
                    if 0 <= idx < len(video_files):
                        config.video_source = os.path.join(
                            'videos', video_files[idx])
                    else:
                        config.video_source = file_choice
                except ValueError:
                    config.video_source = file_choice
            else:
                print("  (no files found)")
                config.video_source = input("\nVideo path: ").strip()
        else:
            config.video_source = input("\nVideo path: ").strip()

        print(f"✅ Selected: {config.video_source}\n")

    # Model selection
    if use_seg:
        print("🎭 YOLO SEGMENTATION MODEL:")
        print("  1. yolo11n-seg.pt (fastest, less accurate)")
        print("  2. yolo11s-seg.pt (balanced)")
        print("  3. yolo11m-seg.pt (medium, best choice)")
        print("  4. yolo11l-seg.pt (large)")
        print("  5. yolo11x-seg.pt (most accurate, slower)")
    else:
        print("🤖 YOLO MODEL:")
        print("  1. yolo11n.pt (fastest, less accurate)")
        print("  2. yolo11s.pt (balanced)")
        print("  3. yolo11m.pt (medium, best choice)")
        print("  4. yolo11l.pt (large)")
        print("  5. yolo11x.pt (most accurate, slower)")

    model_choice = input("\nChoice (1-5) [3]: ").strip() or "3"
    config.model_path = models.get(model_choice, models['3'])
    print(f"✅ Selected: {config.model_path}\n")

    # CUDA selection
    try:
        import torch
        if torch.cuda.is_available():
            print("🎮 DEVICE SELECTION:")
            print("  1. CUDA (GPU) - Faster")
            print("  2. CPU - Slower but compatible")

            device_choice = input("\nChoice (1-2) [2]: ").strip() or "2"
            config.use_cuda = device_choice == "1"
            print(
                f"✅ Selected: {
                    'CUDA (GPU)' if config.use_cuda else 'CPU'}\n")
        else:
            print("⚠️  CUDA not available, will use CPU\n")
            config.use_cuda = False
    except ImportError:
        config.use_cuda = False

    # Skip frames
    print("⏭️  FRAME OPTIMIZATION:")
    print("  0 = Process all frames (slower, more accurate)")
    print("  1 = Process 1 in every 2 frames (2x faster)")
    print("  2 = Process 1 in every 3 frames (3x faster)")
    print("  3 = Process 1 in every 4 frames (4x faster)")

    skip = input("\nFrames to skip [1]: ").strip() or "1"
    try:
        config.skip_frames = int(skip)
    except ValueError:
        config.skip_frames = 1
    print(f"✅ Processing 1 in every {config.skip_frames + 1} frames\n")

    # Confidence threshold
    print("🎯 CONFIDENCE THRESHOLD:")
    confidence = input(
        "Minimum confidence (0.0-1.0) [0.25]: ").strip() or "0.25"
    try:
        config.confidence_threshold = float(confidence)
        config.confidence_threshold = max(
            0.0, min(1.0, config.confidence_threshold))
    except ValueError:
        config.confidence_threshold = 0.25
    print(f"✅ Minimum confidence: {config.confidence_threshold}\n")

    # Processing resolution
    print("📐 PROCESSING RESOLUTION:")
    print("  1. Original (no resize)")
    print("  2. 1080x1920 (Full HD Portrait, 9:16)")
    print("  3. 720x1280 (HD Portrait, 9:16) - Recommended")
    print("  4. 540x960 (qHD Portrait, 9:16) - Faster")
    print("  5. 1920x1080 (Full HD Landscape, 16:9)")
    print("  6. 1280x720 (HD Landscape, 16:9)")
    print("  7. Custom resolution")

    res_choice = input("\nChoice (1-7) [1]: ").strip() or "1"

    if res_choice == "1":
        config.processing_resolution = None
        print("✅ Using original resolution\n")
    elif res_choice == "2":
        config.processing_resolution = (1080, 1920)
        print("✅ Processing at: 1080x1920 (9:16 Portrait)\n")
    elif res_choice == "3":
        config.processing_resolution = (720, 1280)
        print("✅ Processing at: 720x1280 (9:16 Portrait) ⭐\n")
    elif res_choice == "4":
        config.processing_resolution = (540, 960)
        print("✅ Processing at: 540x960 (9:16 Portrait)\n")
    elif res_choice == "5":
        config.processing_resolution = (1920, 1080)
        print("✅ Processing at: 1920x1080 (16:9 Landscape)\n")
    elif res_choice == "6":
        config.processing_resolution = (1280, 720)
        print("✅ Processing at: 1280x720 (16:9 Landscape)\n")
    elif res_choice == "7":
        try:
            custom_width = int(input("  Width: ").strip())
            custom_height = int(input("  Height: ").strip())
            config.processing_resolution = (custom_width, custom_height)
            aspect_ratio = custom_width / custom_height
            orientation = "Portrait" if custom_height > custom_width else "Landscape"
            print(
                f"✅ Processing at: {custom_width}x{custom_height} ({orientation}, aspect ratio: {
                    aspect_ratio:.2f}:1)\n")
        except ValueError:
            print("⚠️  Invalid input, using 720x1280")
            config.processing_resolution = (720, 1280)
            print("✅ Processing at: 720x1280 (9:16 Portrait)\n")
    else:
        config.processing_resolution = (720, 1280)
        print("✅ Processing at: 720x1280 (9:16 Portrait)\n")

    # Output video
    print("💾 SAVE PROCESSED VIDEO:")
    save = input("Save video? (y/n) [n]: ").strip().lower() or "n"

    if save == 'y':
        default_output = "output_video.mp4"
        output_path = input(
            f"File name [{default_output}]: ").strip() or default_output
        config.output_path = output_path
        print(f"✅ Video will be saved to: {config.output_path}\n")
    else:
        config.output_path = None
        print("✅ Video will not be saved\n")

    # Live display
    print("🖥️  LIVE VISUALIZATION:")
    display = input(
        "Show video during processing? (y/n) [y]: ").strip().lower() or "y"
    config.show_live = display == 'y'
    print(
        f"✅ Visualization: {
            'Enabled' if config.show_live else 'Disabled'}\n")

    # Debug mode (only for identification/segmentation modes)
    if mode in ('identification', 'identification-segmentation'):
        print("\n🔍 DEBUG MODE:")
        print("  Shows HSV channels and color detection masks")
        print("  Useful for understanding and adjusting color thresholds")
        debug = input("Enable debug mode? (y/n) [n]: ").strip().lower() or "n"
        config.debug_mode = debug == 'y'
        print(
            f"✅ Debug mode: {
                'Enabled - HSV visualization will be shown' if config.debug_mode else 'Disabled'}\n")

    print("=" * 60)
    print("⚙️  CONFIGURATION COMPLETE")
    print("=" * 60 + "\n")

    return config
