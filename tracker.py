"""Cat tracker implementations - Simple, Color-based, and Segmentation."""

import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    Config, ColorThresholds, TimingStats, Timer,
    CAT_CLASS_ID, COLOR_DEFAULT, COLOR_DALI, COLOR_INDY, COLOR_TEXT,
    MASK_COLOR_DALI, MASK_COLOR_INDY, MASK_ALPHA,
    NAME_BLACK_CAT, NAME_OTHER_CAT
)


# =============================================================================
# COLOR ANALYZER
# =============================================================================

class ColorAnalyzer:
    """Analyzes HSV colors to distinguish black vs brown cats."""

    def __init__(self, thresholds: ColorThresholds):
        self.th = thresholds

    def analyze(self,
                image: np.ndarray,
                mask: np.ndarray = None) -> Tuple[str,
                                                  float,
                                                  float,
                                                  float,
                                                  np.ndarray,
                                                  np.ndarray]:
        """
        Analyze color in image region.

        Returns: (color_type, confidence, black_pct, brown_pct, black_mask, brown_mask)
        """
        if image.size == 0:
            return 'unknown', 0.0, 0.0, 0.0, None, None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Analysis area
        if mask is not None:
            area_mask = mask > 0
            total = np.sum(area_mask)
            if total == 0:
                return 'unknown', 0.0, 0.0, 0.0, None, None
        else:
            area_mask = np.ones(v.shape, dtype=bool)
            total = v.size

        # Black detection
        black_pixels = (v < self.th.black_brightness_max) & (
            s < self.th.black_saturation_max) & area_mask
        black_mask = self._clean_mask(black_pixels.astype(
            np.uint8) * 255) if mask is None else black_pixels.astype(np.uint8) * 255
        black_pct = (np.sum(black_mask > 127) / total) * 100

        # Brown detection
        brown_pixels = (
            (h >= self.th.brown_hue_min) & (
                h <= self.th.brown_hue_max) & (
                s >= self.th.brown_saturation_min) & (
                s <= self.th.brown_saturation_max) & (
                    v >= self.th.brown_brightness_min) & (
                        v <= self.th.brown_brightness_max) & area_mask)
        brown_mask = self._clean_mask(brown_pixels.astype(
            np.uint8) * 255) if mask is None else brown_pixels.astype(np.uint8) * 255
        brown_pct = (np.sum(brown_mask > 127) / total) * 100

        # Determine winner
        if black_pct >= self.th.black_percentage_threshold and black_pct > brown_pct:
            return 'black', black_pct, black_pct, brown_pct, black_mask, brown_mask
        elif brown_pct >= self.th.brown_percentage_threshold:
            return 'brown', brown_pct, black_pct, brown_pct, black_mask, brown_mask
        return 'unknown', max(
            black_pct, brown_pct), black_pct, brown_pct, black_mask, brown_mask

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean noise."""
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.medianBlur(mask, 7)


# =============================================================================
# CAT IDENTIFIER
# =============================================================================

class CatIdentifier:
    """Identifies cats based on color history."""

    def __init__(self, min_detections: int = 2):
        self.min_detections = min_detections
        self._names: Dict[int, str] = {}
        self._history: Dict[int, List[int]] = defaultdict(list)

    def get_name(self, track_id: int, color_type: str) -> str:
        """Get or assign cat name based on color detection history."""
        # Record detection (1=black, 0=brown, -1=unknown)
        value = 1 if color_type == 'black' else (
            0 if color_type == 'brown' else -1)
        history = self._history[track_id]
        history.append(value)
        if len(history) > 10:
            history.pop(0)

        # Return if already assigned
        if track_id in self._names:
            return self._names[track_id]

        # Try to assign based on history
        if len(history) >= self.min_detections:
            black_count = sum(1 for x in history if x == 1)
            brown_count = sum(1 for x in history if x == 0)
            known = black_count + brown_count

            if known >= self.min_detections:
                if black_count > brown_count and black_count >= known * 0.6:
                    self._names[track_id] = NAME_BLACK_CAT
                elif brown_count > black_count and brown_count >= known * 0.6:
                    self._names[track_id] = NAME_OTHER_CAT

            # Force assignment after many frames
            if len(history) >= 8 and track_id not in self._names:
                self._names[track_id] = NAME_BLACK_CAT if black_count >= brown_count else NAME_OTHER_CAT

        # Return assigned or temporary name
        if track_id in self._names:
            return self._names[track_id]

        # Temporary name
        if color_type == 'black':
            return NAME_BLACK_CAT
        elif color_type == 'brown':
            return NAME_OTHER_CAT

        black_count = sum(1 for x in history if x == 1)
        brown_count = sum(1 for x in history if x == 0)
        return NAME_BLACK_CAT if black_count >= brown_count else NAME_OTHER_CAT

    @property
    def named_cats(self) -> Dict[int, str]:
        return dict(self._names)

    def print_summary(self):
        if self._names:
            print("\n🐱 Detected cats:")
            for tid, name in sorted(self._names.items()):
                print(f"  • ID {tid}: {name}")
        else:
            print("\n⚠️  No cats detected in video")

# =============================================================================
# DEBUG DISPLAY
# =============================================================================


class DebugDisplay:
    """HSV debug visualization with full grid layout."""

    # Fixed window dimensions
    WINDOW_WIDTH = 900
    WINDOW_HEIGHT = 700
    INFO_HEIGHT = 100

    def __init__(self, thresholds: ColorThresholds):
        self.th = thresholds
        self._grids = {NAME_BLACK_CAT: None, NAME_OTHER_CAT: None}
        self._created = False

    def create_windows(self):
        if self._created:
            return
        cv2.namedWindow('HSV Debug - DALI (Black Cat)', cv2.WINDOW_NORMAL)
        cv2.namedWindow('HSV Debug - INDY (Other Cat)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            'HSV Debug - DALI (Black Cat)',
            self.WINDOW_WIDTH,
            self.WINDOW_HEIGHT)
        cv2.resizeWindow(
            'HSV Debug - INDY (Other Cat)',
            self.WINDOW_WIDTH,
            self.WINDOW_HEIGHT)
        self._create_trackbars()
        self._created = True
        print(
            "🔍 Debug mode: Separate window for each cat + thresholds window + BROWN sliders")
        print("   Press 'R' to reset BROWN thresholds | Press 'P' to print current values")

    def _create_trackbars(self):
        win = 'BROWN Threshold Adjustments (INDY)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 500, 350)
        cv2.createTrackbar(
            'Hue Min',
            win,
            self.th.brown_hue_min,
            179,
            lambda v: setattr(
                self.th,
                'brown_hue_min',
                v))
        cv2.createTrackbar(
            'Hue Max',
            win,
            self.th.brown_hue_max,
            179,
            lambda v: setattr(
                self.th,
                'brown_hue_max',
                v))
        cv2.createTrackbar(
            'Sat Min',
            win,
            self.th.brown_saturation_min,
            255,
            lambda v: setattr(
                self.th,
                'brown_saturation_min',
                v))
        cv2.createTrackbar(
            'Sat Max',
            win,
            self.th.brown_saturation_max,
            255,
            lambda v: setattr(
                self.th,
                'brown_saturation_max',
                v))
        cv2.createTrackbar(
            'Bright Min',
            win,
            self.th.brown_brightness_min,
            255,
            lambda v: setattr(
                self.th,
                'brown_brightness_min',
                v))
        cv2.createTrackbar(
            'Bright Max',
            win,
            self.th.brown_brightness_max,
            255,
            lambda v: setattr(
                self.th,
                'brown_brightness_max',
                v))

    def update(
            self,
            cat_name: str,
            region: np.ndarray,
            black_mask: np.ndarray,
            brown_mask: np.ndarray,
            black_pct: float = 0.0,
            brown_pct: float = 0.0,
            track_id: int = None):
        """Update debug grid with full HSV visualization."""
        if region.size == 0:
            return

        # Calculate cell dimensions
        cell_width = self.WINDOW_WIDTH // 3
        cell_height = (self.WINDOW_HEIGHT - self.INFO_HEIGHT) // 2

        # Convert to HSV and extract channels
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        brightness = hsv[:, :, 2]

        # Calculate mean values
        mean_h = np.mean(hue)
        mean_s = np.mean(saturation)
        mean_v = np.mean(brightness)

        # Resize original
        original_resized = cv2.resize(region, (cell_width, cell_height))

        # Create hue visualization (colorful)
        hue_resized = cv2.resize(hue, (cell_width, cell_height))
        hue_color = cv2.cvtColor(
            cv2.merge([hue_resized,
                      np.full((cell_height, cell_width), 255, dtype=np.uint8),
                      np.full((cell_height, cell_width), 255, dtype=np.uint8)]),
            cv2.COLOR_HSV2BGR
        )

        # Create saturation and brightness visualizations
        saturation_color = cv2.applyColorMap(cv2.resize(
            saturation, (cell_width, cell_height)), cv2.COLORMAP_BONE)
        brightness_color = cv2.applyColorMap(cv2.resize(
            brightness, (cell_width, cell_height)), cv2.COLORMAP_BONE)

        # Color the masks
        black_resized = cv2.resize(
            black_mask, (cell_width, cell_height)) if black_mask is not None else np.zeros(
            (cell_height, cell_width), dtype=np.uint8)
        black_colored = cv2.cvtColor(black_resized, cv2.COLOR_GRAY2BGR)
        black_colored[black_resized > 127] = [0, 0, 255]  # Red

        brown_resized = cv2.resize(
            brown_mask, (cell_width, cell_height)) if brown_mask is not None else np.zeros(
            (cell_height, cell_width), dtype=np.uint8)
        brown_colored = cv2.cvtColor(brown_resized, cv2.COLOR_GRAY2BGR)
        brown_colored[brown_resized > 127] = [0, 165, 255]  # Orange

        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        cv2.putText(original_resized, "Original", (10, 25),
                    font, font_scale, (255, 255, 255), thickness)
        cv2.putText(
            hue_color, f"Hue: {
                mean_h:.1f}", (10, 25), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(
            saturation_color, f"Sat: {
                mean_s:.1f}", (10, 25), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(
            brightness_color, f"Bright: {
                mean_v:.1f}", (10, 25), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(
            black_colored,
            f"BLACK: {
                black_pct:.1f}%",
            (10,
             25),
            font,
            font_scale,
            (255,
             255,
             255),
            thickness)
        cv2.putText(
            brown_colored,
            f"BROWN: {
                brown_pct:.1f}%",
            (10,
             25),
            font,
            font_scale,
            (255,
             255,
             255),
            thickness)

        # Stack images in a grid: 2 rows x 3 columns
        row1 = np.hstack([original_resized, hue_color, saturation_color])
        row2 = np.hstack([brightness_color, black_colored, brown_colored])
        grid = np.vstack([row1, row2])

        # Create info panel
        info_panel = np.zeros(
            (self.INFO_HEIGHT, grid.shape[1], 3), dtype=np.uint8)

        # HSV Mean values
        track_text = f" (Track ID: {track_id})" if track_id else ""
        cv2.putText(
            info_panel, f"HSV Means: H={
                mean_h:.1f} S={
                mean_s:.1f} V={
                mean_v:.1f}{track_text}", (10, 25), font, 0.6, (255, 255, 255), 1)

        # Color detection result
        if black_pct >= self.th.black_percentage_threshold and black_pct > brown_pct:
            detected_color = "BLACK"
            color_indicator = (0, 0, 255)
        elif brown_pct >= self.th.brown_percentage_threshold and brown_pct > black_pct:
            detected_color = "BROWN"
            color_indicator = (0, 165, 255)
        else:
            detected_color = "UNKNOWN"
            color_indicator = (255, 255, 0)

        detection_text = f"COLOR DETECTED: {detected_color} (Black: {
            black_pct:.1f}% | Brown: {
            brown_pct:.1f}%)"
        cv2.putText(info_panel, detection_text, (10, 50),
                    font, 0.55, color_indicator, 1)

        # Assigned name
        if cat_name == NAME_BLACK_CAT:
            name_text = "ASSIGNED NAME: DALI (Black Cat)"
            name_color = (0, 0, 255)
        elif cat_name == NAME_OTHER_CAT:
            name_text = "ASSIGNED NAME: INDY (Brown Cat)"
            name_color = (0, 165, 255)
        else:
            name_text = "ASSIGNED NAME: UNKNOWN"
            name_color = (255, 255, 0)

        cv2.putText(info_panel, name_text, (10, 75), font, 0.55, name_color, 1)

        # Combine grid with info panel
        grid_with_info = np.vstack([grid, info_panel])
        self._grids[cat_name] = grid_with_info

    def show(self):
        if self._grids[NAME_BLACK_CAT] is not None:
            cv2.imshow(
                'HSV Debug - DALI (Black Cat)',
                self._grids[NAME_BLACK_CAT])
        if self._grids[NAME_OTHER_CAT] is not None:
            cv2.imshow(
                'HSV Debug - INDY (Other Cat)',
                self._grids[NAME_OTHER_CAT])

    def reset_thresholds(self):
        defaults = ColorThresholds()
        for attr in [
            'brown_hue_min',
            'brown_hue_max',
            'brown_saturation_min',
            'brown_saturation_max',
            'brown_brightness_min',
                'brown_brightness_max']:
            setattr(self.th, attr, getattr(defaults, attr))

        win = 'BROWN Threshold Adjustments (INDY)'
        cv2.setTrackbarPos('Hue Min', win, self.th.brown_hue_min)
        cv2.setTrackbarPos('Hue Max', win, self.th.brown_hue_max)
        cv2.setTrackbarPos('Sat Min', win, self.th.brown_saturation_min)
        cv2.setTrackbarPos('Sat Max', win, self.th.brown_saturation_max)
        cv2.setTrackbarPos('Bright Min', win, self.th.brown_brightness_min)
        cv2.setTrackbarPos('Bright Max', win, self.th.brown_brightness_max)
        print("🔄 BROWN thresholds reset to defaults")

    def print_thresholds(self):
        print("\n" + "=" * 50)
        print("📋 CURRENT BROWN THRESHOLD VALUES:")
        print("=" * 50)
        print(f"    BROWN_HUE_MIN = {self.th.brown_hue_min}")
        print(f"    BROWN_HUE_MAX = {self.th.brown_hue_max}")
        print(f"    BROWN_SATURATION_MIN = {self.th.brown_saturation_min}")
        print(f"    BROWN_SATURATION_MAX = {self.th.brown_saturation_max}")
        print(f"    BROWN_BRIGHTNESS_MIN = {self.th.brown_brightness_min}")
        print(f"    BROWN_BRIGHTNESS_MAX = {self.th.brown_brightness_max}")
        print("=" * 50 + "\n")


# =============================================================================
# BASE TRACKER (Abstract)
# =============================================================================

class BaseTracker(ABC):
    """Base tracker with common functionality."""

    def __init__(self, config: Config):
        self.config = config
        self.timing = TimingStats()

        # Load model
        model_file = os.path.join('models', config.model_path)
        if not os.path.exists(model_file):
            print(f"⚠️  Model not found: {model_file}")
            print(f"📥 Downloading model {config.model_path}...")
            os.makedirs('models', exist_ok=True)

        self.model = YOLO(model_file)

        # Setup device
        import torch
        if config.use_cuda and torch.cuda.is_available():
            self.device = 'cuda'
            self.model.to('cuda')
            print("🖥️  Device: CUDA")
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("🖥️  Device: CPU")
            if config.use_cuda and not torch.cuda.is_available():
                print("⚠️  CUDA requested but not available, using CPU")

        print(f"📦 Model: {config.model_path}")
        print(f"🎯 Minimum confidence: {config.confidence_threshold}")

        # State
        self.frame_count = 0
        self.total_detections = 0
        self.last_detections = None

    def _get_inference_kwargs(self) -> dict:
        kwargs = {
            'persist': True,
            'classes': [CAT_CLASS_ID],
            'verbose': False,
            'conf': self.config.confidence_threshold
        }
        if self.device == 'cuda':
            kwargs['half'] = True
        return kwargs

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1

        if (self.frame_count - 1) % (self.config.skip_frames + 1) == 0:
            with Timer(self.timing, 'YOLO Inference'):
                results = self.model.track(
                    frame, **self._get_inference_kwargs())
            self._process_detections(frame, results[0])
            self.last_detections = results[0]
        elif self.last_detections:
            self._process_detections(frame, self.last_detections)

        return frame

    @abstractmethod
    def _process_detections(self, frame: np.ndarray, detections):
        pass

    @abstractmethod
    def _get_window_name(self) -> str:
        pass

    def _setup_debug(self):
        pass

    def _handle_keys(self) -> bool:
        return (cv2.waitKey(1) & 0xFF) == ord('q')

    def _print_summary(self):
        pass

    def _print_progress(
            self,
            frames_processed: int,
            total_frames: int,
            start_time: float):
        """Print processing progress in original format."""
        elapsed = time.time() - start_time
        current_fps = frames_processed / elapsed
        timing_avgs = self.timing.get_averages_ms()

        # Move cursor to home position and clear screen
        print("\033[H\033[J", end='')

        print("=" * 60)
        print("🎬 REAL-TIME PROCESSING")
        print("=" * 60)

        if total_frames > 0:
            progress = (frames_processed / total_frames) * 100
            print(
                f"📊 Progress: {
                    progress:.1f}% ({frames_processed}/{total_frames} frames)")
        else:
            print(f"📊 Frames processed: {frames_processed}")

        print(f"🚀 FPS: {current_fps:.1f}")
        print(f"🐱 Total detections: {self.total_detections}")

        print("\n" + "-" * 60)
        print("⏱️  PROCESSING TIMES:")
        print("-" * 60)
        print(
            f"   Frame reading:     {
                timing_avgs.get(
                    'Frame Reading',
                    0):7.1f} ms")
        print(
            f"   YOLO Inference:    {
                timing_avgs.get(
                    'YOLO Inference',
                    0):7.1f} ms")
        if 'HSV Analysis' in timing_avgs:
            print(
                f"   HSV Analysis:      {
                    timing_avgs.get(
                        'HSV Analysis',
                        0):7.2f} ms")
        if 'Mask Processing' in timing_avgs:
            print(
                f"   Mask Processing:   {
                    timing_avgs.get(
                        'Mask Processing',
                        0):7.2f} ms")
        print(
            f"   Annotation:        {
                timing_avgs.get(
                    'Annotation',
                    0):7.2f} ms")
        print(f"   Display:           {timing_avgs.get('ImShow', 0):7.1f} ms")
        print("=" * 60)
        print("\nPress 'q' to stop processing", end='', flush=True)

    def run(self):
        """Main processing loop."""
        cfg = self.config

        # Open video
        cap = cv2.VideoCapture(cfg.video_source)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video: {cfg.video_source}")
            return

        # Get properties
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        w, h = cfg.processing_resolution or (orig_w, orig_h)

        # Print video info in original format
        if cfg.processing_resolution:
            print(f"📹 Original resolution: {orig_w}x{orig_h} @ {fps} FPS")
            print(f"🔄 Processing resolution: {w}x{h}")
        else:
            print(f"📹 Resolution: {w}x{h} @ {fps} FPS")

        if total_frames > 0:
            print(f"📊 Total frames: {total_frames}")

        # Writer
        writer = None
        if cfg.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(cfg.output_path, fourcc, fps, (w, h))
            print(f"💾 Saving to: {cfg.output_path}")

        # Display window
        if cfg.show_live:
            cv2.namedWindow(self._get_window_name(), cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._get_window_name(), w, h)

        self._setup_debug()

        print("\n🎬 Processing video... Press 'q' to quit\n")

        # Clear screen once at the beginning
        print("\033[2J\033[H", end='')

        start_time = time.time()
        frames_processed = 0

        try:
            while cap.isOpened():
                with Timer(self.timing, 'Frame Reading'):
                    ret, frame = cap.read()

                if not ret:
                    break

                if cfg.processing_resolution:
                    frame = cv2.resize(frame, (w, h))

                frames_processed += 1
                frame = self.process_frame(frame)

                # Progress every 30 frames
                if frames_processed % 30 == 0:
                    self._print_progress(
                        frames_processed, total_frames, start_time)

                if cfg.show_live:
                    with Timer(self.timing, 'ImShow'):
                        cv2.imshow(self._get_window_name(), frame)
                    if self._handle_keys():
                        print("\n\n⏹️  Processing interrupted by user")
                        break

                if writer:
                    writer.write(frame)

        except KeyboardInterrupt:
            print("\n\n⏹️  Processing interrupted by user")

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

            # Summary in original format
            elapsed = time.time() - start_time
            avg_fps = frames_processed / elapsed if elapsed > 0 else 0

            print("\n\n" + "=" * 60)
            print("✅ Processing complete!")
            print("=" * 60)
            print(f"⏱️  Total time: {elapsed:.2f}s")
            print(f"📊 Frames processed: {frames_processed}")
            print(f"🚀 Average FPS: {avg_fps:.2f}")
            print(f"🐱 Total cat detections: {self.total_detections}")

            self.timing.print_summary()
            self._print_summary()
            print("=" * 60)


# =============================================================================
# SIMPLE TRACKER (No identification)
# =============================================================================

class SimpleTracker(BaseTracker):
    """Simple tracking without cat identification."""

    def _get_window_name(self) -> str:
        return 'Cat Detector - Tracking Only'

    def _process_detections(self, frame: np.ndarray, detections):
        if detections.boxes is None or detections.boxes.id is None:
            return

        boxes = detections.boxes.xyxy.cpu().numpy()
        ids = detections.boxes.id.cpu().numpy().astype(int)
        confs = detections.boxes.conf.cpu().numpy()

        valid = confs >= self.config.confidence_threshold
        self.total_detections += np.sum(valid)

        with Timer(self.timing, 'Annotation'):
            for box, tid, conf in zip(boxes[valid], ids[valid], confs[valid]):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DEFAULT, 2)

                label = f"Cat (ID: {tid}) {conf:.2f}"
                (tw, th_text), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th_text - 10),
                              (x1 + tw + 5, y1), COLOR_DEFAULT, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)


# =============================================================================
# COLOR TRACKER (Bounding box analysis)
# =============================================================================

class ColorTracker(BaseTracker):
    """Tracker with color-based identification."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.analyzer = ColorAnalyzer(config.color_thresholds)
        self.identifier = CatIdentifier(
            config.color_thresholds.min_detections_required)
        self.debug = DebugDisplay(
            config.color_thresholds) if config.debug_mode else None

    def _get_window_name(self) -> str:
        return 'Cat Detector - DALI (Black) vs INDY (Other colors)'

    def _setup_debug(self):
        if self.debug:
            self.debug.create_windows()

    def _handle_keys(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return True
        if self.debug:
            if key in (ord('r'), ord('R')):
                self.debug.reset_thresholds()
            elif key in (ord('p'), ord('P')):
                self.debug.print_thresholds()
        return False

    def _process_detections(self, frame: np.ndarray, detections):
        if detections.boxes is None or detections.boxes.id is None:
            return

        boxes = detections.boxes.xyxy.cpu().numpy()
        ids = detections.boxes.id.cpu().numpy().astype(int)
        confs = detections.boxes.conf.cpu().numpy()

        valid = confs >= self.config.confidence_threshold
        self.total_detections += np.sum(valid)

        for box, tid, conf in zip(boxes[valid], ids[valid], confs[valid]):
            x1, y1, x2, y2 = map(int, box)
            region = frame[y1:y2, x1:x2]

            with Timer(self.timing, 'HSV Analysis'):
                color_type, _, black_pct, brown_pct, black_mask, brown_mask = self.analyzer.analyze(
                    region)

            cat_name = self.identifier.get_name(tid, color_type)
            color = COLOR_DALI if cat_name == NAME_BLACK_CAT else COLOR_INDY

            if self.debug:
                self.debug.update(
                    cat_name,
                    region,
                    black_mask,
                    brown_mask,
                    black_pct,
                    brown_pct,
                    tid)

            with Timer(self.timing, 'Annotation'):
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cat_name} (ID: {tid}) {conf:.2f}"
                (tw, th_text), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th_text - 10),
                              (x1 + tw + 5, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)

        if self.debug:
            self.debug.show()

    def _print_summary(self):
        self.identifier.print_summary()

        if self.config.debug_mode:
            th = self.config.color_thresholds
            defaults = ColorThresholds()
            values_changed = (
                th.brown_hue_min != defaults.brown_hue_min or
                th.brown_hue_max != defaults.brown_hue_max or
                th.brown_saturation_min != defaults.brown_saturation_min or
                th.brown_saturation_max != defaults.brown_saturation_max or
                th.brown_brightness_min != defaults.brown_brightness_min or
                th.brown_brightness_max != defaults.brown_brightness_max
            )
            if values_changed:
                print("\n🎚️  MODIFIED BROWN THRESHOLD VALUES:")
                print("-" * 60)
                print("   Copy these values to update the class constants:")
                print(f"    BROWN_HUE_MIN = {th.brown_hue_min}")
                print(f"    BROWN_HUE_MAX = {th.brown_hue_max}")
                print(f"    BROWN_SATURATION_MIN = {th.brown_saturation_min}")
                print(f"    BROWN_SATURATION_MAX = {th.brown_saturation_max}")
                print(f"    BROWN_BRIGHTNESS_MIN = {th.brown_brightness_min}")
                print(f"    BROWN_BRIGHTNESS_MAX = {th.brown_brightness_max}")
                print("-" * 60)


# =============================================================================
# SEGMENTATION TRACKER (Mask-based analysis)
# =============================================================================

class SegmentationTracker(ColorTracker):
    """Tracker using segmentation masks for precise identification."""

    def _get_window_name(self) -> str:
        return 'Cat Segmentation - DALI (Black) vs INDY (Other colors)'

    def _get_inference_kwargs(self) -> dict:
        kwargs = super()._get_inference_kwargs()
        kwargs['show'] = False
        return kwargs

    def _process_detections(self, frame: np.ndarray, detections):
        if detections.boxes is None or detections.boxes.id is None:
            return

        boxes = detections.boxes.xyxy.cpu().numpy()
        ids = detections.boxes.id.cpu().numpy().astype(int)
        confs = detections.boxes.conf.cpu().numpy()
        masks = detections.masks.data.cpu().numpy(
        ) if detections.masks is not None else None

        valid = confs >= self.config.confidence_threshold
        self.total_detections += np.sum(valid)

        boxes, ids, confs = boxes[valid], ids[valid], confs[valid]
        if masks is not None:
            masks = masks[valid]

        for i, (box, tid, conf) in enumerate(zip(boxes, ids, confs)):
            x1, y1, x2, y2 = map(int, box)

            # Get mask
            seg_mask = None
            if masks is not None and i < len(masks):
                with Timer(self.timing, 'Mask Processing'):
                    seg_mask = cv2.resize(
                        masks[i], (frame.shape[1], frame.shape[0]))
                    seg_mask = (seg_mask > 0.5).astype(np.uint8)

            # Analyze color
            with Timer(self.timing, 'HSV Analysis'):
                if seg_mask is not None:
                    color_type, _, black_pct, brown_pct, black_mask, brown_mask = self.analyzer.analyze(
                        frame, seg_mask)
                else:
                    region = frame[y1:y2, x1:x2]
                    color_type, _, black_pct, brown_pct, black_mask, brown_mask = self.analyzer.analyze(
                        region)

            cat_name = self.identifier.get_name(tid, color_type)
            color = COLOR_DALI if cat_name == NAME_BLACK_CAT else COLOR_INDY
            mask_color = MASK_COLOR_DALI if cat_name == NAME_BLACK_CAT else MASK_COLOR_INDY

            with Timer(self.timing, 'Annotation'):
                # Draw mask
                if seg_mask is not None:
                    overlay = np.zeros_like(frame)
                    overlay[seg_mask > 0] = mask_color
                    cv2.addWeighted(overlay, MASK_ALPHA, frame, 1, 0, frame)
                    contours, _ = cv2.findContours(
                        seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, mask_color, 2)

                if self.debug:
                    region = frame[y1:y2, x1:x2]
                    self.debug.update(
                        cat_name,
                        region,
                        black_mask,
                        brown_mask,
                        black_pct,
                        brown_pct,
                        tid)

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cat_name} (ID: {tid}) {conf:.2f}"
                (tw, th_text), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th_text - 10),
                              (x1 + tw + 5, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)

        if self.debug:
            self.debug.show()
