#!/usr/bin/env python3
"""
Cat Detector - Main entry point.

Usage:
    python main.py                              # Interactive mode selection
    python main.py detection                    # Simple tracking
    python main.py identification               # Color-based identification
    python main.py identification-segmentation  # Segmentation-based identification
"""

import sys
from config import check_installation, show_menu
from tracker import SimpleTracker, ColorTracker, SegmentationTracker


TRACKERS = {
    'detection': SimpleTracker,
    'identification': ColorTracker,
    'identification-segmentation': SegmentationTracker,
}


def main(mode: str = None):
    """Run the cat detector."""

    if not check_installation():
        sys.exit(1)

    # Select mode if not provided
    if mode is None:
        print("\n🐱 SELECT MODE:")
        print("  1. Detection (tracking only)")
        print("  2. Identification (DALI vs INDY)")
        print("  3. Identification with Segmentation (precise masks)")

        choice = input("\nChoice (1-3) [2]: ").strip() or "2"
        mode = {'1': 'detection', '2': 'identification',
                '3': 'identification-segmentation'}.get(choice, 'color')

    # Get configuration
    config = show_menu(mode)

    # Create and run tracker
    tracker_class = TRACKERS.get(mode, ColorTracker)

    try:
        tracker = tracker_class(config)
        tracker.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Check for command line argument
    mode = sys.argv[1] if len(sys.argv) > 1 else None

    if mode and mode not in TRACKERS:
        print(f"❌ Unknown mode: {mode}")
        print(f"   Available: {', '.join(TRACKERS.keys())}")
        sys.exit(1)

    main(mode)
