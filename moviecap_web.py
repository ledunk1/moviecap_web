#!/usr/bin/env python3
"""
GUI Launcher for Auto Movie Recap Editor
Created by Trialota
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the GUI
from app import VideoEditorGUI

if __name__ == '__main__':
    print("🎬 Starting Auto Movie Recap Editor GUI...")
    print("Created by Trialota")
    print("-" * 40)
    
    gui = VideoEditorGUI()
    gui.run()