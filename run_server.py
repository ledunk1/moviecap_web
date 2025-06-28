#!/usr/bin/env python3
"""
Server Launcher for Auto Movie Recap Editor
Created by Trialota
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the server
from app import app

if __name__ == '__main__':
    print("🎬 Starting Auto Movie Recap Editor Server...")
    print("Created by Trialota")
    print("-" * 40)
    
    app.run(host='0.0.0.0', port=5000, debug=True)