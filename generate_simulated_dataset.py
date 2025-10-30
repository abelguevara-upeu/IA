#!/usr/bin/env python3
"""
Top-level wrapper to run the real generator located at p2/resources/generate_simulated_dataset.py

This lets you run the script from the repository root using:
    python3 generate_simulated_dataset.py

It simply locates the script by path and executes it with runpy.run_path so
relative paths inside the original script resolve relative to its own file location.
"""
import runpy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(HERE, 'p2', 'resources', 'generate_simulated_dataset.py')

if not os.path.exists(script):
    print(f"ERROR: expected script not found at: {script}")
    sys.exit(2)

# Run the target script in its own namespace (so __file__ and relative resource paths work)
runpy.run_path(script, run_name='__main__')
