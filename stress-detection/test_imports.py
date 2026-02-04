#!/usr/bin/env python
"""Test script to check if all required modules are available"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

modules_to_test = [
    'keras',
    'tensorflow', 
    'cv2',
    'numpy',
    'matplotlib'
]

for module in modules_to_test:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"[OK] {module}: {version}")
    except ImportError as e:
        print(f"[FAIL] {module}: {e}")

print("\nChecking specific imports...")
try:
    from keras.utils import img_to_array
    print("[OK] from keras.utils import img_to_array: Success")
except ImportError as e:
    print(f"[FAIL] from keras.utils import img_to_array: {e}")


print("\nAttempting to run main.py...")
