#!/bin/bash
echo "=========================================="
echo "    Mac Standalone Compiler Kit"
echo "=========================================="
echo "Installing requirements..."
pip3 install -r requirements.txt
pip3 install pyinstaller

echo "Compiling Mac Application using PyInstaller..."
pyinstaller QLoRA_Trainer_Mac.spec

echo "Done! You can find the Mac Application in the 'dist' folder."
