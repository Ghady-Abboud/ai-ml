#!/bin/sh

python3 src/main.py

rm -rf $(find . -type d -name "__pycache__")
