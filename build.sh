#!/usr/bin/env bash
# upgrade pip + build tools before installing deps
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
