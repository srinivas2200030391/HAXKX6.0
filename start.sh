#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting server..."
gunicorn --bind 0.0.0.0:10000 wsgi:app
