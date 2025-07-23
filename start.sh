#!/bin/bash
conda activate chatbotenv
gunicorn --bind 0.0.0.0:10000 wsgi:app
