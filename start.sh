echo "Setting up Conda..."
conda env create -f environment.yaml || true
source /opt/conda/etc/profile.d/conda.sh
conda activate chatbotenv

echo "Starting server..."
pip uninstall -y pinecone-plugin-inference

gunicorn --bind 0.0.0.0:10000 wsgi:app