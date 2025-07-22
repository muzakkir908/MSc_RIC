# Setup Guide

## Local Development
1. Clone repository
2. Create virtual environment: \`python -m venv venv\`
3. Activate: \`venv\Scripts\activate\`
4. Install: \`pip install -r requirements.txt\`

## AWS Deployment
See \`04_deployment/02_aws_scripts/\` for deployment scripts.

## Model Training
1. Data collection: \`02_src/01_data_collection/\`
2. LSTM training: \`02_src/03_model_training/lstm/\`
3. Q-learning: \`02_src/03_model_training/qlearning/\`