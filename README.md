# Cloud Gaming Optimization with LSTM and Q-Learning

## Overview
This project implements a player-aware resource management system for cloud gaming using:
- **LSTM** for combat prediction (98.5% accuracy)
- **Q-Learning** for dynamic network slice optimization
- **3-Tier Architecture** (Edge-Fog-Cloud) for scalable deployment

## Project Structure
\\\
01_data/          - Datasets (raw, processed, synthetic)
02_src/           - Source code
03_models/        - Trained models
04_deployment/    - Deployment scripts and servers
05_tests/         - Test suites
06_results/       - Results and visualizations
07_docs/          - Documentation
08_scripts/       - Utility scripts
\\\

## Quick Start
1. Install dependencies: \pip install -r requirements.txt\
2. Run edge server: \python 04_deployment/01_servers/edge/edge_server.py\
3. Run cloud server: \python 04_deployment/01_servers/cloud/cloud_server.py\
4. Run tests: \python 05_tests/integration_tests/test_aws_complete.py\

## Key Results
- **Prediction Accuracy**: 98.5%
- **Latency Reduction**: 35% during combat
- **Cost Savings**: 90% vs always-premium allocatio
