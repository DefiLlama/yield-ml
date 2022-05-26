#!/bin/bash
echo "Running ml_dashboard.py"
python /home/ec2-user/workspace/yield-ml-feature-drift-dashboard/ml_dashboard.py

echo "Copying html output to s3://llama-apy-prediction-prod/ml_dashboard/"
aws s3 cp /home/ec2-user/workspace/yield-ml-feature-drift-dashboard/ml_dashboard_output/drift_check.html s3://llama-apy-prediction-prod/ml_dashboard/
