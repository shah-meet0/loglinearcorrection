#!/bin/bash
# Setup script for GCP GPU simulation runner

echo "GCP GPU Simulation Setup"
echo "========================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed."
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get current project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)

# Prompt for project ID
echo -n "Enter your GCP Project ID (current: $CURRENT_PROJECT): "
read PROJECT_ID
PROJECT_ID=${PROJECT_ID:-$CURRENT_PROJECT}

# Set project
gcloud config set project $PROJECT_ID

# Prompt for bucket name
echo -n "Enter your GCS bucket name (or 'create' to create a new one): "
read BUCKET_NAME

if [ "$BUCKET_NAME" = "create" ]; then
    BUCKET_NAME="gpu-simulation-${PROJECT_ID}-$(date +%s)"
    echo "Creating bucket: gs://$BUCKET_NAME"
    gsutil mb -p $PROJECT_ID gs://$BUCKET_NAME
fi

# Check if bucket exists
if ! gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
    echo "Error: Bucket gs://$BUCKET_NAME does not exist."
    echo "Please create it or provide a valid bucket name."
    exit 1
fi

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable compute.googleapis.com
gcloud services enable storage-component.googleapis.com

# Check GPU quota
echo "Checking GPU quota in us-central1..."
gcloud compute project-info describe --project=$PROJECT_ID | grep -A5 "NVIDIA_T4_GPUS"

# Create scripts directory in bucket
gsutil mkdir -p gs://$BUCKET_NAME/scripts/

# Update the configuration in the Python script
echo "Updating configuration..."
sed -i.bak "s/'project_id': 'YOUR_PROJECT_ID'/'project_id': '$PROJECT_ID'/g" gcp_gpu_simulation_runner.py
sed -i.bak "s/'bucket_name': 'YOUR_BUCKET_NAME'/'bucket_name': '$BUCKET_NAME'/g" gcp_gpu_simulation_runner.py

echo ""
echo "Setup complete! Configuration:"
echo "- Project ID: $PROJECT_ID"
echo "- Bucket: gs://$BUCKET_NAME"
echo ""
echo "Next steps:"
echo "1. Ensure you have the following files in the current directory:"
echo "   - bias_variance_simulation.py (or the memory_optimized version)"
echo "   - loglinearcorrection.py (your DRE implementation)"
echo "   - Any other required modules"
echo ""
echo "2. Run the simulation with:"
echo "   python3 gcp_gpu_simulation_runner.py"
echo ""
echo "3. Monitor the instance in the GCP Console:"
echo "   https://console.cloud.google.com/compute/instances?project=$PROJECT_ID"
echo ""
echo "Note: This will incur charges for the GPU instance (~$0.35/hour for T4)"
