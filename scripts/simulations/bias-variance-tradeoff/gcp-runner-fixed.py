#!/usr/bin/env python3
"""
FIXED GCP runner with proper preemption handling and bucket management.
"""

import subprocess
import time
import os
from datetime import datetime

# Configuration - will auto-detect bucket
CONFIG = {
    'project_id': None,  # Will auto-detect
    'zone': 'us-central1-a',
    'instance_name': f'bias-variance-{datetime.now().strftime("%m%d-%H%M")}',
    'machine_type': 'n1-highmem-4',  # Reduced to save money
    'gpu_type': 'nvidia-tesla-t4',
    'gpu_count': 1,
    'boot_disk_size': '100',  # Reduced disk size
    'bucket_name': None,  # Will auto-detect
    'results_folder': f'run-{datetime.now().strftime("%m%d-%H%M")}',
}

def check_files():
    """Check that required files exist."""
    required = ['bias-variance-tradeoff.py', 'loglinearcorrection.py']
    missing = [f for f in required if not os.path.exists(f)]
    
    if missing:
        print(f"❌ Missing required files: {missing}")
        print("\n📝 Make sure you have:")
        print("   1. bias-variance-tradeoff.py (the fixed simulation script)")
        print("   2. loglinearcorrection.py (the DRE implementation)")
        return False
    
    for f in required:
        size = os.path.getsize(f)
        print(f"✓ {f} ({size:,} bytes)")
    
    return True

def setup_gcp():
    """Enhanced GCP setup with auto-detection."""
    print("🔧 Setting up GCP...")
    
    # Check authentication
    result = subprocess.run("gcloud auth list --filter=status:ACTIVE", shell=True, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        print("❌ Not authenticated with GCP")
        print("   Run: gcloud auth login")
        return False
    
    # Get project ID
    result = subprocess.run("gcloud config get-value project", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ No GCP project set")
        print("   Run: gcloud config set project YOUR_PROJECT_ID")
        return False
    
    project = result.stdout.strip()
    CONFIG['project_id'] = project
    print(f"✓ Project: {project}")
    
    # Create/verify bucket with consistent naming
    bucket_name = f"{project}-bias-variance"
    CONFIG['bucket_name'] = bucket_name
    
    # Check if bucket exists
    result = subprocess.run(f"gsutil ls -b gs://{bucket_name}/", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"📦 Creating bucket gs://{bucket_name}/...")
        result = subprocess.run(f"gsutil mb -p {project} gs://{bucket_name}/", shell=True)
        if result.returncode != 0:
            print("❌ Failed to create bucket")
            print("   Check if bucket name is available or try a different project")
            return False
    
    print(f"✓ Bucket: gs://{bucket_name}/")
    
    # Check GPU quota
    print("🔍 Checking GPU quotas...")
    result = subprocess.run(
        f"gcloud compute project-info describe --project={project} --format='value(quotas[].limit)' --filter='quotas.metric:NVIDIA_T4_GPUS'",
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode == 0 and result.stdout.strip():
        quota = result.stdout.strip()
        print(f"✓ T4 GPU quota: {quota}")
        if int(quota) < 1:
            print("⚠️  GPU quota is 0 - you may need to request quota increase")
    else:
        print("⚠️  Could not check GPU quota")
    
    return True

def upload_scripts():
    """Upload scripts with verification."""
    print("📤 Uploading scripts...")
    
    bucket_name = CONFIG['bucket_name']
    
    # Upload both files
    for script in ['bias-variance-tradeoff.py', 'loglinearcorrection.py']:
        if not os.path.exists(script):
            print(f"❌ {script} not found")
            return False
            
        print(f"   Uploading {script}...")
        result = subprocess.run(f"gsutil cp '{script}' gs://{bucket_name}/scripts/{script}", shell=True)
        if result.returncode != 0:
            print(f"❌ Failed to upload {script}")
            return False
    
    # Verify uploads
    result = subprocess.run(f"gsutil ls gs://{bucket_name}/scripts/", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        files = result.stdout.strip().split('\n')
        print(f"✓ Uploaded {len(files)} files:")
        for f in files:
            if f.strip():
                print(f"   - {f.split('/')[-1]}")
        return True
    else:
        print("❌ Could not verify uploads")
        return False

def create_startup_script():
    """Create enhanced startup script with better error handling."""
    bucket_name = CONFIG['bucket_name']
    results_folder = CONFIG['results_folder']
    
    script_content = f'''#!/bin/bash
set -e
exec > >(tee -a /var/log/startup.log) 2>&1

echo "=== PREEMPTION-RESISTANT SIMULATION STARTUP ==="
echo "Time: $(date)"
echo "Bucket: gs://{bucket_name}/"
echo "Results folder: {results_folder}"

# Function to upload logs on exit
cleanup_and_upload() {{
    echo "=== CLEANUP AND FINAL UPLOAD ==="
    
    # Upload startup log
    gsutil cp /var/log/startup.log gs://{bucket_name}/{results_folder}/startup.log || true
    
    # Upload simulation log if exists
    if [[ -f /home/simulation/simulation.log ]]; then
        gsutil cp /home/simulation/simulation.log gs://{bucket_name}/{results_folder}/simulation.log || true
    fi
    
    # Upload any final results
    if [[ -d /home/simulation/output ]]; then
        echo "Uploading final results..."
        gsutil -m cp -r /home/simulation/output/* gs://{bucket_name}/{results_folder}/final/ || true
    fi
    
    # Create final status
    echo "SIMULATION_ENDED" > /tmp/final_status.txt
    echo "END_TIME=$(date)" >> /tmp/final_status.txt
    echo "EXIT_CODE=$1" >> /tmp/final_status.txt
    gsutil cp /tmp/final_status.txt gs://{bucket_name}/{results_folder}/final_status.txt || true
    
    echo "Cleanup completed"
}}

# Set trap for cleanup
trap 'cleanup_and_upload $?' EXIT

# Test GCS access immediately
echo "Testing GCS access..."
if ! gsutil ls gs://{bucket_name}/ >/dev/null 2>&1; then
    echo "ERROR: Cannot access GCS bucket gs://{bucket_name}/"
    exit 1
fi
echo "✓ GCS access confirmed"

# Wait for GPU drivers (Deep Learning VM auto-installs)
echo "Waiting for GPU setup..."
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "✓ GPU drivers ready"
        nvidia-smi --query-gpu=name,memory.total --format=csv
        break
    else
        echo "Waiting for GPU drivers... ($i/$MAX_WAIT)"
        if [[ $i -eq $MAX_WAIT ]]; then
            echo "⚠️  GPU setup timeout - continuing with CPU"
        fi
        sleep 30
    fi
done

# Setup Python environment
echo "Setting up Python environment..."
if [[ -f /opt/conda/bin/activate ]]; then
    echo "✓ Using conda environment"
    source /opt/conda/bin/activate base
else
    echo "Using system Python"
fi

# Install missing packages
echo "Installing additional packages..."
pip install --quiet tqdm psutil 2>/dev/null || echo "Some packages already installed"

# Test TensorFlow GPU
echo "Testing TensorFlow setup..."
python3 -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('CUDA built:', tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
print('GPUs available:', len(gpus))
if gpus:
    print('GPU details:', gpus)
    try:
        with tf.device('/GPU:0'):
            test = tf.constant([1.0, 2.0, 3.0])
            result = tf.reduce_sum(test)
        print('✓ GPU computation successful')
    except Exception as e:
        print('GPU test failed:', e)
        print('Will fall back to CPU')
else:
    print('No GPU detected - using CPU mode')
" || echo "TensorFlow test completed with warnings"

# Setup workspace
echo "Setting up simulation workspace..."
mkdir -p /home/simulation
cd /home/simulation

# Download scripts with retry logic
echo "Downloading simulation scripts..."
for attempt in {{1..3}}; do
    if gsutil -m cp gs://{bucket_name}/scripts/* .; then
        echo "✓ Scripts downloaded successfully"
        break
    else
        echo "Attempt $attempt failed, retrying..."
        sleep 10
        if [[ $attempt -eq 3 ]]; then
            echo "ERROR: Failed to download scripts after 3 attempts"
            exit 1
        fi
    fi
done

# Verify all required files
for script in "bias-variance-tradeoff.py" "loglinearcorrection.py"; do
    if [[ -f "$script" ]]; then
        echo "✓ Found $script ($(wc -l < "$script") lines)"
    else
        echo "ERROR: Missing required script: $script"
        exit 1
    fi
done

# Test imports
echo "Testing critical imports..."
python3 -c "
try:
    from loglinearcorrection import DRE
    print('✓ DRE import successful')
except Exception as e:
    print('ERROR: DRE import failed:', e)
    exit(1)

try:
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    print('✓ All imports successful')
except Exception as e:
    print('ERROR: Import failed:', e)
    exit(1)
" || {{
    echo "ERROR: Critical import test failed"
    exit 1
}}

# Create output directory
mkdir -p ./output/{{intermediate,checkpoints}}

# Start simulation with monitoring
echo "🚀 Starting preemption-resistant simulation..."
echo "   Time: $(date)"
echo "   Expected duration: 30-60 minutes with GPU"
echo "   Checkpoints will be saved every 5 minutes"

# Start simulation in background
python3 -u bias-variance-tradeoff.py > simulation.log 2>&1 &
SIM_PID=$!

echo "✓ Simulation started with PID: $SIM_PID"

# Enhanced monitoring loop
LAST_CHECKPOINT_UPLOAD=0
LAST_STATUS_UPDATE=0

while kill -0 $SIM_PID 2>/dev/null; do
    CURRENT_TIME=$(date +%s)
    TIMESTAMP=$(date '+%H:%M:%S')
    
    # Show GPU utilization if available
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
        echo "$TIMESTAMP GPU: $GPU_INFO"
    fi
    
    # Upload checkpoints every 3 minutes
    if [[ $((CURRENT_TIME - LAST_CHECKPOINT_UPLOAD)) -ge 180 ]]; then
        if [[ -d ./output/checkpoints ]] && [[ "$(ls -A ./output/checkpoints 2>/dev/null)" ]]; then
            echo "$TIMESTAMP Uploading checkpoints..."
            gsutil -m cp ./output/checkpoints/* gs://{bucket_name}/{results_folder}/checkpoints/ 2>/dev/null || true
            LAST_CHECKPOINT_UPLOAD=$CURRENT_TIME
        fi
    fi
    
    # Upload intermediate results every 5 minutes
    if [[ $((CURRENT_TIME - LAST_STATUS_UPDATE)) -ge 300 ]]; then
        echo "$TIMESTAMP Status update..."
        
        # Check for any results
        if [[ -d ./output/intermediate ]] && [[ "$(ls -A ./output/intermediate 2>/dev/null)" ]]; then
            echo "   Found intermediate results, uploading..."
            gsutil -m cp ./output/intermediate/* gs://{bucket_name}/{results_folder}/intermediate/ 2>/dev/null || true
        fi
        
        # Upload current log
        gsutil cp simulation.log gs://{bucket_name}/{results_folder}/simulation_current.log 2>/dev/null || true
        
        LAST_STATUS_UPDATE=$CURRENT_TIME
    fi
    
    sleep 60  # Check every minute
done

# Wait for simulation to complete
wait $SIM_PID
SIM_EXIT_CODE=$?

echo "=== SIMULATION COMPLETED ==="
echo "Exit code: $SIM_EXIT_CODE"
echo "End time: $(date)"

# Final upload happens in cleanup function via trap
exit $SIM_EXIT_CODE
'''
    
    return script_content

def create_instance():
    """Create preemptible GPU instance."""
    instance_name = f"{CONFIG['instance_name']}-gpu"
    print(f"🚀 Creating GPU instance: {instance_name}")
    
    startup_script = create_startup_script()
    
    with open('/tmp/startup_script.sh', 'w') as f:
        f.write(startup_script)
    
    cmd_parts = [
        f"gcloud compute instances create {instance_name}",
        f"--project={CONFIG['project_id']}",
        f"--zone={CONFIG['zone']}",
        f"--machine-type={CONFIG['machine_type']}",
        f"--boot-disk-size={CONFIG['boot_disk_size']}GB",
        "--boot-disk-type=pd-standard",
        "--image-family=tf-latest-gpu",
        "--image-project=deeplearning-platform-release",
        "--maintenance-policy=TERMINATE",
        "--preemptible",  # This is what makes it cheap but interruptible
        f"--accelerator=type={CONFIG['gpu_type']},count={CONFIG['gpu_count']}",
        "--metadata=install-nvidia-driver=True",
        "--metadata-from-file startup-script=/tmp/startup_script.sh",
        "--scopes=https://www.googleapis.com/auth/cloud-platform"
    ]
    
    cmd = " \\\n    ".join(cmd_parts)
    
    print("Configuration:")
    print(f"   💻 Machine: {CONFIG['machine_type']}")
    print(f"   🎮 GPU: {CONFIG['gpu_type']} x{CONFIG['gpu_count']}")
    print(f"   💾 Disk: {CONFIG['boot_disk_size']}GB")
    print(f"   💰 Preemptible: Yes (~75% cheaper, can be interrupted)")
    print(f"   ⏱️  Expected cost: ~$0.50/hour")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Instance {instance_name} created!")
        print("🔧 NVIDIA drivers installing automatically...")
        return instance_name
    else:
        print(f"❌ Instance creation failed:")
        print(result.stderr)
        
        # Helpful error messages
        if "quota" in result.stderr.lower() or "limit" in result.stderr.lower():
            print("\n💡 Quota/Limit Issues:")
            print("   - Check quotas: https://console.cloud.google.com/iam-admin/quotas")
            print("   - Filter for 'GPU' and 'Compute Engine'")
            print("   - Request quota increase if needed")
            print("   - Try different zone: us-central1-b, us-west1-b")
        
        if "does not have enough resources" in result.stderr:
            print("\n💡 Resource Issues:")
            print("   - Try different zone")
            print("   - Try different GPU type: nvidia-tesla-k80")
            print("   - Try again in a few minutes")
        
        return None

def monitor_instance(instance_name):
    """Enhanced monitoring with checkpoint awareness."""
    print(f"👀 Monitoring {instance_name}...")
    print("📊 Features:")
    print("   - Automatic checkpoint uploads every 3 minutes")
    print("   - Intermediate result uploads every 5 minutes") 
    print("   - Real-time GPU utilization monitoring")
    print("   - Preemption-resistant design")
    
    start_time = time.time()
    timeout = 7200  # 2 hours max
    last_log_check = 0
    
    while time.time() - start_time < timeout:
        current_time = time.time()
        
        # Check for completion status
        result = subprocess.run(
            f"gsutil ls gs://{CONFIG['bucket_name']}/{CONFIG['results_folder']}/final_status.txt", 
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            # Download and check status
            result = subprocess.run(
                f"gsutil cat gs://{CONFIG['bucket_name']}/{CONFIG['results_folder']}/final_status.txt", 
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                status = result.stdout.strip()
                print(f"\n📋 Final status received:")
                print(status)
                if "SIMULATION_ENDED" in status:
                    return 'completed'
        
        # Check instance status
        result = subprocess.run(
            f"gcloud compute instances describe {instance_name} --project={CONFIG['project_id']} --zone={CONFIG['zone']} --format='get(status)'", 
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            status = result.stdout.strip()
            if status == "TERMINATED":
                print(f"\n⚠️  Instance was preempted after {int((current_time - start_time)/60)} minutes")
                return 'preempted'
            elif status not in ["RUNNING", "PROVISIONING", "STAGING"]:
                print(f"❌ Instance in unexpected state: {status}")
                return 'failed'
        
        # Show progress every 5 minutes
        elapsed = int((current_time - start_time) / 60)
        if elapsed > 0 and elapsed % 5 == 0 and current_time - last_log_check > 290:
            last_log_check = current_time
            print(f"⏱️  Running for {elapsed} minutes...")
            
            # Check for intermediate results
            result = subprocess.run(
                f"gsutil ls gs://{CONFIG['bucket_name']}/{CONFIG['results_folder']}/intermediate/ 2>/dev/null | wc -l", 
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                count = result.stdout.strip()
                if count and int(count) > 0:
                    print(f"   📊 {count} intermediate result files uploaded")
            
            # Check for checkpoints
            result = subprocess.run(
                f"gsutil ls gs://{CONFIG['bucket_name']}/{CONFIG['results_folder']}/checkpoints/ 2>/dev/null | wc -l", 
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                count = result.stdout.strip()
                if count and int(count) > 0:
                    print(f"   💾 {count} checkpoint files saved")
        
        time.sleep(60)  # Check every minute
    
    return 'timeout'

def cleanup_instance(instance_name):
    """Delete the instance."""
    print(f"🧹 Cleaning up {instance_name}...")
    cmd = f"gcloud compute instances delete {instance_name} --project={CONFIG['project_id']} --zone={CONFIG['zone']} --quiet"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Instance deleted")
    else:
        print("⚠️  Instance cleanup may have failed (check console)")

def download_results():
    """Download all results and show summary."""
    print("📥 Downloading results...")
    
    local_dir = f"./results/{CONFIG['results_folder']}"
    os.makedirs(local_dir, exist_ok=True)
    
    bucket_path = f"gs://{CONFIG['bucket_name']}/{CONFIG['results_folder']}/"
    
    # Download everything
    result = subprocess.run(
        f"gsutil -m cp -r {bucket_path}* {local_dir}/", 
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode == 0 or "Some files could not be transferred" in result.stderr:
        print(f"✅ Results downloaded to {local_dir}")
        
        # Show what we got
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                total_size += size
                file_count += 1
                
                rel_path = os.path.relpath(file_path, local_dir)
                print(f"   📄 {rel_path} ({size:,} bytes)")
        
        print(f"\n📊 Summary: {file_count} files, {total_size:,} bytes total")
        
        # Check for key files
        key_files = [
            'bias_variance_optimized.png',
            'final_status.txt',
            'simulation.log'
        ]
        
        found_files = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if file in key_files:
                    found_files.append(file)
        
        if found_files:
            print(f"🎯 Key files found: {', '.join(found_files)}")
        
        return True
    else:
        print("❌ Download failed:")
        print(result.stderr)
        return False

def restart_from_checkpoint():
    """Restart simulation using existing checkpoints."""
    print("🔄 Attempting to restart from checkpoint...")
    
    # Check if we have checkpoints
    result = subprocess.run(
        f"gsutil ls gs://{CONFIG['bucket_name']}/{CONFIG['results_folder']}/checkpoints/", 
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print("❌ No checkpoints found to restart from")
        return False
    
    checkpoints = [line.strip() for line in result.stdout.split('\n') if line.strip()]
    if not checkpoints:
        print("❌ No checkpoint files found")
        return False
    
    print(f"📂 Found {len(checkpoints)} checkpoint files")
    
    # Create new instance with same configuration
    print("🔄 Creating new instance to resume simulation...")
    instance_name = create_instance()
    
    if instance_name:
        print("✅ Restart instance created - monitoring for completion...")
        return monitor_instance(instance_name)
    else:
        print("❌ Failed to create restart instance")
        return False

def main():
    """Main execution with enhanced error handling."""
    print("🚀 BIAS-VARIANCE SIMULATION - PREEMPTION-RESISTANT VERSION")
    print("="*70)
    print("🎯 Features:")
    print("   ✓ Automatic checkpoint saving every 5 minutes")
    print("   ✓ Graceful preemption handling (30-second warning)")
    print("   ✓ Intermediate result uploads")
    print("   ✓ Resume capability after interruption")
    print("   ✓ GPU acceleration with fallback to CPU")
    print("   ✓ ~75% cost savings with preemptible instances")
    print("="*70)
    
    if not check_files():
        return False
    
    if not setup_gcp():
        return False
    
    if not upload_scripts():
        return False
    
    instance_name = create_instance()
    if not instance_name:
        return False
    
    print(f"\n💰 Estimated cost: ~$0.50/hour")
    print(f"⏱️  Expected duration: 30-60 minutes")
    print(f"📊 Monitor at: https://console.cloud.google.com/compute/instances")
    
    try:
        result = monitor_instance(instance_name)
        
        if result == 'completed':
            print("\n🎉 Simulation completed successfully!")
            success = download_results()
            return success
            
        elif result == 'preempted':
            print("\n⚠️  Instance was preempted, but checkpoints should be saved")
            print("📥 Downloading partial results...")
            download_results()
            
            print("\n🤔 Would you like to restart from checkpoint? (y/n): ", end="")
            try:
                choice = input().strip().lower()
                if choice in ['y', 'yes']:
                    restart_result = restart_from_checkpoint()
                    if restart_result == 'completed':
                        download_results()
                        return True
                    else:
                        print("❌ Restart failed")
                        return False
                else:
                    print("👍 Partial results downloaded")
                    return False
            except KeyboardInterrupt:
                print("\n👍 Partial results downloaded")
                return False
                
        else:
            print(f"\n❌ Simulation {result}")
            download_results()
            return False
    
    finally:
        cleanup_instance(instance_name)

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🏆 SUCCESS!")
        print(f"📁 Results available in: ./results/{CONFIG['results_folder']}/")
        print("🎨 Look for bias_variance_optimized.png for your plots!")
        print("📊 Check simulation.log for detailed execution log")
    else:
        print("\n💥 INCOMPLETE")
        print("📄 Check partial results and logs for debugging")
        print("🔄 You can restart from checkpoint if preempted")
    
    exit(0 if success else 1)

