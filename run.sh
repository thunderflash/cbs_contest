#!/bin/bash
set -e

echo "============================================================"
echo "  SAIS Inference Pipeline - Starting"
echo "============================================================"
echo ""

# Check input directory
if [ ! -d "/saisdata/50/eval/images" ]; then
    echo "❌ Error: /saisdata/50/eval/images directory not found"
    exit 1
fi

# Create output directory if not exists
if [ ! -d "/saisresult" ]; then
    echo "Creating /saisresult directory..."
    mkdir -p /saisresult
fi

# Check model files
echo "Checking model files..."
if [ ! -f "/app/models/last.pt" ]; then
    echo "⚠️  Warning: Detection model not found at /app/models/last.pt"
fi

if [ ! -f "/app/models/pure_swin_classifier.pt" ]; then
    echo "❌ Error: Recognition model not found at /app/models/pure_swin_classifier.pt"
    exit 1
fi

if [ ! -f "/app/models/ppmi.pkl" ]; then
    echo "❌ Error: PPMI file not found at /app/models/ppmi.pkl"
    exit 1
fi

if [ ! -f "/app/models/unified_clean_manifest.json" ]; then
    echo "❌ Error: characters file not found at /app/models/unified_clean_manifest.json"
    exit 1
fi

echo "✅ All required model files found"
echo ""

# Display system info
echo "System Information:"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Not available')"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep 'release' || echo 'Not available')"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED - CRITICAL ERROR!')"
echo ""

# Verify critical dependencies
echo "Verifying Python dependencies..."
python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ CRITICAL ERROR: PyTorch is not installed!"
    echo "The Docker build may have failed to install dependencies."
    echo "Please rebuild the Docker image and check for pip install errors."
    exit 1
fi
echo "✅ PyTorch is available"

python -c "import ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: ultralytics (YOLO) is not installed. Detection will be disabled."
else
    echo "✅ Ultralytics (YOLO) is available"
fi
echo ""

# Run inference
echo "Starting inference..."
echo ""
python /app/src/run_inference.py

# Check output
if [ ! -f "/saisresult/prediction.json" ]; then
    echo "❌ Error: prediction.json not generated"
    exit 1
fi

# Display results summary
echo ""
echo "============================================================"
echo "  ✅ Inference Complete!"
echo "============================================================"
echo ""
echo "Output file: /saisresult/prediction.json"
echo "File size: $(ls -lh /saisresult/prediction.json | awk '{print $5}')"
echo "Number of images: $(python -c "import json; data=json.load(open('/saisresult/prediction.json')); print(len(data))")"
echo ""
echo "Done!"
