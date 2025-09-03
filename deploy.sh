#!/bin/bash
# Complete deployment script for Challenge 1

set -e

# Configuration
EC2_HOST="2025-app-9-ssh-82fe44c8c97046c2.elb.ap-southeast-1.amazonaws.com"
EC2_USER="ec2-user"
KEY_FILE="ec2-key.pem"
APP_NAME="risk-analytics-api"

echo "🚀 Starting deployment to production EC2..."

# Step 1: Set correct permissions for key file
chmod 600 $KEY_FILE
echo "✅ Key file permissions set"

# Step 2: Build optimized release binary
echo "🔨 Building optimized release binary..."
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma" cargo build --release
echo "✅ Binary built with native optimizations"

# Step 3: Create deployment package
echo "📦 Creating deployment package..."
tar czf deployment.tar.gz \
    target/release/$APP_NAME \
    config.production.toml \
    README.md

echo "✅ Deployment package created"

# Step 4: Copy files to EC2
echo "📤 Copying files to EC2..."
scp -i $KEY_FILE -o StrictHostKeyChecking=no deployment.tar.gz $EC2_USER@$EC2_HOST:/home/ec2-user/
echo "✅ Files copied to EC2"

# Step 5: Deploy and configure on EC2
echo "🔧 Deploying and configuring on EC2..."
ssh -i $KEY_FILE -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST << 'ENDSSH'

# Extract deployment package
tar xzf deployment.tar.gz
chmod +x risk-analytics-api

# Kill any existing process
sudo pkill -f risk-analytics-api || true
sleep 2

# System optimizations for performance
echo "🚀 Applying system optimizations..."

# Network optimizations
sudo bash -c 'cat >> /etc/sysctl.conf << EOF
# Network optimizations for high-performance API
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 120
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.tcp_keepalive_probes = 3
EOF'

# Memory optimizations  
sudo bash -c 'cat >> /etc/sysctl.conf << EOF
# Memory optimizations for caching workload
vm.swappiness = 1
vm.dirty_ratio = 15  
vm.dirty_background_ratio = 5
vm.overcommit_memory = 1
EOF'

# Apply sysctl changes
sudo sysctl -p

# File descriptor limits
sudo bash -c 'cat >> /etc/security/limits.conf << EOF
* soft nofile 65535
* hard nofile 65535  
ec2-user soft nofile 65535
ec2-user hard nofile 65535
EOF'

# Set CPU governor to performance if available
sudo bash -c 'echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor' 2>/dev/null || true

# Create systemd service
sudo bash -c 'cat > /etc/systemd/system/risk-analytics-api.service << EOF
[Unit]
Description=Risk Analytics API
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user
ExecStart=/home/ec2-user/risk-analytics-api
Restart=always
RestartSec=1
Environment=RUST_LOG=info
Environment=CONFIG_FILE=config.production.toml
Environment=TOKIO_WORKER_THREADS=8
LimitNOFILE=65535
Nice=-10

[Install]  
WantedBy=multi-user.target
EOF'

# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl enable risk-analytics-api
sudo systemctl start risk-analytics-api

# Wait a moment and check status
sleep 5
sudo systemctl status risk-analytics-api

# Verify the service is listening on port 8000
ss -tuln | grep :8000

echo "✅ Service deployed and running!"
echo "🌐 API available at: http://2025-app-9-http-1646610974.ap-southeast-1.elb.amazonaws.com:8000"

ENDSSH

echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Testing endpoints:"
echo "curl http://2025-app-9-http-1646610974.ap-southeast-1.elb.amazonaws.com:8000/health"
echo "curl 'http://2025-app-9-http-1646610974.ap-southeast-1.elb.amazonaws.com:8000/portfolio-price?portfolioId=TEST&date=2024-01-01'"

# Clean up
rm deployment.tar.gz
echo "🧹 Cleaned up deployment artifacts"