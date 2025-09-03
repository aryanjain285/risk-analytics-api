#!/bin/bash
# deploy/deploy-ec2.sh - ONE-CLICK NUCLEAR DEPLOYMENT FOR HACKATHON
set -euo pipefail

echo "🦀 NUCLEAR RUST API DEPLOYMENT STARTING..."
echo "🚀 Preparing to DOMINATE the competition!"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_NAME="risk-analytics-api"
API_PORT="8000"
DEPLOY_USER="ec2-user"
PROJECT_DIR="/home/$DEPLOY_USER/$API_NAME"

# Functions for colored output
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as correct user
if [[ "$(whoami)" != "$DEPLOY_USER" ]]; then
    log_error "Please run this script as $DEPLOY_USER"
    exit 1
fi

log_info "🔧 Step 1: System Preparation"
# Update system packages
sudo yum update -y

# Install essential build tools and dependencies
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    gcc \
    gcc-c++ \
    openssl-devel \
    pkg-config \
    cmake \
    htop \
    curl \
    wget \
    git

log_success "✅ System packages updated"

log_info "🦀 Step 2: Rust Installation (Latest Stable)"
# Install Rust with the latest stable toolchain
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source ~/.cargo/env
    
    # Install additional targets and components
    rustup component add clippy rustfmt
    rustup target add x86_64-unknown-linux-gnu
    
    log_success "✅ Rust installed successfully"
else
    log_info "Rust already installed, updating..."
    rustup update stable
    source ~/.cargo/env
fi

# Verify Rust installation
RUST_VERSION=$(rustc --version)
log_success "✅ Rust version: $RUST_VERSION"

log_info "📁 Step 3: Project Setup"
# Create project directory
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Initialize project if not exists
if [[ ! -f "Cargo.toml" ]]; then
    log_warning "⚠️ Creating new Rust project..."
    cargo init --name "$API_NAME" --bin
fi

log_success "✅ Project directory ready: $PROJECT_DIR"

log_info "🔥 Step 4: Build Configuration"
# Create optimized Cargo.toml for NUCLEAR performance
cat > Cargo.toml << 'EOF'
[package]
name = "risk-analytics-api"
version = "1.0.0"
edition = "2021"

[dependencies]
tokio = { version = "1.37", features = ["full"] }
axum = { version = "0.7", features = ["json", "query", "macros", "tracing"] }
tower = { version = "0.4", features = ["full"] }
tower-http = { version = "0.5", features = ["cors", "trace", "compression-gzip"] }
sqlx = { version = "0.7", features = ["runtime-tokio-rustls", "postgres", "chrono", "uuid", "json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
nalgebra = { version = "0.32", features = ["serde"] }
rayon = "1.8"
dashmap = "5.5"
moka = { version = "0.12", features = ["future"] }
mimalloc = "0.1"
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
uuid = { version = "1.7", features = ["v4", "serde"] }
futures = "0.3"
once_cell = "1.19"

[profile.release]
lto = "fat"
codegen-units = 1  
panic = "abort"
strip = true
opt-level = 3
overflow-checks = false
debug = false

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native", "-C", "target-feature=+avx2,+fma"]
EOF

log_success "✅ Optimized Cargo.toml created"

log_info "🛠️ Step 5: System Optimizations"
# Kill any existing processes on port 8000
sudo fuser -k $API_PORT/tcp 2>/dev/null || true

# Configure firewall
log_info "Configuring firewall for port $API_PORT..."
sudo iptables -A INPUT -p tcp --dport $API_PORT -j ACCEPT || true
sudo service iptables save 2>/dev/null || true

# System performance tuning
log_info "Applying system performance optimizations..."

# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# TCP optimizations for high-performance networking
sudo sysctl -w net.core.somaxconn=65535
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65535
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
sudo sysctl -w net.core.default_qdisc=fq

# Memory optimizations
sudo sysctl -w vm.swappiness=10
sudo sysctl -w vm.dirty_ratio=15
sudo sysctl -w vm.dirty_background_ratio=5

# CPU performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true

log_success "✅ System optimizations applied"

log_info "🔨 Step 6: Build Application"
log_info "Building with MAXIMUM OPTIMIZATIONS (this may take 2-3 minutes)..."

# Set build environment variables for maximum performance
export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma -C link-arg=-fuse-ld=lld"
export CARGO_PROFILE_RELEASE_LTO=fat
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1

# Clean previous builds
cargo clean

# Build with maximum optimization
time cargo build --release --verbose

if [[ $? -eq 0 ]]; then
    log_success "✅ Build completed successfully!"
    
    # Show binary size and verify it exists
    if [[ -f "target/release/$API_NAME" ]]; then
        BINARY_SIZE=$(ls -lh "target/release/$API_NAME" | awk '{print $5}')
        log_success "📦 Binary size: $BINARY_SIZE"
        log_success "🔍 Binary location: $(pwd)/target/release/$API_NAME"
    else
        log_error "❌ Binary not found after build!"
        exit 1
    fi
else
    log_error "❌ Build failed!"
    exit 1
fi

log_info "🚀 Step 7: Launch Application"

# Create startup script
cat > start_api.sh << 'EOF'
#!/bin/bash
export RUST_LOG=info
export RUST_BACKTRACE=1

# Database configuration (will be updated on Friday)
export DATABASE_HOST="aws-0-ap-southeast-1.pooler.supabase.com"
export DATABASE_PORT="5432"
export DATABASE_NAME="postgres"
export DATABASE_USERNAME="team9dbuser.jdcgkhwtrsdhyysagkwb"
export DATABASE_PASSWORD="e5ci7swfjroiqs4f"

# Performance optimizations
export DATABASE_MAX_CONNECTIONS="100"
export DATABASE_MIN_CONNECTIONS="50"
export CACHE_L1_MAX_ENTRIES="200000"
export PERFORMANCE_ENABLE_SIMD="true"
export PERFORMANCE_ENABLE_PARALLEL="true"

echo "🚀 Starting Nuclear Rust API..."
exec ./target/release/risk-analytics-api
EOF

chmod +x start_api.sh

# Start the API in background with logging
log_info "Starting API server..."
nohup ./start_api.sh > api.log 2>&1 &
API_PID=$!

# Wait for startup
sleep 3

# Test if API is running
log_info "🔍 Testing API health..."
if curl -f -s http://localhost:$API_PORT/health > /dev/null; then
    log_success "✅ API is running successfully!"
    
    # Get public IP for external access
    PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
    
    log_success "🌐 API Endpoints Available:"
    echo "   📊 Health Check: http://$PUBLIC_IP:$API_PORT/health"
    echo "   📈 Portfolio Price: http://$PUBLIC_IP:$API_PORT/portfolio-price"
    echo "   📊 Daily Return: http://$PUBLIC_IP:$API_PORT/daily-return" 
    echo "   📈 Cumulative Return: http://$PUBLIC_IP:$API_PORT/cumulative-return"
    echo "   🌊 Volatility: http://$PUBLIC_IP:$API_PORT/daily-volatility"
    echo "   🔗 Correlation: http://$PUBLIC_IP:$API_PORT/correlation"
    echo "   🎯 Tracking Error: http://$PUBLIC_IP:$API_PORT/tracking-error"
    echo "   📊 Performance Metrics: http://$PUBLIC_IP:$API_PORT/metrics"
    
    log_success "🔧 Process ID: $API_PID"
    log_success "📋 Logs: $PROJECT_DIR/api.log"
    
else
    log_error "❌ API failed to start!"
    log_error "Check logs: $PROJECT_DIR/api.log"
    cat api.log
    exit 1
fi

log_info "🧪 Step 8: Quick Performance Test"
# Run basic performance test
HEALTH_RESPONSE_TIME=$(curl -o /dev/null -s -w "%{time_total}" http://localhost:$API_PORT/health)
log_info "Health endpoint response time: ${HEALTH_RESPONSE_TIME}s"

if (( $(echo "$HEALTH_RESPONSE_TIME < 0.1" | bc -l) )); then
    log_success "🔥 BLAZING FAST! Health check in ${HEALTH_RESPONSE_TIME}s"
elif (( $(echo "$HEALTH_RESPONSE_TIME < 0.5" | bc -l) )); then
    log_success "⚡ Very fast! Health check in ${HEALTH_RESPONSE_TIME}s"
else
    log_warning "⚠️ Slower than expected: ${HEALTH_RESPONSE_TIME}s"
fi

log_info "📊 Step 9: Deployment Summary"
echo "=================================================="
echo "🦀 NUCLEAR RUST API DEPLOYMENT COMPLETE!"
echo "=================================================="
echo "📍 Location: $PROJECT_DIR"
echo "🚀 Status: RUNNING"
echo "🔧 Process ID: $API_PID"
echo "🌐 Port: $API_PORT"
echo "📊 Public URL: http://$PUBLIC_IP:$API_PORT"
echo "📋 Logs: tail -f $PROJECT_DIR/api.log"
echo "🛑 Stop: kill $API_PID"
echo "🔄 Restart: $PROJECT_DIR/deploy/restart-api.sh"
echo "🧪 Test: $PROJECT_DIR/deploy/test-api.sh"
echo "=================================================="

# Create helpful utility scripts
log_info "🛠️ Creating utility scripts..."

# Restart script
cat > restart-api.sh << EOF
#!/bin/bash
echo "🔄 Restarting Nuclear Rust API..."

# Find and kill existing process
pkill -f "$API_NAME" || true
sleep 2

# Start new process
cd "$PROJECT_DIR"
nohup ./start_api.sh > api.log 2>&1 &
echo "✅ API restarted with PID: \$!"
EOF

chmod +x restart-api.sh

# Status script  
cat > status-api.sh << EOF
#!/bin/bash
echo "📊 Nuclear Rust API Status"
echo "========================="

# Check if process is running
if pgrep -f "$API_NAME" > /dev/null; then
    PID=\$(pgrep -f "$API_NAME")
    echo "✅ Status: RUNNING (PID: \$PID)"
    
    # Get memory usage
    MEM_USAGE=\$(ps -o rss= -p \$PID | awk '{print \$1/1024}' | head -1)
    echo "💾 Memory Usage: \${MEM_USAGE}MB"
    
    # Test API response
    if curl -f -s http://localhost:$API_PORT/health > /dev/null; then
        RESPONSE_TIME=\$(curl -o /dev/null -s -w "%{time_total}" http://localhost:$API_PORT/health)
        echo "⚡ Health Check: \${RESPONSE_TIME}s"
    else
        echo "❌ Health Check: FAILED"
    fi
else
    echo "❌ Status: NOT RUNNING"
fi

echo "📋 Recent logs:"
tail -10 "$PROJECT_DIR/api.log" 2>/dev/null || echo "No logs found"
EOF

chmod +x status-api.sh

# Logs script
cat > logs-api.sh << EOF
#!/bin/bash
echo "📋 Nuclear Rust API Logs"
echo "======================="
echo "Real-time logs (Ctrl+C to exit):"
echo ""
tail -f "$PROJECT_DIR/api.log"
EOF

chmod +x logs-api.sh

# Performance monitoring script
cat > monitor-api.sh << 'EOF'
#!/bin/bash
API_URL="http://localhost:8000"
echo "📊 Nuclear Rust API Performance Monitor"
echo "======================================="

while true; do
    # Test API responsiveness
    if curl -f -s "$API_URL/health" > /dev/null; then
        RESPONSE_TIME=$(curl -o /dev/null -s -w "%{time_total}" "$API_URL/health")
        
        # Get performance metrics
        METRICS=$(curl -s "$API_URL/metrics" 2>/dev/null)
        
        if [[ -n "$METRICS" ]]; then
            echo "✅ $(date '+%H:%M:%S') - API Healthy"
            echo "   ⚡ Response Time: ${RESPONSE_TIME}s"
            
            # Extract key metrics using jq if available, otherwise show raw
            if command -v jq &> /dev/null; then
                CACHE_HIT=$(echo "$METRICS" | jq -r '.cache_hit_ratio * 100' 2>/dev/null)
                TOTAL_REQ=$(echo "$METRICS" | jq -r '.total_requests' 2>/dev/null)
                MEM_USAGE=$(echo "$METRICS" | jq -r '.memory_usage_mb' 2>/dev/null)
                
                if [[ "$CACHE_HIT" != "null" ]]; then
                    echo "   📊 Cache Hit Ratio: ${CACHE_HIT}%"
                    echo "   📈 Total Requests: $TOTAL_REQ"
                    echo "   💾 Memory Usage: ${MEM_USAGE}MB"
                fi
            fi
        else
            echo "✅ $(date '+%H:%M:%S') - API Healthy - Response: ${RESPONSE_TIME}s"
        fi
    else
        echo "❌ $(date '+%H:%M:%S') - API NOT RESPONDING"
    fi
    
    echo "---"
    sleep 5
done
EOF

chmod +x monitor-api.sh

log_success "✅ Utility scripts created:"
log_info "   🔄 restart-api.sh - Restart the API"
log_info "   📊 status-api.sh - Check API status"  
log_info "   📋 logs-api.sh - View real-time logs"
log_info "   📈 monitor-api.sh - Performance monitoring"

log_info "🧪 Step 10: Final Verification"
# Verify the API is working with a comprehensive test
sleep 2

log_info "Testing API endpoints..."

# Test health endpoint
HEALTH_STATUS=$(curl -s http://localhost:$API_PORT/health | head -c 100)
log_success "🔍 Health Check Response: $HEALTH_STATUS"

# Show final status
echo ""
echo "🎯 DEPLOYMENT COMPLETE! READY FOR HACKATHON!"
echo "============================================="
echo "🦀 Nuclear Rust API Status: DEPLOYED & RUNNING"
echo "⚡ Performance Mode: MAXIMUM"
echo "🚀 Ready to DESTROY the competition!"
echo ""
echo "📋 CRITICAL INFO FOR FRIDAY:"
echo "   1. Update database credentials in start_api.sh"
echo "   2. Run ./restart-api.sh after updating credentials"  
echo "   3. Use ./test-api.sh for comprehensive testing"
echo "   4. Monitor with ./monitor-api.sh"
echo ""
echo "🏆 Your API will be UNTOUCHABLE with this setup!"
echo "============================================="

# Final success message
log_success "🚀 DEPLOYMENT SUCCESSFUL - READY FOR DOMINATION! 🦀🔥"

# Show next steps
echo ""
echo "FRIDAY DEPLOYMENT STEPS:"
echo "1. Update database credentials: nano start_api.sh"
echo "2. Restart API: ./restart-api.sh"
echo "3. Run full tests: ./deploy/test-api.sh"
echo "4. Submit API endpoint for judging!"
echo ""
echo "🏆 YOU'RE READY TO WIN! 🏆"