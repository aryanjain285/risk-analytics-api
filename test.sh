#!/bin/bash

# Risk Analytics API Test Suite
# Tests all API endpoints with various scenarios and edge cases

set -e

# Configuration
BASE_URL="http://localhost:8000"
TIMEOUT=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Logging
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_TESTS++))
}

error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_TESTS++))
}

warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test helper function
run_test() {
    local test_name="$1"
    local endpoint="$2"
    local expected_status="$3"
    local description="$4"
    
    ((TOTAL_TESTS++))
    
    log "Testing: $test_name - $description"
    
    # Make the request with timeout
    local response
    local status_code
    local response_time
    
    start_time=$(date +%s%3N)
    
    if response=$(curl -s -w "\n%{http_code}" -m "$TIMEOUT" "$BASE_URL$endpoint" 2>/dev/null); then
        status_code=$(echo "$response" | tail -n1)
        response_body=$(echo "$response" | head -n -1)
        end_time=$(date +%s%3N)
        response_time=$((end_time - start_time))
        
        # Check if response is valid JSON (basic check without jq)
        if [[ "$status_code" == "200" && "$response_body" == *"{"* ]]; then
            # Basic JSON validation - just check if it starts with { and contains }
            if [[ "$response_body" != *"}"* ]]; then
                error "$test_name: Invalid JSON response (no closing brace)"
                echo "Response: $response_body"
                return 1
            fi
        fi
        
        # Check status code
        if [[ "$status_code" == "$expected_status" ]]; then
            success "$test_name (${response_time}ms) - Status: $status_code"
            
            # Display response for successful tests (truncated)
            if [[ ${#response_body} -gt 200 ]]; then
                echo "Response: ${response_body:0:200}..."
            else
                echo "Response: $response_body"
            fi
            
            return 0
        else
            error "$test_name: Expected status $expected_status, got $status_code"
            echo "Response: $response_body"
            return 1
        fi
    else
        error "$test_name: Request failed or timed out"
        return 1
    fi
}

# Performance test helper
perf_test() {
    local test_name="$1"
    local endpoint="$2"
    local max_response_time="$3"
    
    log "Performance test: $test_name (max ${max_response_time}ms)"
    
    start_time=$(date +%s%3N)
    
    if response=$(curl -s -w "\n%{http_code}" -m "$TIMEOUT" "$BASE_URL$endpoint" 2>/dev/null); then
        end_time=$(date +%s%3N)
        response_time=$((end_time - start_time))
        status_code=$(echo "$response" | tail -n1)
        
        if [[ "$status_code" == "200" && "$response_time" -le "$max_response_time" ]]; then
            success "$test_name: ${response_time}ms (within ${max_response_time}ms limit)"
        elif [[ "$status_code" == "200" ]]; then
            warning "$test_name: ${response_time}ms (exceeds ${max_response_time}ms limit)"
        else
            error "$test_name: Failed with status $status_code"
        fi
    else
        error "$test_name: Request failed"
    fi
}

# Wait for server to be ready
wait_for_server() {
    log "Waiting for server to be ready..."
    
    for i in {1..30}; do
        if curl -s -f "$BASE_URL/health" >/dev/null 2>&1; then
            success "Server is ready!"
            return 0
        fi
        warning "Attempt $i: Server not ready, waiting 2 seconds..."
        sleep 2
    done
    
    error "Server failed to start within 60 seconds"
    exit 1
}

# Print header
echo "=========================================="
echo "🦀 Risk Analytics API Test Suite"
echo "=========================================="
echo "Base URL: $BASE_URL"
echo "Timeout: ${TIMEOUT}s"
echo "=========================================="

# Wait for server
wait_for_server

echo
echo "🏥 HEALTH CHECK TESTS"
echo "=========================================="

run_test "health_check" "/health" "200" "Basic health check"

echo
echo "💰 PORTFOLIO PRICE TESTS"
echo "=========================================="

# Valid portfolio price tests
run_test "portfolio_price_valid" "/portfolio-price?portfolioId=P1&date=2025-07-19" "200" "Valid portfolio price request"
run_test "portfolio_price_another_date" "/portfolio-price?portfolioId=P1&date=2025-07-18" "200" "Portfolio price for another date"

# Invalid portfolio price tests
run_test "portfolio_price_not_found" "/portfolio-price?portfolioId=INVALID&date=2025-07-19" "404" "Non-existent portfolio ID"
run_test "portfolio_price_missing_params" "/portfolio-price?portfolioId=P1" "400" "Missing date parameter"
run_test "portfolio_price_invalid_date" "/portfolio-price?portfolioId=P1&date=invalid-date" "400" "Invalid date format"

echo
echo "📈 DAILY RETURN TESTS"
echo "=========================================="

# Valid daily return tests
run_test "daily_return_valid" "/daily-return?portfolioId=P1&date=2025-07-19" "200" "Valid daily return request"
run_test "daily_return_another_portfolio" "/daily-return?portfolioId=P2&date=2025-07-19" "200" "Daily return for another portfolio"

# Invalid daily return tests
run_test "daily_return_not_found" "/daily-return?portfolioId=INVALID&date=2025-07-19" "404" "Non-existent portfolio for daily return"
run_test "daily_return_missing_params" "/daily-return?portfolioId=P1" "400" "Missing date parameter for daily return"

echo
echo "📊 CUMULATIVE RETURN TESTS"
echo "=========================================="

# Valid cumulative return tests
run_test "cumulative_return_valid" "/cumulative-return?portfolioId=P1&startDate=2025-07-18&endDate=2025-07-19" "200" "Valid cumulative return request"
run_test "cumulative_return_longer_period" "/cumulative-return?portfolioId=P1&startDate=2025-07-17&endDate=2025-07-19" "200" "Cumulative return for longer period"

# Invalid cumulative return tests
run_test "cumulative_return_not_found" "/cumulative-return?portfolioId=INVALID&startDate=2025-07-18&endDate=2025-07-19" "404" "Non-existent portfolio for cumulative return"
run_test "cumulative_return_invalid_range" "/cumulative-return?portfolioId=P1&startDate=2025-07-19&endDate=2025-07-18" "400" "Invalid date range (end before start)"

echo
echo "🌊 VOLATILITY TESTS"
echo "=========================================="

# Valid volatility tests
run_test "volatility_valid" "/daily-volatility?portfolioId=P1&startDate=2025-07-17&endDate=2025-07-19" "200" "Valid volatility request"

# Invalid volatility tests
run_test "volatility_not_found" "/daily-volatility?portfolioId=INVALID&startDate=2025-07-17&endDate=2025-07-19" "404" "Non-existent portfolio for volatility"
run_test "volatility_insufficient_data" "/daily-volatility?portfolioId=P1&startDate=2025-07-19&endDate=2025-07-19" "404" "Insufficient data for volatility calculation"

echo
echo "🔗 CORRELATION TESTS"
echo "=========================================="

# Valid correlation tests
run_test "correlation_valid" "/correlation?portfolioId1=P1&portfolioId2=P2&startDate=2025-07-17&endDate=2025-07-19" "200" "Valid correlation request"

# Invalid correlation tests
run_test "correlation_not_found" "/correlation?portfolioId1=INVALID&portfolioId2=P2&startDate=2025-07-17&endDate=2025-07-19" "404" "Non-existent portfolio in correlation"
run_test "correlation_same_portfolio" "/correlation?portfolioId1=P1&portfolioId2=P1&startDate=2025-07-17&endDate=2025-07-19" "400" "Same portfolio correlation (should be invalid)"

echo
echo "🎯 TRACKING ERROR TESTS"
echo "=========================================="

# Valid tracking error tests
run_test "tracking_error_valid" "/tracking-error?portfolioId=P1&benchmarkId=1&startDate=2025-07-17&endDate=2025-07-19" "200" "Valid tracking error request"

# Invalid tracking error tests
run_test "tracking_error_not_found" "/tracking-error?portfolioId=INVALID&benchmarkId=1&startDate=2025-07-17&endDate=2025-07-19" "404" "Non-existent portfolio for tracking error"
run_test "tracking_error_invalid_benchmark" "/tracking-error?portfolioId=P1&benchmarkId=999&startDate=2025-07-17&endDate=2025-07-19" "404" "Non-existent benchmark"

echo
echo "📊 METRICS TESTS"
echo "=========================================="

run_test "metrics" "/metrics" "200" "Performance metrics endpoint"
run_test "performance_stats" "/stats" "200" "Performance statistics endpoint"

echo
echo "⚡ PERFORMANCE TESTS"
echo "=========================================="

perf_test "health_check_performance" "/health" 50
perf_test "portfolio_price_performance" "/portfolio-price?portfolioId=P1&date=2025-07-19" 100
perf_test "daily_return_performance" "/daily-return?portfolioId=P1&date=2025-07-19" 100

echo
echo "🔒 EDGE CASE TESTS"
echo "=========================================="

# Test various edge cases
run_test "future_date" "/portfolio-price?portfolioId=P1&date=2030-01-01" "404" "Future date handling"
run_test "very_old_date" "/portfolio-price?portfolioId=P1&date=1900-01-01" "404" "Very old date handling"
run_test "leap_year_date" "/portfolio-price?portfolioId=P1&date=2024-02-29" "404" "Leap year date handling"

# Test URL encoding
run_test "url_encoded_portfolio" "/portfolio-price?portfolioId=P%201&date=2025-07-19" "404" "URL encoded portfolio ID"

# Test very long portfolio ID
LONG_PORTFOLIO_ID=$(printf 'A%.0s' {1..100})
run_test "long_portfolio_id" "/portfolio-price?portfolioId=${LONG_PORTFOLIO_ID}&date=2025-07-19" "400" "Very long portfolio ID"

echo
echo "🧪 CACHE TESTS"
echo "=========================================="

# Test cache by making same request multiple times
log "Testing cache effectiveness with repeated requests..."

# First request (cache miss expected)
start_time=$(date +%s%3N)
curl -s "$BASE_URL/portfolio-price?portfolioId=P1&date=2025-07-19" >/dev/null
end_time=$(date +%s%3N)
first_request_time=$((end_time - start_time))

# Second request (cache hit expected)
start_time=$(date +%s%3N)
curl -s "$BASE_URL/portfolio-price?portfolioId=P1&date=2025-07-19" >/dev/null
end_time=$(date +%s%3N)
second_request_time=$((end_time - start_time))

if [[ $second_request_time -lt $first_request_time ]]; then
    success "Cache optimization: First request: ${first_request_time}ms, Second request: ${second_request_time}ms"
else
    warning "Cache may not be working optimally: First: ${first_request_time}ms, Second: ${second_request_time}ms"
fi

echo
echo "🔍 STRESS TESTS"
echo "=========================================="

# Simple concurrent test
log "Running concurrent request test (10 parallel requests)..."

concurrent_test() {
    for i in {1..10}; do
        curl -s "$BASE_URL/portfolio-price?portfolioId=P1&date=2025-07-19" >/dev/null &
    done
    wait
}

start_time=$(date +%s%3N)
concurrent_test
end_time=$(date +%s%3N)
total_time=$((end_time - start_time))

if [[ $total_time -lt 5000 ]]; then  # Less than 5 seconds for 10 requests
    success "Concurrent requests completed in ${total_time}ms"
else
    warning "Concurrent requests took ${total_time}ms (may indicate performance issues)"
fi

echo
echo "=========================================="
echo "📋 TEST SUMMARY"
echo "=========================================="
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo "Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
echo "=========================================="

if [[ $FAILED_TESTS -eq 0 ]]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED! 🦀${NC}"
    exit 0
else
    echo -e "${RED}❌ $FAILED_TESTS TESTS FAILED${NC}"
    exit 1
fi