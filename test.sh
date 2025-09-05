#!/bin/bash

# Risk Analytics API Test Suite - Simple Direct Curl Tests
set -e

BASE_URL="http://localhost:8000"

echo "🦀 Testing Risk Analytics API Endpoints"
echo "=========================================="
echo "Base URL: $BASE_URL"
echo ""

echo "🏥 HEALTH CHECK"
echo "----------------------------------------"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/health"
echo ""

echo "💰 PORTFOLIO PRICE TESTS"
echo "----------------------------------------"
echo "Testing leaf portfolio (ID: 3):"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/portfolio-price?portfolioId=3&date=2024-07-29"
echo ""

echo "Testing summary portfolio (ID: 4):"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/portfolio-price?portfolioId=4&date=2024-07-29"
echo ""

echo "Testing another date:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/portfolio-price?portfolioId=3&date=2024-07-30"
echo ""

echo "Testing invalid portfolio (should return 404/500):"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/portfolio-price?portfolioId=999999&date=2024-07-29"
echo ""

echo "📈 DAILY RETURN TESTS"
echo "----------------------------------------"
echo "Testing daily return for portfolio 3:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/daily-return?portfolioId=3&date=2024-07-30"
echo ""

echo "Testing daily return for summary portfolio:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/daily-return?portfolioId=4&date=2024-07-30"
echo ""

echo "📊 CUMULATIVE RETURN TESTS"
echo "----------------------------------------"
echo "Testing cumulative return (2 days):"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/cumulative-return?portfolioId=3&startDate=2024-07-29&endDate=2024-07-30"
echo ""

echo "Testing cumulative return (5 days):"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/cumulative-return?portfolioId=3&startDate=2024-07-27&endDate=2024-07-31"
echo ""

echo "Testing cumulative return for summary portfolio:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/cumulative-return?portfolioId=4&startDate=2024-07-29&endDate=2024-07-30"
echo ""

echo "🌊 DAILY VOLATILITY TESTS"
echo "----------------------------------------"
echo "Testing volatility for portfolio 3:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/daily-volatility?portfolioId=3&startDate=2024-07-27&endDate=2024-07-31"
echo ""

echo "Testing volatility for summary portfolio:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/daily-volatility?portfolioId=4&startDate=2024-07-27&endDate=2024-07-31"
echo ""

echo "🔗 CORRELATION TESTS"
echo "----------------------------------------"
echo "Testing correlation between portfolios 3 and 5:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/correlation?portfolioId1=3&portfolioId2=5&startDate=2024-07-27&endDate=2024-07-31"
echo ""

echo "Testing correlation between summary and leaf:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/correlation?portfolioId1=4&portfolioId2=3&startDate=2024-07-27&endDate=2024-07-31"
echo ""

echo "🎯 TRACKING ERROR TESTS"
echo "----------------------------------------"
echo "Testing tracking error for portfolio 3 vs benchmark 1:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/tracking-error?portfolioId=3&benchmarkId=1&startDate=2024-07-27&endDate=2024-07-31"
echo ""

echo "Testing tracking error for summary portfolio:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/tracking-error?portfolioId=4&benchmarkId=1&startDate=2024-07-27&endDate=2024-07-31"
echo ""

echo "📊 MONITORING ENDPOINTS"
echo "----------------------------------------"
echo "Testing metrics endpoint:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/metrics"
echo ""

echo "Testing stats endpoint:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/stats"
echo ""

echo "⚡ PERFORMANCE TESTS"
echo "----------------------------------------"
echo "Testing cache effectiveness - First request:"
time curl -s "$BASE_URL/portfolio-price?portfolioId=3&date=2024-07-29" > /dev/null
echo ""

echo "Testing cache effectiveness - Second request (should be faster):"
time curl -s "$BASE_URL/portfolio-price?portfolioId=3&date=2024-07-29" > /dev/null
echo ""

echo "🔄 CONCURRENT TEST"
echo "----------------------------------------"
echo "Running 5 parallel requests to test concurrency..."
for i in {1..5}; do
  curl -s "$BASE_URL/portfolio-price?portfolioId=3&date=2024-07-29" > /dev/null &
done
wait
echo "All parallel requests completed!"
echo ""

echo "🚫 ERROR HANDLING TESTS"
echo "----------------------------------------"
echo "Testing missing parameter:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/portfolio-price?portfolioId=3"
echo ""

echo "Testing invalid date format:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/portfolio-price?portfolioId=3&date=invalid-date"
echo ""

echo "Testing invalid date range:"
curl -w "Response Time: %{time_total}s | Status: %{http_code}\n" "$BASE_URL/cumulative-return?portfolioId=3&startDate=2024-07-30&endDate=2024-07-29"
echo ""

echo "✅ ALL ENDPOINT TESTS COMPLETED!"
echo "=========================================="