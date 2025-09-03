// test_mock_data.rs - Mock data testing for Risk Analytics API
use reqwest;
use serde_json::{json, Value};
use std::collections::HashMap;
use tokio;

const API_BASE_URL: &str = "http://localhost:3000";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Starting Risk Analytics API Mock Data Tests");
    
    // Create HTTP client
    let client = reqwest::Client::new();
    
    // Test 1: Health Check
    println!("\n1️⃣ Testing Health Check...");
    let response = client
        .get(&format!("{}/health", API_BASE_URL))
        .send()
        .await?;
    
    println!("Status: {}", response.status());
    let health: Value = response.json().await?;
    println!("Response: {}", serde_json::to_string_pretty(&health)?);
    
    // Test 2: Portfolio Risk Analysis with Mock Data
    println!("\n2️⃣ Testing Portfolio Risk Analysis...");
    let mock_portfolio = json!({
        "portfolio_id": "test-portfolio-123",
        "positions": [
            {
                "symbol": "AAPL",
                "quantity": 100.0,
                "current_price": 150.0
            },
            {
                "symbol": "GOOGL", 
                "quantity": 50.0,
                "current_price": 2500.0
            },
            {
                "symbol": "MSFT",
                "quantity": 75.0,
                "current_price": 300.0
            }
        ],
        "risk_parameters": {
            "confidence_level": 0.95,
            "time_horizon_days": 252,
            "calculation_method": "monte_carlo"
        }
    });
    
    let response = client
        .post(&format!("{}/api/risk/portfolio", API_BASE_URL))
        .header("Content-Type", "application/json")
        .json(&mock_portfolio)
        .send()
        .await?;
    
    println!("Status: {}", response.status());
    if response.status().is_success() {
        let risk_analysis: Value = response.json().await?;
        println!("Portfolio Risk Analysis: {}", serde_json::to_string_pretty(&risk_analysis)?);
    } else {
        println!("Error: {}", response.text().await?);
    }
    
    // Test 3: Market Data Price History with Mock Data
    println!("\n3️⃣ Testing Market Data Retrieval...");
    let query_params = [
        ("symbols", "AAPL,GOOGL,MSFT"),
        ("start_date", "2024-01-01"),
        ("end_date", "2024-12-31"),
        ("interval", "daily"),
    ];
    
    let response = client
        .get(&format!("{}/api/market/prices", API_BASE_URL))
        .query(&query_params)
        .send()
        .await?;
    
    println!("Status: {}", response.status());
    if response.status().is_success() {
        let market_data: Value = response.json().await?;
        println!("Market Data: {}", serde_json::to_string_pretty(&market_data)?);
    } else {
        println!("Error: {}", response.text().await?);
    }
    
    // Test 4: VaR Calculation with Mock Historical Data
    println!("\n4️⃣ Testing VaR Calculation...");
    let mock_price_data = json!({
        "symbol": "AAPL",
        "prices": [
            145.0, 147.2, 148.5, 146.8, 149.1, 151.3, 149.7, 152.0, 150.5, 153.2,
            151.8, 154.1, 152.6, 155.0, 153.4, 156.2, 154.7, 157.3, 155.8, 158.1
        ],
        "position_value": 15000.0,
        "confidence_levels": [0.95, 0.99],
        "time_horizon": 1
    });
    
    let response = client
        .post(&format!("{}/api/risk/var", API_BASE_URL))
        .header("Content-Type", "application/json")
        .json(&mock_price_data)
        .send()
        .await?;
    
    println!("Status: {}", response.status());
    if response.status().is_success() {
        let var_result: Value = response.json().await?;
        println!("VaR Calculation: {}", serde_json::to_string_pretty(&var_result)?);
    } else {
        println!("Error: {}", response.text().await?);
    }
    
    // Test 5: Performance Monitoring
    println!("\n5️⃣ Testing Performance Monitoring...");
    let response = client
        .get(&format!("{}/api/monitoring/performance", API_BASE_URL))
        .send()
        .await?;
    
    println!("Status: {}", response.status());
    if response.status().is_success() {
        let performance: Value = response.json().await?;
        println!("Performance Metrics: {}", serde_json::to_string_pretty(&performance)?);
    } else {
        println!("Error: {}", response.text().await?);
    }
    
    // Test 6: Load Testing
    println!("\n6️⃣ Testing Load Performance...");
    let start_time = std::time::Instant::now();
    let mut handles = vec![];
    
    // Send 10 concurrent requests
    for i in 0..10 {
        let client = client.clone();
        let handle = tokio::spawn(async move {
            let response = client
                .get(&format!("{}/health", API_BASE_URL))
                .send()
                .await;
            (i, response)
        });
        handles.push(handle);
    }
    
    let mut successful_requests = 0;
    let mut failed_requests = 0;
    
    for handle in handles {
        match handle.await {
            Ok((i, Ok(response))) => {
                if response.status().is_success() {
                    successful_requests += 1;
                    println!("Request {} succeeded: {}", i, response.status());
                } else {
                    failed_requests += 1;
                    println!("Request {} failed: {}", i, response.status());
                }
            }
            Ok((i, Err(e))) => {
                failed_requests += 1;
                println!("Request {} error: {}", i, e);
            }
            Err(e) => {
                failed_requests += 1;
                println!("Task error: {}", e);
            }
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("\n📊 Load Test Results:");
    println!("Total time: {:?}", elapsed);
    println!("Successful requests: {}", successful_requests);
    println!("Failed requests: {}", failed_requests);
    println!("Requests per second: {:.2}", 10.0 / elapsed.as_secs_f64());
    
    println!("\n✅ Mock data testing completed!");
    Ok(())
}