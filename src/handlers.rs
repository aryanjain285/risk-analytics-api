// src/handlers.rs - Nuclear-powered API handlers for massive datasets
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use chrono::NaiveDate;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use crate::{AppState, cache::RollingVolatility, calculations::SIMDFinancialCalculator, models::*};

/// Health check endpoint with comprehensive diagnostics
pub async fn health_check(
    State(state): State<Arc<AppState>>,
) -> Result<Json<HealthResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();

    // Test database connection
    match state.db.health_check().await {
        Ok(db_stats) => {
            let response_time = start.elapsed();

            Ok(Json(HealthResponse {
                status: "healthy".to_string(),
                message: "Nuclear-powered Rust Risk Analytics API - Ready for DOMINATION!"
                    .to_string(),
                response_time_ms: response_time.as_millis() as f64,
                database_status: "connected".to_string(),
                cache_status: format!(
                    "active - {:.1}% hit ratio",
                    state.cache.get_cache_stats().overall_hit_ratio * 100.0
                ),
                database_response_ms: db_stats.response_time_ms,
                active_connections: db_stats.active_connections,
                version: "1.0.0".to_string(),
            }))
        }
        Err(e) => {
            error!("Database health check failed: {}", e);
            Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "Database connection failed".to_string(),
                }),
            ))
        }
    }
}

/// Portfolio price endpoint - NUCLEAR OPTIMIZED
pub async fn nuclear_portfolio_price(
    Query(query): Query<PortfolioPriceQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<PortfolioPriceResponse>, (StatusCode, Json<ErrorResponse>)> {
    debug!(
        "🎯 Portfolio price request: {} on {}",
        query.portfolio_id, query.date
    );

    // L1 Cache: Sub-microsecond lookup
    if let Some(price) = state
        .cache
        .get_portfolio_price(&query.portfolio_id, query.date)
        .await
    {
        debug!("⚡ L1 cache hit for portfolio price");
        return Ok(Json(PortfolioPriceResponse {
            portfolio_id: query.portfolio_id,
            date: query.date.to_string(),
            price,
        }));
    }

    // Database query with error handling
    match state
        .db
        .get_portfolio_price(&query.portfolio_id, query.date)
        .await
    {
        Ok(Some(price)) => {
            // Cache the result for future requests
            state
                .cache
                .cache_portfolio_price(&query.portfolio_id, query.date, price);

            debug!("📊 Portfolio price retrieved from database: {}", price);
            Ok(Json(PortfolioPriceResponse {
                portfolio_id: query.portfolio_id,
                date: query.date.to_string(),
                price,
            }))
        }
        Ok(None) => {
            warn!(
                "Portfolio price not found: {} on {}",
                query.portfolio_id, query.date
            );
            Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "Portfolio price not found for given date".to_string(),
                }),
            ))
        }
        Err(e) => {
            error!("Database error in portfolio price query: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                }),
            ))
        }
    }
}

/// Daily return endpoint - LIGHTNING FAST
pub async fn nuclear_daily_return(
    Query(query): Query<DailyReturnQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<DailyReturnResponse>, (StatusCode, Json<ErrorResponse>)> {
    debug!(
        "📈 Daily return request: {} on {}",
        query.portfolio_id, query.date
    );

    // Try database with optimized query
    match state
        .db
        .get_daily_return(&query.portfolio_id, query.date)
        .await
    {
        Ok(Some(daily_return)) => {
            debug!("📊 Daily return calculated: {}", daily_return);
            Ok(Json(DailyReturnResponse {
                portfolio_id: query.portfolio_id,
                date: query.date.to_string(),
                daily_return,
            }))
        }
        Ok(None) => {
            warn!(
                "Daily return cannot be calculated: {} on {}",
                query.portfolio_id, query.date
            );
            Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "Daily return cannot be calculated for given date".to_string(),
                }),
            ))
        }
        Err(e) => {
            error!("Database error in daily return query: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                }),
            ))
        }
    }
}

/// Cumulative return endpoint - OPTIMIZED FOR LARGE DATE RANGES
pub async fn nuclear_cumulative_return(
    Query(query): Query<CumulativeReturnQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<CumulativeReturnResponse>, (StatusCode, Json<ErrorResponse>)> {
    debug!(
        "📊 Cumulative return request: {} from {} to {}",
        query.portfolio_id, query.start_date, query.end_date
    );

    // Check L2 cache for price series
    if let Some(price_series) = state
        .cache
        .get_price_series_l2(&query.portfolio_id, query.start_date, query.end_date)
        .await
    {
        let price_values: Vec<f64> = price_series.iter().map(|(_, price)| *price).collect();
        let cumulative_return = tokio::task::spawn_blocking(move || {
            SIMDFinancialCalculator::cumulative_return_from_prices(&price_values)
        }).await.map_err(|_| {
            error!("CPU task panicked during cumulative return calculation");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
                error: "Internal server error".to_string(),
            }))
        })?;
        debug!("⚡ L2 cache hit for price series");

        return Ok(Json(CumulativeReturnResponse {
            portfolio_id: query.portfolio_id,
            cumulative_return,
        }));
    }

    // Fetch from database
    match state
        .db
        .get_portfolio_price_series(&query.portfolio_id, query.start_date, query.end_date)
        .await
    {
        Ok(price_series) if price_series.len() >= 2 => {
            let price_values: Vec<f64> = price_series.iter().map(|(_, price)| *price).collect();
            let cumulative_return = tokio::task::spawn_blocking(move || {
                SIMDFinancialCalculator::cumulative_return_from_prices(&price_values)
            }).await.map_err(|_| {
                error!("CPU task panicked during cumulative return calculation");
                (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                }))
            })?;

            // Cache the price series for future use
            state
                .cache
                .cache_price_series_l2(
                    &query.portfolio_id,
                    query.start_date,
                    query.end_date,
                    price_series,
                )
                .await;

            debug!("📊 Cumulative return calculated: {}", cumulative_return);
            Ok(Json(CumulativeReturnResponse {
                portfolio_id: query.portfolio_id,
                cumulative_return,
            }))
        }
        Ok(_) => {
            warn!(
                "Insufficient data for cumulative return: {}",
                query.portfolio_id
            );
            Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "Insufficient data for cumulative return calculation".to_string(),
                }),
            ))
        }
        Err(e) => {
            error!("Database error in cumulative return query: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                }),
            ))
        }
    }
}

/// Daily volatility endpoint - SIMD + CACHING OPTIMIZED
pub async fn nuclear_daily_volatility(
    Query(query): Query<VolatilityQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<VolatilityResponse>, (StatusCode, Json<ErrorResponse>)> {
    debug!(
        "🌊 Volatility request: {} from {} to {}",
        query.portfolio_id, query.start_date, query.end_date
    );

    // Check cache for existing volatility calculation
    if let Some(volatility) = state
        .cache
        .get_or_compute_volatility(&query.portfolio_id, query.start_date, query.end_date)
        .await
    {
        debug!("⚡ Cache hit for volatility calculation");
        return Ok(Json(VolatilityResponse {
            portfolio_id: query.portfolio_id,
            volatility,
        }));
    }

    // Fetch price series (check L2 cache first)
    let price_series = if let Some(cached_series) = state
        .cache
        .get_price_series_l2(&query.portfolio_id, query.start_date, query.end_date)
        .await
    {
        cached_series
    } else {
        match state
            .db
            .get_portfolio_price_series(&query.portfolio_id, query.start_date, query.end_date)
            .await
        {
            Ok(series) => {
                // Cache for future use
                state
                    .cache
                    .cache_price_series_l2(
                        &query.portfolio_id,
                        query.start_date,
                        query.end_date,
                        series.clone(),
                    )
                    .await;
                series
            }
            Err(e) => {
                error!("Database error in volatility query: {}", e);
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Internal server error".to_string(),
                    }),
                ));
            }
        }
    };

    if price_series.len() < 2 {
        warn!("Insufficient data for volatility: {}", query.portfolio_id);
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Insufficient data for volatility calculation".to_string(),
            }),
        ));
    }

    // SIMD-optimized calculation offloaded to thread pool
    let prices: Vec<f64> = price_series.iter().map(|(_, price)| *price).collect();
    let (returns, volatility) = tokio::task::spawn_blocking(move || {
        let returns = SIMDFinancialCalculator::daily_returns_simd(&prices);
        let volatility = SIMDFinancialCalculator::volatility_optimized(&returns);
        (returns, volatility)
    }).await.map_err(|_| {
        error!("CPU task panicked during volatility calculation");
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: "Internal server error".to_string(),
        }))
    })?;

    // Create rolling volatility for future incremental updates
    let rolling_vol = RollingVolatility::new(&returns);
    state.cache.cache_volatility(
        &query.portfolio_id,
        query.start_date,
        query.end_date,
        rolling_vol,
    );

    debug!("📊 Volatility calculated: {}", volatility);
    Ok(Json(VolatilityResponse {
        portfolio_id: query.portfolio_id,
        volatility,
    }))
}

/// Correlation endpoint - PARALLEL DATA FETCHING + SIMD
pub async fn nuclear_correlation(
    Query(query): Query<CorrelationQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<CorrelationResponse>, (StatusCode, Json<ErrorResponse>)> {
    debug!(
        "🔗 Correlation request: {} vs {} from {} to {}",
        query.portfolio_id1, query.portfolio_id2, query.start_date, query.end_date
    );

    // Check cache first
    if let Some(correlation) = state
        .cache
        .get_correlation(
            &query.portfolio_id1,
            &query.portfolio_id2,
            query.start_date,
            query.end_date,
        )
        .await
    {
        debug!("⚡ Cache hit for correlation calculation");
        return Ok(Json(CorrelationResponse {
            portfolio_id1: query.portfolio_id1,
            portfolio_id2: query.portfolio_id2,
            correlation,
        }));
    }

    // Fetch aligned data with single optimized query
    match state
        .db
        .get_aligned_price_series_parallel(
            &query.portfolio_id1,
            &query.portfolio_id2,
            query.start_date,
            query.end_date,
        )
        .await
    {
        Ok((prices1, prices2)) if prices1.len() >= 2 && prices2.len() >= 2 => {
            // SIMD-optimized correlation calculation offloaded to thread pool
            let correlation = tokio::task::spawn_blocking(move || {
                let returns1 = SIMDFinancialCalculator::daily_returns_simd(&prices1);
                let returns2 = SIMDFinancialCalculator::daily_returns_simd(&prices2);
                SIMDFinancialCalculator::correlation_simd(&returns1, &returns2)
            }).await.map_err(|_| {
                error!("CPU task panicked during correlation calculation");
                (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                }))
            })?;

            // Cache the result
            state.cache.cache_correlation(
                &query.portfolio_id1,
                &query.portfolio_id2,
                query.start_date,
                query.end_date,
                correlation,
            );

            debug!("📊 Correlation calculated: {}", correlation);
            Ok(Json(CorrelationResponse {
                portfolio_id1: query.portfolio_id1,
                portfolio_id2: query.portfolio_id2,
                correlation,
            }))
        }
        Ok(_) => {
            warn!(
                "Insufficient overlapping data for correlation: {} vs {}",
                query.portfolio_id1, query.portfolio_id2
            );
            Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "Insufficient overlapping data for correlation calculation".to_string(),
                }),
            ))
        }
        Err(e) => {
            error!("Database error in correlation query: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                }),
            ))
        }
    }
}

/// Tracking error endpoint - OPTIMIZED FOR MASSIVE BENCHMARKS
pub async fn nuclear_tracking_error(
    Query(query): Query<TrackingErrorQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<TrackingErrorResponse>, (StatusCode, Json<ErrorResponse>)> {
    debug!(
        "🎯 Tracking error request: {} vs benchmark {} from {} to {}",
        query.portfolio_id, query.benchmark_id, query.start_date, query.end_date
    );

    // Fetch portfolio price series
    let portfolio_series = match state
        .db
        .get_portfolio_price_series(&query.portfolio_id, query.start_date, query.end_date)
        .await
    {
        Ok(series) if series.len() >= 2 => series,
        Ok(_) => {
            warn!("Insufficient portfolio data for tracking error: {}", query.portfolio_id);
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "Insufficient portfolio data for tracking error calculation".to_string(),
                }),
            ));
        }
        Err(e) => {
            error!("Database error fetching portfolio data: {}", e);
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                }),
            ));
        }
    };

    // Fetch benchmark returns
    let benchmark_returns = match state
        .db
        .get_benchmark_returns(&query.benchmark_id, query.start_date, query.end_date)
        .await
    {
        Ok(returns) if !returns.is_empty() => returns,
        Ok(_) => {
            warn!("Insufficient benchmark data: {}", query.benchmark_id);
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "Insufficient benchmark data for tracking error calculation".to_string(),
                }),
            ));
        }
        Err(e) => {
            error!("Database error fetching benchmark data: {}", e);
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                }),
            ));
        }
    };

    // Calculate portfolio returns and tracking error
    let tracking_error = tokio::task::spawn_blocking(move || {
        let portfolio_prices: Vec<f64> = portfolio_series.iter().map(|(_, price)| *price).collect();
        let portfolio_returns = SIMDFinancialCalculator::daily_returns_simd(&portfolio_prices);
        
        // Ensure we have matching return periods
        let min_len = portfolio_returns.len().min(benchmark_returns.len());
        if min_len == 0 {
            return 0.0;
        }
        
        SIMDFinancialCalculator::tracking_error_optimized(
            &portfolio_returns[..min_len],
            &benchmark_returns[..min_len],
        )
    }).await.map_err(|_| {
        error!("CPU task panicked during tracking error calculation");
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: "Internal server error".to_string(),
        }))
    })?;

    debug!("📊 Tracking error calculated: {}", tracking_error);
    Ok(Json(TrackingErrorResponse {
        portfolio_id: query.portfolio_id,
        benchmark_id: query.benchmark_id,
        tracking_error,
    }))
}

/// Performance metrics endpoint for monitoring
pub async fn get_metrics(State(state): State<Arc<AppState>>) -> Json<PerformanceMetrics> {
    let cache_stats = state.cache.get_cache_stats();
    let db_stats =
        state
            .db
            .get_database_stats()
            .await
            .unwrap_or_else(|_| crate::database::DatabaseStats {
                portfolio_count: 0,
                total_records: 0,
                min_date: None,
                max_date: None,
            });

    Json(PerformanceMetrics {
        cache_hit_ratio: cache_stats.overall_hit_ratio,
        l1_cache_entries: cache_stats.l1_portfolio_prices
            + cache_stats.l1_daily_returns
            + cache_stats.l1_volatilities
            + cache_stats.l1_correlations,
        l2_cache_entries: cache_stats.l2_price_series + cache_stats.l2_calculations,
        total_requests: cache_stats.total_requests,
        database_portfolio_count: db_stats.portfolio_count,
        database_total_records: db_stats.total_records,
        memory_usage_mb: get_memory_usage_mb(),
        uptime_seconds: get_uptime_seconds(),
    })
}

/// Performance statistics endpoint
pub async fn get_performance_stats(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let cache_stats = state.cache.get_cache_stats();

    Json(serde_json::json!({
        "performance_grade": get_performance_grade(cache_stats.overall_hit_ratio),
        "cache_efficiency": {
            "hit_ratio": format!("{:.2}%", cache_stats.overall_hit_ratio * 100.0),
            "l1_entries": cache_stats.l1_portfolio_prices + cache_stats.l1_daily_returns +
                         cache_stats.l1_volatilities + cache_stats.l1_correlations,
            "l2_entries": cache_stats.l2_price_series + cache_stats.l2_calculations,
            "total_requests": cache_stats.total_requests
        },
        "system_resources": {
            "memory_usage_mb": get_memory_usage_mb(),
            "cpu_usage_percent": get_cpu_usage_percent(),
            "uptime_seconds": get_uptime_seconds()
        },
        "api_status": "🦀 NUCLEAR PERFORMANCE MODE ACTIVATED 🚀"
    }))
}

// Utility functions for system monitoring
fn get_memory_usage_mb() -> f64 {
    // Simplified memory usage - in production, use system APIs
    let usage = std::process::Command::new("ps")
        .args(&["-o", "rss=", "-p"])
        .arg(std::process::id().to_string())
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .ok()?
                .trim()
                .parse::<f64>()
                .ok()
        })
        .unwrap_or(0.0);

    usage / 1024.0 // Convert KB to MB
}

fn get_cpu_usage_percent() -> f64 {
    // Simplified CPU usage - in production, use system monitoring
    0.0 // Placeholder
}

static START_TIME: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn get_uptime_seconds() -> f64 {
    let start_time = START_TIME.get_or_init(|| std::time::Instant::now());
    start_time.elapsed().as_secs_f64()
}

fn get_performance_grade(hit_ratio: f64) -> &'static str {
    match hit_ratio {
        r if r >= 0.95 => "A++ NUCLEAR PERFORMANCE 🦀🔥",
        r if r >= 0.90 => "A+ BLAZING FAST ⚡",
        r if r >= 0.80 => "A EXCELLENT 👍",
        r if r >= 0.70 => "B GOOD 👌",
        r if r >= 0.50 => "C NEEDS OPTIMIZATION ⚠️",
        _ => "D CRITICAL PERFORMANCE ISSUES 🚨",
    }
}

/// Error handling utilities
pub fn map_db_error(e: anyhow::Error) -> (StatusCode, Json<ErrorResponse>) {
    error!("Database error: {}", e);

    if e.to_string().contains("not found") || e.to_string().contains("no rows") {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Data not found".to_string(),
            }),
        )
    } else if e.to_string().contains("timeout") {
        (
            StatusCode::REQUEST_TIMEOUT,
            Json(ErrorResponse {
                error: "Request timeout - database query took too long".to_string(),
            }),
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Internal server error".to_string(),
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integration tests would go here
    // These would test the complete request-response cycle
}
