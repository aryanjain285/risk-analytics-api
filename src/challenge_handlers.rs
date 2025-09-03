// src/challenge_handlers.rs - Exact API handlers for challenge specification
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info};

use crate::{AppState, calculations::SIMDFinancialCalculator};

// Request/Response models matching exact Swagger specification

#[derive(Debug, Deserialize)]
pub struct PortfolioPriceQuery {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    pub date: String, // Will parse to NaiveDate
}

#[derive(Debug, Deserialize)]
pub struct DailyReturnQuery {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    pub date: String,
}

#[derive(Debug, Deserialize)]
pub struct CumulativeReturnQuery {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    #[serde(rename = "startDate")]
    pub start_date: String,
    #[serde(rename = "endDate")]
    pub end_date: String,
}

#[derive(Debug, Deserialize)]
pub struct VolatilityQuery {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    #[serde(rename = "startDate")]
    pub start_date: String,
    #[serde(rename = "endDate")]
    pub end_date: String,
}

#[derive(Debug, Deserialize)]
pub struct CorrelationQuery {
    #[serde(rename = "portfolioId1")]
    pub portfolio_id1: String,
    #[serde(rename = "portfolioId2")]
    pub portfolio_id2: String,
    #[serde(rename = "startDate")]
    pub start_date: String,
    #[serde(rename = "endDate")]
    pub end_date: String,
}

#[derive(Debug, Deserialize)]
pub struct TrackingErrorQuery {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    #[serde(rename = "benchmarkId")]
    pub benchmark_id: String,
    #[serde(rename = "startDate")]
    pub start_date: String,
    #[serde(rename = "endDate")]
    pub end_date: String,
}

// Response models matching exact Swagger specification

#[derive(Debug, Serialize)]
pub struct PortfolioPriceResponse {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    pub date: String,
    pub price: f64,
}

#[derive(Debug, Serialize)]
pub struct DailyReturnResponse {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    pub date: String,
    #[serde(rename = "return")]
    pub return_value: f64,
}

#[derive(Debug, Serialize)]
pub struct CumulativeReturnResponse {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    #[serde(rename = "cumulativeReturn")]
    pub cumulative_return: f64,
}

#[derive(Debug, Serialize)]
pub struct VolatilityResponse {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    pub volatility: f64,
}

#[derive(Debug, Serialize)]
pub struct CorrelationResponse {
    #[serde(rename = "portfolioId1")]
    pub portfolio_id1: String,
    #[serde(rename = "portfolioId2")]
    pub portfolio_id2: String,
    pub correlation: f64,
}

#[derive(Debug, Serialize)]
pub struct TrackingErrorResponse {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    #[serde(rename = "benchmarkId")]
    pub benchmark_id: String,
    #[serde(rename = "trackingError")]
    pub tracking_error: f64,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
}

// Helper function to parse date strings
fn parse_date(date_str: &str) -> Result<NaiveDate, String> {
    NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
        .map_err(|_| format!("Invalid date format: {}. Expected YYYY-MM-DD", date_str))
}

// API Handlers - Exact paths from Swagger specification

/// GET /portfolio-price
pub async fn get_portfolio_price(
    Query(params): Query<PortfolioPriceQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<PortfolioPriceResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start_time = std::time::Instant::now();
    
    // Parse date
    let date = match parse_date(&params.date) {
        Ok(d) => d,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e,
                    code: "INVALID_DATE".to_string(),
                }),
            ));
        }
    };

    // Check cache first
    let cache_key = format!("portfolio_price_{}_{}", params.portfolio_id, date);
    if let Some(cached_price) = state.cache.get_l1_portfolio_price(&params.portfolio_id, date).await {
        debug!("Cache hit for portfolio price: {}", cache_key);
        return Ok(Json(PortfolioPriceResponse {
            portfolio_id: params.portfolio_id,
            date: params.date,
            price: cached_price,
        }));
    }

    // Fetch from database
    match state.db.get_portfolio_price(&params.portfolio_id, date).await {
        Ok(Some(price)) => {
            // Cache the result
            state.cache.cache_portfolio_price_async(&params.portfolio_id, date, price).await;
            
            let response_time = start_time.elapsed();
            info!(
                "Portfolio price fetched: {} on {} = {} ({}ms)",
                params.portfolio_id, params.date, price, response_time.as_millis()
            );

            Ok(Json(PortfolioPriceResponse {
                portfolio_id: params.portfolio_id,
                date: params.date,
                price,
            }))
        }
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("No price found for portfolio {} on {}", params.portfolio_id, params.date),
                code: "NOT_FOUND".to_string(),
            }),
        )),
        Err(e) => {
            error!("Database error fetching portfolio price: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                    code: "DATABASE_ERROR".to_string(),
                }),
            ))
        }
    }
}

/// GET /daily-return
pub async fn get_daily_return(
    Query(params): Query<DailyReturnQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<DailyReturnResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start_time = std::time::Instant::now();
    
    let date = match parse_date(&params.date) {
        Ok(d) => d,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e,
                    code: "INVALID_DATE".to_string(),
                }),
            ));
        }
    };

    // Check cache first
    let cache_key = format!("daily_return_{}_{}", params.portfolio_id, date);
    if let Some(cached_return) = state.cache.get_l1_daily_return(&params.portfolio_id, date).await {
        debug!("Cache hit for daily return: {}", cache_key);
        return Ok(Json(DailyReturnResponse {
            portfolio_id: params.portfolio_id,
            date: params.date,
            return_value: cached_return,
        }));
    }

    // Fetch from database
    match state.db.get_daily_return(&params.portfolio_id, date).await {
        Ok(Some(return_value)) => {
            // Cache the result
            state.cache.cache_daily_return(&params.portfolio_id, date, return_value).await;
            
            let response_time = start_time.elapsed();
            info!(
                "Daily return calculated: {} on {} = {:.4} ({}ms)",
                params.portfolio_id, params.date, return_value, response_time.as_millis()
            );

            Ok(Json(DailyReturnResponse {
                portfolio_id: params.portfolio_id,
                date: params.date,
                return_value,
            }))
        }
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("No return data found for portfolio {} on {}", params.portfolio_id, params.date),
                code: "NOT_FOUND".to_string(),
            }),
        )),
        Err(e) => {
            error!("Database error fetching daily return: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                    code: "DATABASE_ERROR".to_string(),
                }),
            ))
        }
    }
}

/// GET /cumulative-return
pub async fn get_cumulative_return(
    Query(params): Query<CumulativeReturnQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<CumulativeReturnResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start_time = std::time::Instant::now();
    
    let start_date = match parse_date(&params.start_date) {
        Ok(d) => d,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e,
                    code: "INVALID_START_DATE".to_string(),
                }),
            ));
        }
    };

    let end_date = match parse_date(&params.end_date) {
        Ok(d) => d,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e,
                    code: "INVALID_END_DATE".to_string(),
                }),
            ));
        }
    };

    if start_date >= end_date {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Start date must be before end date".to_string(),
                code: "INVALID_DATE_RANGE".to_string(),
            }),
        ));
    }

    // Check cache first
    let cache_key = format!("cumulative_return_{}_{}_{}", params.portfolio_id, start_date, end_date);
    if let Some(cumulative_return) = state.cache.get_l2_calculation(&cache_key).await {
        debug!("Cache hit for cumulative return: {}", cache_key);
        return Ok(Json(CumulativeReturnResponse {
            portfolio_id: params.portfolio_id,
            cumulative_return,
        }));
    }

    // Fetch price series from database
    match state.db.get_portfolio_price_series(&params.portfolio_id, start_date, end_date).await {
        Ok(price_series) => {
            if price_series.is_empty() {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: format!("No price data found for portfolio {} in date range", params.portfolio_id),
                        code: "NOT_FOUND".to_string(),
                    }),
                ));
            }

            // Calculate cumulative return
            let prices: Vec<f64> = price_series.iter().map(|(_, price)| *price).collect();
            let returns = SIMDFinancialCalculator::daily_returns_simd(&prices);
            
            // Cumulative return = (final_price / initial_price) - 1
            let cumulative_return = if prices.len() >= 2 {
                (prices[prices.len() - 1] / prices[0]) - 1.0
            } else {
                0.0
            };

            // Cache the result
            state.cache.cache_l2_calculation(cache_key, cumulative_return).await;
            
            let response_time = start_time.elapsed();
            info!(
                "Cumulative return calculated: {} ({} to {}) = {:.4} ({}ms)",
                params.portfolio_id, params.start_date, params.end_date, 
                cumulative_return, response_time.as_millis()
            );

            Ok(Json(CumulativeReturnResponse {
                portfolio_id: params.portfolio_id,
                cumulative_return,
            }))
        }
        Err(e) => {
            error!("Database error fetching price series: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                    code: "DATABASE_ERROR".to_string(),
                }),
            ))
        }
    }
}

/// GET /daily-volatility
pub async fn get_daily_volatility(
    Query(params): Query<VolatilityQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<VolatilityResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start_time = std::time::Instant::now();
    
    let start_date = match parse_date(&params.start_date) {
        Ok(d) => d,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e,
                    code: "INVALID_START_DATE".to_string(),
                }),
            ));
        }
    };

    let end_date = match parse_date(&params.end_date) {
        Ok(d) => d,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e,
                    code: "INVALID_END_DATE".to_string(),
                }),
            ));
        }
    };

    if start_date >= end_date {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Start date must be before end date".to_string(),
                code: "INVALID_DATE_RANGE".to_string(),
            }),
        ));
    }

    // Check cache first
    let cache_key = format!("volatility_{}_{}_{}", params.portfolio_id, start_date, end_date);
    if let Some(cached_volatility) = state.cache.get_l1_volatility(&cache_key).await {
        debug!("Cache hit for volatility: {}", cache_key);
        return Ok(Json(VolatilityResponse {
            portfolio_id: params.portfolio_id,
            volatility: cached_volatility,
        }));
    }

    // Fetch price series and calculate volatility
    match state.db.get_portfolio_price_series(&params.portfolio_id, start_date, end_date).await {
        Ok(price_series) => {
            if price_series.len() < 2 {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "Insufficient data for volatility calculation (need at least 2 data points)".to_string(),
                        code: "INSUFFICIENT_DATA".to_string(),
                    }),
                ));
            }

            let prices: Vec<f64> = price_series.iter().map(|(_, price)| *price).collect();
            let returns = SIMDFinancialCalculator::daily_returns_simd(&prices);
            let volatility = SIMDFinancialCalculator::volatility_optimized(&returns);

            // Cache the result
            state.cache.cache_volatility_keyed_async(&cache_key, volatility).await;
            
            let response_time = start_time.elapsed();
            info!(
                "Volatility calculated: {} ({} to {}) = {:.4} ({}ms)",
                params.portfolio_id, params.start_date, params.end_date, 
                volatility, response_time.as_millis()
            );

            Ok(Json(VolatilityResponse {
                portfolio_id: params.portfolio_id,
                volatility,
            }))
        }
        Err(e) => {
            error!("Database error fetching price series: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                    code: "DATABASE_ERROR".to_string(),
                }),
            ))
        }
    }
}

/// GET /correlation
pub async fn get_correlation(
    Query(params): Query<CorrelationQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<CorrelationResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start_time = std::time::Instant::now();
    
    let start_date = match parse_date(&params.start_date) {
        Ok(d) => d,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e,
                    code: "INVALID_START_DATE".to_string(),
                }),
            ));
        }
    };

    let end_date = match parse_date(&params.end_date) {
        Ok(d) => d,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e,
                    code: "INVALID_END_DATE".to_string(),
                }),
            ));
        }
    };

    if start_date >= end_date {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Start date must be before end date".to_string(),
                code: "INVALID_DATE_RANGE".to_string(),
            }),
        ));
    }

    // Check cache first
    let cache_key = format!("correlation_{}_{}_{}_{}", 
        params.portfolio_id1, params.portfolio_id2, start_date, end_date);
    if let Some(cached_correlation) = state.cache.get_l1_correlation(&cache_key).await {
        debug!("Cache hit for correlation: {}", cache_key);
        return Ok(Json(CorrelationResponse {
            portfolio_id1: params.portfolio_id1,
            portfolio_id2: params.portfolio_id2,
            correlation: cached_correlation,
        }));
    }

    // Fetch aligned price data for both portfolios
    match state.db.get_aligned_price_series_parallel(&params.portfolio_id1, &params.portfolio_id2, start_date, end_date).await {
        Ok((prices1, prices2)) => {
            if prices1.len() < 2 || prices2.len() < 2 {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "Insufficient data for correlation calculation (need at least 2 aligned data points)".to_string(),
                        code: "INSUFFICIENT_DATA".to_string(),
                    }),
                ));
            }

            let returns1 = SIMDFinancialCalculator::daily_returns_simd(&prices1);
            let returns2 = SIMDFinancialCalculator::daily_returns_simd(&prices2);
            let correlation = SIMDFinancialCalculator::correlation_simd(&returns1, &returns2);

            // Cache the result
            state.cache.cache_correlation_keyed_async(&cache_key, correlation).await;
            
            let response_time = start_time.elapsed();
            info!(
                "Correlation calculated: {} vs {} ({} to {}) = {:.4} ({}ms)",
                params.portfolio_id1, params.portfolio_id2, params.start_date, params.end_date, 
                correlation, response_time.as_millis()
            );

            Ok(Json(CorrelationResponse {
                portfolio_id1: params.portfolio_id1,
                portfolio_id2: params.portfolio_id2,
                correlation,
            }))
        }
        Err(e) => {
            error!("Database error fetching aligned portfolio data: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                    code: "DATABASE_ERROR".to_string(),
                }),
            ))
        }
    }
}

/// GET /tracking-error
pub async fn get_tracking_error(
    Query(params): Query<TrackingErrorQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<TrackingErrorResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start_time = std::time::Instant::now();
    
    let start_date = match parse_date(&params.start_date) {
        Ok(d) => d,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e,
                    code: "INVALID_START_DATE".to_string(),
                }),
            ));
        }
    };

    let end_date = match parse_date(&params.end_date) {
        Ok(d) => d,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e,
                    code: "INVALID_END_DATE".to_string(),
                }),
            ));
        }
    };

    if start_date >= end_date {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Start date must be before end date".to_string(),
                code: "INVALID_DATE_RANGE".to_string(),
            }),
        ));
    }

    // Check cache first  
    let cache_key = format!("tracking_error_{}_{}_{}_{}", 
        params.portfolio_id, params.benchmark_id, start_date, end_date);
    if let Some(tracking_error) = state.cache.get_l2_calculation(&cache_key).await {
        debug!("Cache hit for tracking error: {}", cache_key);
        return Ok(Json(TrackingErrorResponse {
            portfolio_id: params.portfolio_id,
            benchmark_id: params.benchmark_id,
            tracking_error,
        }));
    }

    // Fetch aligned data for portfolio and benchmark
    match state.db.get_aligned_price_series_parallel(&params.portfolio_id, &params.benchmark_id, start_date, end_date).await {
        Ok((portfolio_prices, benchmark_prices)) => {
            if portfolio_prices.len() < 2 || benchmark_prices.len() < 2 {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "Insufficient data for tracking error calculation".to_string(),
                        code: "INSUFFICIENT_DATA".to_string(),
                    }),
                ));
            }

            let portfolio_returns = SIMDFinancialCalculator::daily_returns_simd(&portfolio_prices);
            let benchmark_returns = SIMDFinancialCalculator::daily_returns_simd(&benchmark_prices);
            let tracking_error = SIMDFinancialCalculator::tracking_error_optimized(&portfolio_returns, &benchmark_returns);

            // Cache the result
            state.cache.cache_l2_calculation(cache_key, tracking_error).await;
            
            let response_time = start_time.elapsed();
            info!(
                "Tracking error calculated: {} vs {} ({} to {}) = {:.4} ({}ms)",
                params.portfolio_id, params.benchmark_id, params.start_date, params.end_date, 
                tracking_error, response_time.as_millis()
            );

            Ok(Json(TrackingErrorResponse {
                portfolio_id: params.portfolio_id,
                benchmark_id: params.benchmark_id,
                tracking_error,
            }))
        }
        Err(e) => {
            error!("Database error fetching portfolio vs benchmark data: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Internal server error".to_string(),
                    code: "DATABASE_ERROR".to_string(),
                }),
            ))
        }
    }
}