// src/models.rs - Production data models for all API endpoints
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use axum::{http::StatusCode, response::Json};

// === REQUEST MODELS ===

#[derive(Debug, Deserialize)]
pub struct PortfolioPriceQuery {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    pub date: NaiveDate,
}

#[derive(Debug, Deserialize)]
pub struct DailyReturnQuery {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    pub date: NaiveDate,
}

#[derive(Debug, Deserialize)]
pub struct CumulativeReturnQuery {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    #[serde(rename = "startDate")]
    pub start_date: NaiveDate,
    #[serde(rename = "endDate")]
    pub end_date: NaiveDate,
}

#[derive(Debug, Deserialize)]
pub struct VolatilityQuery {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    #[serde(rename = "startDate")]
    pub start_date: NaiveDate,
    #[serde(rename = "endDate")]
    pub end_date: NaiveDate,
}

#[derive(Debug, Deserialize)]
pub struct CorrelationQuery {
    #[serde(rename = "portfolioId1")]
    pub portfolio_id1: String,
    #[serde(rename = "portfolioId2")]
    pub portfolio_id2: String,
    #[serde(rename = "startDate")]
    pub start_date: NaiveDate,
    #[serde(rename = "endDate")]
    pub end_date: NaiveDate,
}

#[derive(Debug, Deserialize)]
pub struct TrackingErrorQuery {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    #[serde(rename = "benchmarkId")]
    pub benchmark_id: String,
    #[serde(rename = "startDate")]
    pub start_date: NaiveDate,
    #[serde(rename = "endDate")]
    pub end_date: NaiveDate,
}

// === RESPONSE MODELS ===

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
    pub daily_return: f64,
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
pub struct HealthResponse {
    pub status: String,
    pub message: String,
    pub response_time_ms: f64,
    pub database_status: String,
    pub cache_status: String,
    pub database_response_ms: f64,
    pub active_connections: u32,
    pub version: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Serialize)]
pub struct PerformanceMetrics {
    pub cache_hit_ratio: f64,
    pub l1_cache_entries: usize,
    pub l2_cache_entries: u64,
    pub total_requests: u64,
    pub database_portfolio_count: i64,
    pub database_total_records: i64,
    pub memory_usage_mb: f64,
    pub uptime_seconds: f64,
}

// === INTERNAL DATA MODELS ===

/// Portfolio price data point
#[derive(Debug, Clone)]
pub struct PricePoint {
    pub date: NaiveDate,
    pub price: f64,
}

/// Return calculation result
#[derive(Debug, Clone)]
pub struct ReturnCalculation {
    pub portfolio_id: String,
    pub date: NaiveDate,
    pub return_value: f64,
    pub price_current: f64,
    pub price_previous: f64,
}

/// Volatility calculation with metadata
#[derive(Debug, Clone)]
pub struct VolatilityCalculation {
    pub portfolio_id: String,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub volatility: f64,
    pub annualized: bool,
    pub observation_count: usize,
    pub calculation_method: String,
}

/// Correlation calculation with metadata
#[derive(Debug, Clone)]
pub struct CorrelationCalculation {
    pub portfolio_id1: String,
    pub portfolio_id2: String,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub correlation: f64,
    pub observation_count: usize,
    pub calculation_method: String,
}

/// Risk metrics summary
#[derive(Debug, Serialize)]
pub struct RiskMetrics {
    pub portfolio_id: String,
    pub date_range: (NaiveDate, NaiveDate),
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub value_at_risk_95: f64,
    pub value_at_risk_99: f64,
    pub expected_shortfall_95: f64,
    pub beta: Option<f64>,  // vs benchmark
    pub alpha: Option<f64>, // vs benchmark
}

// === VALIDATION MODELS ===

/// Validation constraints for input parameters
pub struct ValidationRules;

impl ValidationRules {
    pub fn validate_portfolio_id(portfolio_id: &str) -> Result<(), String> {
        if portfolio_id.is_empty() {
            return Err("Portfolio ID cannot be empty".to_string());
        }
        if portfolio_id.len() > 50 {
            return Err("Portfolio ID too long (max 50 characters)".to_string());
        }
        if !portfolio_id
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        {
            return Err(
                "Portfolio ID can only contain alphanumeric characters, underscores, and hyphens"
                    .to_string(),
            );
        }
        Ok(())
    }

    pub fn validate_date_range(start_date: NaiveDate, end_date: NaiveDate) -> Result<(), String> {
        if start_date >= end_date {
            return Err("Start date must be before end date".to_string());
        }

        let date_diff = end_date - start_date;
        if date_diff.num_days() > 3650 {
            // Max 10 years
            return Err("Date range too large (maximum 10 years)".to_string());
        }

        let today = chrono::Utc::now().naive_utc().date();
        if end_date > today {
            return Err("End date cannot be in the future".to_string());
        }

        Ok(())
    }

    pub fn validate_calculation_feasibility(
        data_points: usize,
        calculation_type: &str,
    ) -> Result<(), String> {
        let min_points = match calculation_type {
            "daily_return" => 2,
            "volatility" => 10,
            "correlation" => 10,
            "tracking_error" => 10,
            _ => 2,
        };

        if data_points < min_points {
            return Err(format!(
                "Insufficient data for {} calculation (need at least {} points, got {})",
                calculation_type, min_points, data_points
            ));
        }

        Ok(())
    }
}

// === ERROR TYPES ===

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("Calculation error: {0}")]
    Calculation(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl ApiError {
    pub fn to_status_code(&self) -> StatusCode {
        match self {
            ApiError::Database(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::Cache(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::Calculation(_) => StatusCode::BAD_REQUEST,
            ApiError::Validation(_) => StatusCode::BAD_REQUEST,
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn to_error_response(&self) -> ErrorResponse {
        ErrorResponse {
            error: self.to_string(),
        }
    }
}

// === BATCH OPERATION MODELS ===

#[derive(Debug, Deserialize)]
pub struct BatchPortfolioPriceQuery {
    pub requests: Vec<PortfolioPriceRequest>,
}

#[derive(Debug, Deserialize)]
pub struct PortfolioPriceRequest {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    pub date: NaiveDate,
}

#[derive(Debug, Serialize)]
pub struct BatchPortfolioPriceResponse {
    pub results: Vec<PortfolioPriceResult>,
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub processing_time_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct PortfolioPriceResult {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    pub date: String,
    pub price: Option<f64>,
    pub error: Option<String>,
}

// === STREAMING MODELS FOR MASSIVE DATASETS ===

#[derive(Debug, Deserialize)]
pub struct StreamingQuery {
    #[serde(rename = "portfolioId")]
    pub portfolio_id: String,
    #[serde(rename = "startDate")]
    pub start_date: NaiveDate,
    #[serde(rename = "endDate")]
    pub end_date: NaiveDate,
    pub chunk_size: Option<usize>, // For massive datasets
}

#[derive(Debug, Serialize)]
pub struct StreamingResponse {
    pub portfolio_id: String,
    pub total_chunks: usize,
    pub chunk_size: usize,
    pub total_data_points: usize,
    pub processing_time_ms: f64,
}

// === CONFIGURATION MODELS ===

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connection_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub l1_max_entries: usize,
    pub l2_max_entries: u64,
    pub ttl_seconds: u64,
    pub cleanup_interval_seconds: u64,
    pub memory_threshold_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enable_simd: bool,
    pub enable_parallel: bool,
    pub parallel_threshold: usize,
    pub chunk_size: usize,
    pub max_request_size: usize,
}

// === MONITORING MODELS ===

#[derive(Debug, Serialize)]
pub struct ApiMetrics {
    pub total_requests: u64,
    pub requests_per_second: f64,
    pub average_response_time_ms: f64,
    pub error_rate: f64,
    pub cache_hit_ratio: f64,
    pub database_connection_pool_usage: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

#[derive(Debug, Serialize)]
pub struct EndpointMetrics {
    pub endpoint: String,
    pub request_count: u64,
    pub average_response_time_ms: f64,
    pub min_response_time_ms: f64,
    pub max_response_time_ms: f64,
    pub error_count: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

// === UTILITY FUNCTIONS ===

impl PortfolioPriceQuery {
    pub fn cache_key(&self) -> String {
        format!("price:{}:{}", self.portfolio_id, self.date)
    }
}

impl DailyReturnQuery {
    pub fn cache_key(&self) -> String {
        format!("return:{}:{}", self.portfolio_id, self.date)
    }
}

impl VolatilityQuery {
    pub fn cache_key(&self) -> String {
        format!(
            "vol:{}:{}:{}",
            self.portfolio_id, self.start_date, self.end_date
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        crate::models::ValidationRules::validate_portfolio_id(&self.portfolio_id)?;
        crate::models::ValidationRules::validate_date_range(self.start_date, self.end_date)?;
        Ok(())
    }
}

impl CorrelationQuery {
    pub fn cache_key(&self) -> String {
        // Ensure consistent cache key ordering
        let (p1, p2) = if self.portfolio_id1 < self.portfolio_id2 {
            (&self.portfolio_id1, &self.portfolio_id2)
        } else {
            (&self.portfolio_id2, &self.portfolio_id1)
        };

        format!("corr:{}:{}:{}:{}", p1, p2, self.start_date, self.end_date)
    }

    pub fn validate(&self) -> Result<(), String> {
        crate::models::ValidationRules::validate_portfolio_id(&self.portfolio_id1)?;
        crate::models::ValidationRules::validate_portfolio_id(&self.portfolio_id2)?;
        crate::models::ValidationRules::validate_date_range(self.start_date, self.end_date)?;

        if self.portfolio_id1 == self.portfolio_id2 {
            return Err("Cannot calculate correlation between identical portfolios".to_string());
        }

        Ok(())
    }
}

impl TrackingErrorQuery {
    pub fn cache_key(&self) -> String {
        format!(
            "tracking:{}:{}:{}:{}",
            self.portfolio_id, self.benchmark_id, self.start_date, self.end_date
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        crate::models::ValidationRules::validate_portfolio_id(&self.portfolio_id)?;
        crate::models::ValidationRules::validate_portfolio_id(&self.benchmark_id)?;
        crate::models::ValidationRules::validate_date_range(self.start_date, self.end_date)?;
        Ok(())
    }
}

// === RESPONSE HELPERS ===

impl HealthResponse {
    pub fn healthy(
        response_time_ms: f64,
        db_response_ms: f64,
        active_connections: u32,
        cache_hit_ratio: f64,
    ) -> Self {
        Self {
            status: "healthy".to_string(),
            message: "🦀 Nuclear Rust API - MAXIMUM PERFORMANCE ENGAGED! 🚀".to_string(),
            response_time_ms,
            database_status: "connected".to_string(),
            cache_status: format!("optimal - {:.1}% hit ratio", cache_hit_ratio * 100.0),
            database_response_ms: db_response_ms,
            active_connections,
            version: "1.0.0-nuclear".to_string(),
        }
    }

    pub fn degraded(error_message: &str) -> Self {
        Self {
            status: "degraded".to_string(),
            message: format!("⚠️ API partially operational: {}", error_message),
            response_time_ms: 0.0,
            database_status: "unknown".to_string(),
            cache_status: "unknown".to_string(),
            database_response_ms: 0.0,
            active_connections: 0,
            version: "1.0.0-nuclear".to_string(),
        }
    }
}

impl ErrorResponse {
    pub fn new(error: &str) -> Self {
        Self {
            error: error.to_string(),
        }
    }

    pub fn validation_error(field: &str, message: &str) -> Self {
        Self {
            error: format!("Validation error for {}: {}", field, message),
        }
    }

    pub fn not_found(resource: &str) -> Self {
        Self {
            error: format!("{} not found", resource),
        }
    }

    pub fn internal_error() -> Self {
        Self {
            error: "Internal server error - please try again".to_string(),
        }
    }
}

// === TYPE ALIASES FOR CLARITY ===

pub type ApiResult<T> = Result<Json<T>, (StatusCode, Json<ErrorResponse>)>;
pub type PortfolioId = String;
pub type Price = f64;
pub type ReturnValue = f64;
pub type Volatility = f64;
pub type Correlation = f64;

// === CONSTANTS ===

pub mod constants {
    /// Trading days per year for annualization
    pub const TRADING_DAYS_PER_YEAR: f64 = 252.0;

    /// Maximum allowed date range for calculations (10 years)
    pub const MAX_DATE_RANGE_DAYS: i64 = 3650;

    /// Maximum portfolio ID length
    pub const MAX_PORTFOLIO_ID_LENGTH: usize = 50;

    /// Chunk size for processing massive datasets
    pub const DEFAULT_CHUNK_SIZE: usize = 10_000;

    /// Maximum number of data points for single calculation
    pub const MAX_DATA_POINTS: usize = 100_000;

    /// Cache TTL for different data types
    pub const PRICE_CACHE_TTL_SECONDS: u64 = 300; // 5 minutes
    pub const CALCULATION_CACHE_TTL_SECONDS: u64 = 600; // 10 minutes
    pub const CORRELATION_CACHE_TTL_SECONDS: u64 = 1800; // 30 minutes

    /// Performance thresholds
    pub const EXCELLENT_RESPONSE_TIME_MS: f64 = 1.0;
    pub const GOOD_RESPONSE_TIME_MS: f64 = 10.0;
    pub const ACCEPTABLE_RESPONSE_TIME_MS: f64 = 100.0;

    /// Memory management thresholds
    pub const MAX_CACHE_ENTRIES_L1: usize = 100_000;
    pub const MAX_CACHE_ENTRIES_L2: u64 = 50_000;
    pub const MEMORY_PRESSURE_THRESHOLD_MB: u64 = 500;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_id_validation() {
        assert!(ValidationRules::validate_portfolio_id("PORTFOLIO_001").is_ok());
        assert!(ValidationRules::validate_portfolio_id("BENCH-001").is_ok());
        assert!(ValidationRules::validate_portfolio_id("").is_err());
        assert!(ValidationRules::validate_portfolio_id("PORT@001").is_err());
    }

    #[test]
    fn test_date_range_validation() {
        let start = chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end = chrono::NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
        assert!(ValidationRules::validate_date_range(start, end).is_ok());

        // Test invalid range
        assert!(ValidationRules::validate_date_range(end, start).is_err());
    }

    #[test]
    fn test_cache_key_generation() {
        let query = PortfolioPriceQuery {
            portfolio_id: "TEST001".to_string(),
            date: chrono::NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
        };

        let cache_key = query.cache_key();
        assert_eq!(cache_key, "price:TEST001:2024-01-15");
    }

    #[test]
    fn test_correlation_cache_key_consistency() {
        let query1 = CorrelationQuery {
            portfolio_id1: "A".to_string(),
            portfolio_id2: "B".to_string(),
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 31).unwrap(),
        };

        let query2 = CorrelationQuery {
            portfolio_id1: "B".to_string(),
            portfolio_id2: "A".to_string(),
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 31).unwrap(),
        };

        // Cache keys should be identical regardless of order
        assert_eq!(query1.cache_key(), query2.cache_key());
    }
}
