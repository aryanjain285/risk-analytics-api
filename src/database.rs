// src/database.rs - Updated for actual schema (holdings, prices, portfolios, instruments)
use crate::config::AppConfig;
use anyhow::{Context, Result};
use chrono::NaiveDate;
use sqlx::{
    postgres::{PgConnectOptions, PgPoolOptions, PgRow},
    PgPool, Row,
};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::{error, info, warn};

#[derive(Clone)]
pub struct Database {
    pool: PgPool,
    queries: Arc<DatabaseQueries>,
}

pub struct DatabaseQueries {
    pub portfolio_price: &'static str,
    pub portfolio_price_series: &'static str,
    pub daily_return_optimized: &'static str,
    pub aligned_portfolios: &'static str,
}

impl DatabaseQueries {
    pub fn new() -> Self {
        Self {
            // Calculate portfolio price by aggregating holdings * prices
            portfolio_price: r#"
                SELECT SUM(h.quantity * p.closing_price) as total_value
                FROM holdings h
                JOIN prices p ON h.symbol = p.symbol AND h.date = p.date
                WHERE h.portfolio_id = $1 AND h.date = $2
            "#,

            // Get portfolio price series over date range
            portfolio_price_series: r#"
                SELECT h.date, SUM(h.quantity * p.closing_price) as total_value
                FROM holdings h
                JOIN prices p ON h.symbol = p.symbol AND h.date = p.date
                WHERE h.portfolio_id = $1 AND h.date BETWEEN $2 AND $3
                GROUP BY h.date
                ORDER BY h.date
            "#,

            // Calculate daily return using portfolio values
            daily_return_optimized: r#"
                WITH portfolio_values AS (
                    SELECT h.date, SUM(h.quantity * p.closing_price) as total_value
                    FROM holdings h
                    JOIN prices p ON h.symbol = p.symbol AND h.date = p.date
                    WHERE h.portfolio_id = $1 AND h.date <= $2
                    GROUP BY h.date
                    ORDER BY h.date DESC
                    LIMIT 2
                ),
                value_with_lag AS (
                    SELECT 
                        date,
                        total_value,
                        LAG(total_value) OVER (ORDER BY date) as prev_value
                    FROM portfolio_values
                )
                SELECT 
                    CASE 
                        WHEN prev_value IS NOT NULL AND prev_value != 0 
                        THEN (total_value - prev_value) / prev_value 
                        ELSE NULL 
                    END as daily_return
                FROM value_with_lag 
                WHERE date = $2
            "#,

            // Get aligned portfolio data for correlation/tracking error
            aligned_portfolios: r#"
                WITH p1_values AS (
                    SELECT h.date, SUM(h.quantity * p.closing_price) as portfolio1_value
                    FROM holdings h
                    JOIN prices p ON h.symbol = p.symbol AND h.date = p.date
                    WHERE h.portfolio_id = $1 AND h.date BETWEEN $3 AND $4
                    GROUP BY h.date
                ),
                p2_values AS (
                    SELECT h.date, SUM(h.quantity * p.closing_price) as portfolio2_value
                    FROM holdings h
                    JOIN prices p ON h.symbol = p.symbol AND h.date = p.date
                    WHERE h.portfolio_id = $2 AND h.date BETWEEN $3 AND $4
                    GROUP BY h.date
                )
                SELECT p1.date, p1.portfolio1_value, p2.portfolio2_value
                FROM p1_values p1
                INNER JOIN p2_values p2 ON p1.date = p2.date
                ORDER BY p1.date
            "#,
        }
    }
}

impl Database {
    pub async fn new(config: &AppConfig) -> Result<Self> {
        info!("🗄️ Initializing database connection for actual schema...");

        let connect_options = PgConnectOptions::new()
            .host(&config.database.host)
            .port(config.database.port)
            .username(&config.database.username)
            .password(&config.database.password)
            .database(&config.database.database)
            .application_name("nuclear_risk_api")
            .statement_cache_capacity(5000)
            .options([
                ("tcp_nodelay", "true"),
                ("tcp_user_timeout", "5000"),
                ("statement_timeout", "30s"),
                ("idle_in_transaction_session_timeout", "10s"),
            ]);

        let pool = PgPoolOptions::new()
            .max_connections(80)
            .min_connections(40)
            .acquire_timeout(Duration::from_millis(100))
            .idle_timeout(Duration::from_secs(300))
            .max_lifetime(Duration::from_secs(1800))
            .test_before_acquire(false)
            .after_connect(|conn, _meta| {
                Box::pin(async move {
                    sqlx::query("SET enable_seqscan = off")
                        .execute(&mut *conn)
                        .await?;
                    sqlx::query("SET enable_hashjoin = on")
                        .execute(&mut *conn)
                        .await?;
                    sqlx::query("SET work_mem = '64MB'")
                        .execute(&mut *conn)
                        .await?;
                    Ok(())
                })
            })
            .connect_with(connect_options)
            .await
            .context("Failed to create database connection pool")?;

        // Test connection
        sqlx::query("SELECT 1")
            .execute(&pool)
            .await
            .context("Failed to test database connection")?;

        info!("📊 Database connection established successfully");

        Ok(Self {
            pool,
            queries: Arc::new(DatabaseQueries::new()),
        })
    }

    /// Get portfolio price by aggregating holdings * instrument prices
    pub async fn get_portfolio_price(
        &self,
        portfolio_id: &str,
        date: NaiveDate,
    ) -> Result<Option<f64>> {
        let result = sqlx::query_scalar(self.queries.portfolio_price)
            .bind(portfolio_id)
            .bind(date)
            .fetch_optional(&self.pool)
            .await
            .context("Failed to fetch portfolio price")?;

        Ok(result)
    }

    /// Get portfolio price series over date range
    pub async fn get_portfolio_price_series(
        &self,
        portfolio_id: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<Vec<(NaiveDate, f64)>> {
        use futures::TryStreamExt;

        let mut rows = sqlx::query(self.queries.portfolio_price_series)
            .bind(portfolio_id)
            .bind(start_date)
            .bind(end_date)
            .fetch(&self.pool);

        let mut price_series = Vec::new();

        while let Some(row) = rows
            .try_next()
            .await
            .context("Failed to fetch portfolio price series")?
        {
            let date: NaiveDate = row.get("date");
            let total_value: Option<f64> = row.get("total_value");

            if let Some(value) = total_value {
                price_series.push((date, value));
            }
        }

        Ok(price_series)
    }

    /// Get daily return using portfolio valuation
    pub async fn get_daily_return(
        &self,
        portfolio_id: &str,
        date: NaiveDate,
    ) -> Result<Option<f64>> {
        let result = sqlx::query_scalar(self.queries.daily_return_optimized)
            .bind(portfolio_id)
            .bind(date)
            .fetch_optional(&self.pool)
            .await
            .context("Failed to calculate daily return")?;

        Ok(result.flatten())
    }

    /// Get aligned portfolio data for correlation and tracking error calculations
    pub async fn get_aligned_price_series_parallel(
        &self,
        portfolio_id1: &str,
        portfolio_id2: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        use futures::TryStreamExt;

        let mut rows = sqlx::query(self.queries.aligned_portfolios)
            .bind(portfolio_id1)
            .bind(portfolio_id2)
            .bind(start_date)
            .bind(end_date)
            .fetch(&self.pool);

        let mut prices1 = Vec::new();
        let mut prices2 = Vec::new();

        while let Some(row) = rows
            .try_next()
            .await
            .context("Failed to fetch aligned portfolio data")?
        {
            let p1_value: Option<f64> = row.get("portfolio1_value");
            let p2_value: Option<f64> = row.get("portfolio2_value");

            if let (Some(v1), Some(v2)) = (p1_value, p2_value) {
                prices1.push(v1);
                prices2.push(v2);
            }
        }

        Ok((prices1, prices2))
    }

    /// Get database statistics
    pub async fn get_database_stats(&self) -> Result<DatabaseStats> {
        let portfolio_count: (i64,) =
            sqlx::query_as("SELECT COUNT(DISTINCT portfolio_id) FROM portfolios")
                .fetch_one(&self.pool)
                .await
                .context("Failed to get portfolio count")?;

        let total_holdings: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM holdings")
            .fetch_one(&self.pool)
            .await
            .context("Failed to get total holdings")?;

        let date_range: (Option<NaiveDate>, Option<NaiveDate>) =
            sqlx::query_as("SELECT MIN(date), MAX(date) FROM holdings")
                .fetch_one(&self.pool)
                .await
                .context("Failed to get date range")?;

        Ok(DatabaseStats {
            portfolio_count: portfolio_count.0,
            total_records: total_holdings.0,
            min_date: date_range.0,
            max_date: date_range.1,
        })
    }

    /// Health check with performance metrics
    pub async fn health_check(&self) -> Result<HealthStats> {
        let start = std::time::Instant::now();

        let _: (i64,) = sqlx::query_as("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .context("Database health check failed")?;

        let response_time = start.elapsed();
        let pool_size = self.pool.size();
        let idle_connections = self.pool.num_idle();

        Ok(HealthStats {
            response_time_ms: response_time.as_millis() as f64,
            pool_size,
            idle_connections: idle_connections as u32,
            active_connections: pool_size - idle_connections as u32,
        })
    }

    /// Stream large datasets in chunks for massive portfolio calculations
    pub async fn stream_portfolio_price_series_chunked<F, Fut>(
        &self,
        portfolio_id: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
        chunk_size: usize,
        processor: F,
    ) -> Result<()>
    where
        F: Fn(Vec<(NaiveDate, f64)>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        let mut last_date = start_date;
        let chunk_size = chunk_size as i64;

        loop {
            use futures::TryStreamExt;

            let query = format!(
                r#"
                SELECT h.date, SUM(h.quantity * p.closing_price) as total_value
                FROM holdings h
                JOIN prices p ON h.symbol = p.symbol AND h.date = p.date
                WHERE h.portfolio_id = $1 AND h.date >= $2 AND h.date <= $3
                GROUP BY h.date
                ORDER BY h.date
                LIMIT $4
                "#
            );

            let mut rows = sqlx::query(&query)
                .bind(portfolio_id)
                .bind(last_date)
                .bind(end_date)
                .bind(chunk_size)
                .fetch(&self.pool);

            let mut chunk_data = Vec::new();
            let mut row_count = 0;

            while let Some(row) = rows
                .try_next()
                .await
                .context("Failed to fetch price series chunk")?
            {
                let date: NaiveDate = row.get("date");
                let total_value: Option<f64> = row.get("total_value");

                if let Some(value) = total_value {
                    chunk_data.push((date, value));
                    row_count += 1;
                }
            }

            if chunk_data.is_empty() {
                break;
            }

            if let Some((last_row_date, _)) = chunk_data.last() {
                last_date = *last_row_date + chrono::Duration::days(1);
            }

            processor(chunk_data).await?;

            if row_count < chunk_size as usize {
                break;
            }
        }

        Ok(())
    }

    /// Get benchmark data for tracking error calculations
    pub async fn get_benchmark_series(
        &self,
        benchmark_id: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<Vec<(NaiveDate, f64)>> {
        use futures::TryStreamExt;

        let query = r#"
            SELECT date, bmk_returns
            FROM benchmark
            WHERE bmk_id = $1 AND date BETWEEN $2 AND $3
            ORDER BY date
        "#;

        let mut rows = sqlx::query(query)
            .bind(benchmark_id)
            .bind(start_date)
            .bind(end_date)
            .fetch(&self.pool);

        let mut benchmark_series = Vec::new();

        while let Some(row) = rows
            .try_next()
            .await
            .context("Failed to fetch benchmark data")?
        {
            let date: NaiveDate = row.get("date");
            let return_value: f64 = row.get("bmk_returns");
            benchmark_series.push((date, return_value));
        }

        Ok(benchmark_series)
    }

    /// Get portfolio details for validation
    pub async fn get_portfolio_info(&self, portfolio_id: &str) -> Result<Option<PortfolioInfo>> {
        let result = sqlx::query_as::<_, PortfolioInfo>(
            "SELECT portfolio_id, portfolio_name, portfolio_type FROM portfolios WHERE portfolio_id = $1"
        )
        .bind(portfolio_id)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to fetch portfolio info")?;

        Ok(result)
    }
}

#[derive(Debug)]
pub struct DatabaseStats {
    pub portfolio_count: i64,
    pub total_records: i64,
    pub min_date: Option<NaiveDate>,
    pub max_date: Option<NaiveDate>,
}

#[derive(Debug)]
pub struct HealthStats {
    pub response_time_ms: f64,
    pub pool_size: u32,
    pub idle_connections: u32,
    pub active_connections: u32,
}

#[derive(Debug, sqlx::FromRow)]
pub struct PortfolioInfo {
    pub portfolio_id: String,
    pub portfolio_name: String,
    pub portfolio_type: String,
}

// Optimization indexes for the actual schema
pub const DATABASE_OPTIMIZATION_SQL: &str = r#"
-- Indexes for blazing-fast queries on actual schema
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_holdings_portfolio_date 
ON holdings (portfolio_id, date);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_holdings_symbol_date 
ON holdings (symbol, date);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_prices_symbol_date 
ON prices (symbol, date);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_benchmark_bmk_date 
ON benchmark (bmk_id, date);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_id 
ON portfolios (portfolio_id);

-- Composite index for the main join
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_holdings_portfolio_symbol_date 
ON holdings (portfolio_id, symbol, date);

-- Partial indexes for recent data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_holdings_recent 
ON holdings (portfolio_id, date DESC) 
WHERE date >= CURRENT_DATE - INTERVAL '1 year';

-- Update statistics
ANALYZE holdings;
ANALYZE prices;
ANALYZE benchmark;
ANALYZE portfolios;
ANALYZE instruments;
"#;
