// src/database.rs - Optimized for MASSIVE datasets
use crate::config::AppConfig;
use anyhow::{Context, Result};
use chrono::NaiveDate;
use sqlx::{
    PgPool, Row,
    postgres::{PgConnectOptions, PgPoolOptions, PgRow},
};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::{error, info, warn};

#[derive(Clone)]
pub struct Database {
    pool: PgPool,
    // Pre-compiled queries for zero parsing overhead
    queries: Arc<DatabaseQueries>,
}

pub struct DatabaseQueries {
    pub portfolio_price: &'static str,
    pub portfolio_price_range: &'static str,
    pub daily_return_optimized: &'static str,
    pub price_series_streaming: &'static str,
    pub batch_portfolio_prices: &'static str,
    pub correlation_data_parallel: &'static str,
}

impl DatabaseQueries {
    pub fn new() -> Self {
        Self {
            portfolio_price: "SELECT price FROM portfolio_prices WHERE portfolio_id = $1 AND date = $2",

            portfolio_price_range: "SELECT date, price FROM portfolio_prices 
               WHERE portfolio_id = $1 AND date BETWEEN $2 AND $3 
               ORDER BY date",

            daily_return_optimized: "WITH price_data AS (
                  SELECT date, price, 
                         LAG(price) OVER (ORDER BY date) as prev_price
                  FROM portfolio_prices 
                  WHERE portfolio_id = $1 AND date <= $2
                  ORDER BY date DESC 
                  LIMIT 2
              )
              SELECT 
                  CASE 
                      WHEN prev_price IS NOT NULL AND prev_price != 0 
                      THEN (price - prev_price) / prev_price 
                      ELSE NULL 
                  END as daily_return
              FROM price_data 
              WHERE date = $2",

            price_series_streaming: "SELECT date, price FROM portfolio_prices 
               WHERE portfolio_id = $1 AND date >= $2 AND date <= $3
               ORDER BY date
               LIMIT $4", // Keyset pagination for huge datasets

            batch_portfolio_prices: "SELECT portfolio_id, date, price 
               FROM portfolio_prices 
               WHERE (portfolio_id, date) = ANY($1)",

            correlation_data_parallel: "WITH p1_data AS (
                  SELECT date, price as p1_price
                  FROM portfolio_prices 
                  WHERE portfolio_id = $1 AND date BETWEEN $3 AND $4
              ), p2_data AS (
                  SELECT date, price as p2_price  
                  FROM portfolio_prices
                  WHERE portfolio_id = $2 AND date BETWEEN $3 AND $4
              )
              SELECT p1_data.date, p1_data.p1_price, p2_data.p2_price
              FROM p1_data 
              INNER JOIN p2_data ON p1_data.date = p2_data.date
              ORDER BY p1_data.date",
        }
    }
}

impl Database {
    /// Create database connection optimized for massive datasets
    pub async fn new(config: &AppConfig) -> Result<Self> {
        info!("🗄️  Initializing database connection for MASSIVE datasets...");

        let connect_options = PgConnectOptions::new()
            .host(&config.database.host)
            .port(config.database.port)
            .username(&config.database.username)
            .password(&config.database.password)
            .database(&config.database.database)
            .application_name("nuclear_risk_api")
            // Optimizations for large datasets
            .statement_cache_capacity(5000) // Cache many prepared statements
            .options([
                ("tcp_nodelay", "true"),                            // Reduce network latency
                ("tcp_user_timeout", "5000"),                       // Fast failure detection
                ("statement_timeout", "30s"),                       // Prevent long-running queries
                ("idle_in_transaction_session_timeout", "10s"),     // Clean up idle connections
                ("shared_preload_libraries", "pg_stat_statements"), // Performance monitoring
                ("max_connections", "200"),                         // Handle high concurrency
                ("shared_buffers", "256MB"),                        // Large buffer pool
                ("effective_cache_size", "1GB"),                    // Assume large system cache
                ("work_mem", "64MB"),                               // Large sort/hash operations
                ("maintenance_work_mem", "256MB"),                  // Large maintenance operations
            ]);

        let pool = PgPoolOptions::new()
            .max_connections(80) // High concurrency for massive datasets
            .min_connections(40) // Always-ready connections
            .acquire_timeout(Duration::from_millis(100)) // Fast acquisition
            .idle_timeout(Duration::from_secs(300)) // Keep connections warm
            .max_lifetime(Duration::from_secs(1800)) // Rotate connections periodically
            .test_before_acquire(false) // Skip health checks for speed
            .after_connect(|conn, _meta| {
                Box::pin(async move {
                    // Set session-level optimizations
                    sqlx::query("SET enable_seqscan = off") // Prefer indexes
                        .execute(&mut *conn)
                        .await?;
                    sqlx::query("SET enable_hashjoin = on") // Fast joins
                        .execute(&mut *conn)
                        .await?;
                    sqlx::query("SET enable_mergejoin = on") // Efficient large joins
                        .execute(&mut *conn)
                        .await?;
                    sqlx::query("SET work_mem = '64MB'") // Large operations
                        .execute(&mut *conn)
                        .await?;
                    sqlx::query("SET effective_io_concurrency = 200") // SSD optimization
                        .execute(&mut *conn)
                        .await?;

                    Ok(())
                })
            })
            .connect_with(connect_options)
            .await
            .context("Failed to create database connection pool")?;

        // Simple connectivity test - much faster than COUNT(*)
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

    /// Get single portfolio price (ultra-optimized for point queries)
    pub async fn get_portfolio_price(
        &self,
        portfolio_id: &str,
        date: NaiveDate,
    ) -> Result<Option<f64>> {
        let result = sqlx::query_scalar(
            "SELECT price FROM portfolio_prices WHERE portfolio_id = $1 AND date = $2"
        )
        .bind(portfolio_id)
        .bind(date)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to fetch portfolio price")?;

        Ok(result)
    }

    /// Get portfolio price series with keyset pagination for massive datasets
    pub async fn get_portfolio_price_series(
        &self,
        portfolio_id: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<Vec<(NaiveDate, f64)>> {
        // Use streaming with keyset pagination - much faster than OFFSET
        let mut all_data = Vec::new();
        let mut last_date = start_date;
        const CHUNK_SIZE: i64 = 10000;

        loop {
            // Stream data instead of materializing with fetch_all
            use futures::TryStreamExt;
            
            let mut rows = sqlx::query(
                "SELECT date, price FROM portfolio_prices 
               WHERE portfolio_id = $1 AND date >= $2 AND date <= $3
               ORDER BY date
               LIMIT $4"
            )
            .bind(portfolio_id)
            .bind(last_date)
            .bind(end_date)
            .bind(CHUNK_SIZE)
            .fetch(&self.pool);

            let mut chunk_data = Vec::new();
            let mut row_count = 0;

            while let Some(row) = rows.try_next().await.context("Failed to fetch portfolio price series")? {
                let date: NaiveDate = row.get("date");
                let price: f64 = row.get("price");
                chunk_data.push((date, price));
                row_count += 1;
            }

            if chunk_data.is_empty() {
                break;
            }

            // Update last_date for next iteration (keyset pagination)
            if let Some((last_row_date, _)) = chunk_data.last() {
                last_date = *last_row_date + chrono::Duration::days(1);
            }

            all_data.extend(chunk_data);

            if row_count < CHUNK_SIZE as usize {
                break;
            }
        }

        Ok(all_data)
    }

    /// Get daily return with single optimized query
    pub async fn get_daily_return(
        &self,
        portfolio_id: &str,
        date: NaiveDate,
    ) -> Result<Option<f64>> {
        let result = sqlx::query_scalar(
            r#"
          WITH price_data AS (
              SELECT date, price,
                     LAG(price) OVER (ORDER BY date) as prev_price
              FROM portfolio_prices 
              WHERE portfolio_id = $1 AND date <= $2
              ORDER BY date DESC
              LIMIT 2
          )
          SELECT 
              CASE 
                  WHEN prev_price IS NOT NULL AND prev_price != 0 
                  THEN (price - prev_price) / prev_price 
                  ELSE NULL 
              END as daily_return
          FROM price_data 
          WHERE date = $2
          "#
        )
        .bind(portfolio_id)
        .bind(date)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to calculate daily return")?;

        Ok(result.flatten())
    }

    /// Batch portfolio prices for correlation calculations - NUCLEAR SPEED
    pub async fn get_aligned_price_series_parallel(
        &self,
        portfolio_id1: &str,
        portfolio_id2: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        // Stream aligned data using fetch() instead of fetch_all() for memory efficiency
        use sqlx::Row;
        use futures::TryStreamExt;

        let mut rows = sqlx::query(
            r#"
          WITH p1_data AS (
              SELECT date, price as p1_price
              FROM portfolio_prices 
              WHERE portfolio_id = $1 AND date BETWEEN $3 AND $4
          ), p2_data AS (
              SELECT date, price as p2_price  
              FROM portfolio_prices
              WHERE portfolio_id = $2 AND date BETWEEN $3 AND $4
          )
          SELECT p1_data.date, p1_data.p1_price, p2_data.p2_price
          FROM p1_data 
          INNER JOIN p2_data ON p1_data.date = p2_data.date
          ORDER BY p1_data.date
          "#
        )
        .bind(portfolio_id1)
        .bind(portfolio_id2)
        .bind(start_date)
        .bind(end_date)
        .fetch(&self.pool);

        let mut prices1 = Vec::new();
        let mut prices2 = Vec::new();

        while let Some(row) = rows.try_next().await.context("Failed to fetch aligned portfolio data")? {
            let p1_price: f64 = row.get("p1_price");
            let p2_price: f64 = row.get("p2_price");
            prices1.push(p1_price);
            prices2.push(p2_price);
        }

        Ok((prices1, prices2))
    }

    /// Stream large price series in chunks with keyset pagination
    pub async fn stream_price_series_chunked<F, Fut>(
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
            // Stream chunks instead of materializing with fetch_all
            use futures::TryStreamExt;
            
            let mut rows = sqlx::query(
                "SELECT date, price FROM portfolio_prices 
               WHERE portfolio_id = $1 AND date >= $2 AND date <= $3
               ORDER BY date
               LIMIT $4"
            )
            .bind(portfolio_id)
            .bind(last_date)
            .bind(end_date)
            .bind(chunk_size)
            .fetch(&self.pool);

            let mut chunk_data = Vec::new();
            let mut row_count = 0;

            while let Some(row) = rows.try_next().await.context("Failed to fetch price series chunk")? {
                let date: NaiveDate = row.get("date");
                let price: f64 = row.get("price");
                chunk_data.push((date, price));
                row_count += 1;
            }

            if chunk_data.is_empty() {
                break;
            }

            // Update last_date for next iteration
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

    /// Get database statistics for optimization
    pub async fn get_database_stats(&self) -> Result<DatabaseStats> {
        let portfolio_count: (i64,) =
            sqlx::query_as("SELECT COUNT(DISTINCT portfolio_id) FROM portfolio_prices")
                .fetch_one(&self.pool)
                .await
                .context("Failed to get portfolio count")?;

        let total_records: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM portfolio_prices")
            .fetch_one(&self.pool)
            .await
            .context("Failed to get total records")?;

        let date_range: (Option<NaiveDate>, Option<NaiveDate>) =
            sqlx::query_as("SELECT MIN(date), MAX(date) FROM portfolio_prices")
                .fetch_one(&self.pool)
                .await
                .context("Failed to get date range")?;

        Ok(DatabaseStats {
            portfolio_count: portfolio_count.0,
            total_records: total_records.0,
            min_date: date_range.0,
            max_date: date_range.1,
        })
    }

    /// Batch operations for massive throughput
    pub async fn batch_portfolio_prices(
        &self,
        requests: Vec<(String, NaiveDate)>,
    ) -> Result<HashMap<String, Option<f64>>> {
        if requests.is_empty() {
            return Ok(HashMap::new());
        }

        // Convert to format PostgreSQL expects for array operations
        let portfolio_date_pairs: Vec<String> = requests
            .iter()
            .map(|(portfolio_id, date)| format!("('{}','{}')", portfolio_id, date))
            .collect();

        let values_clause = portfolio_date_pairs.join(",");
        let query = format!(
            "SELECT portfolio_id, date, price 
           FROM portfolio_prices 
           WHERE (portfolio_id, date) IN (VALUES {})",
            values_clause
        );

        let rows = sqlx::query(&query)
            .fetch_all(&self.pool)
            .await
            .context("Failed to execute batch portfolio prices query")?;

        let mut results = HashMap::new();

        for row in rows {
            let portfolio_id: String = row.get("portfolio_id");
            let date: NaiveDate = row.get("date");
            let price: f64 = row.get("price");
            let key = format!("{}:{}", portfolio_id, date);
            results.insert(key, Some(price));
        }

        // Fill in None values for missing data
        for (portfolio_id, date) in requests {
            let key = format!("{}:{}", portfolio_id, date);
            results.entry(key).or_insert(None);
        }

        Ok(results)
    }

    /// Health check with performance metrics
    pub async fn health_check(&self) -> Result<HealthStats> {
        let start = std::time::Instant::now();

        // Simple query to test connection
        let _: (i64,) = sqlx::query_as("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .context("Database health check failed")?;

        let response_time = start.elapsed();

        // Get connection pool stats
        let pool_size = self.pool.size();
        let idle_connections = self.pool.num_idle();

        Ok(HealthStats {
            response_time_ms: response_time.as_millis() as f64,
            pool_size,
            idle_connections: idle_connections as u32,
            active_connections: pool_size - idle_connections as u32,
        })
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

// Database optimization recommendations for massive datasets
pub const DATABASE_OPTIMIZATION_SQL: &str = r#"
-- Indexes for blazing-fast queries on massive datasets
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_prices_portfolio_date 
ON portfolio_prices (portfolio_id, date);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_prices_date 
ON portfolio_prices (date);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_prices_portfolio_id 
ON portfolio_prices (portfolio_id);

-- Partial indexes for commonly queried recent data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_prices_recent 
ON portfolio_prices (portfolio_id, date DESC) 
WHERE date >= CURRENT_DATE - INTERVAL '1 year';

-- Clustered index for range queries (if PostgreSQL 13+)
-- CLUSTER portfolio_prices USING idx_portfolio_prices_portfolio_date;

-- Statistics for query optimization
ANALYZE portfolio_prices;

-- Enable parallel query execution for large datasets
SET max_parallel_workers_per_gather = 4;
SET max_parallel_workers = 8;
SET parallel_tuple_cost = 0.01;
SET parallel_setup_cost = 100;
"#;
