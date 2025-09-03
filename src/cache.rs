// src/cache.rs - Multi-tier caching for massive datasets
use crate::database::Database;
use anyhow::Result;
use chrono::NaiveDate;
use dashmap::DashMap;
use moka::future::Cache;
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{debug, info};

// -------------------------------
// Cache entry with metadata
// -------------------------------
#[derive(Clone, Debug)]
pub struct CacheEntry<T> {
    pub value: T,
    pub created_at: Instant,
    pub access_count: u64,
    pub last_accessed: Instant,
}

impl<T> CacheEntry<T> {
    pub fn new(value: T) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            access_count: 1,
            last_accessed: now,
        }
    }

    pub fn access(&mut self) -> &T {
        self.access_count += 1;
        self.last_accessed = Instant::now();
        &self.value
    }
}

// -------------------------------
// Rolling statistics for O(1) volatility calculations
// -------------------------------
#[derive(Clone, Debug)]
pub struct RollingVolatility {
    pub volatility: f64,
    pub sum: f64,
    pub sum_squares: f64,
    pub count: usize,
    pub window: VecDeque<f64>,
    pub last_updated: Instant,
}

impl RollingVolatility {
    pub fn new(initial_returns: &[f64]) -> Self {
        let mut window = VecDeque::with_capacity(252); // ~1 year trading days
        let mut sum = 0.0;
        let mut sum_squares = 0.0;

        for &ret in initial_returns {
            window.push_back(ret);
            sum += ret;
            sum_squares += ret * ret;
        }

        let volatility = Self::calculate_volatility(sum, sum_squares, window.len());

        Self {
            volatility,
            sum,
            sum_squares,
            count: window.len(),
            window,
            last_updated: Instant::now(),
        }
    }

    #[inline(always)]
    fn calculate_volatility(sum: f64, sum_squares: f64, count: usize) -> f64 {
        if count < 2 {
            return 0.0;
        }
        let mean = sum / count as f64;
        let variance = (sum_squares / count as f64) - (mean * mean);
        variance.sqrt() * (252.0_f64).sqrt() // annualize
    }

    /// Update volatility incrementally (O(1))
    pub fn update_with_new_return(&mut self, new_return: f64) {
        const WINDOW_SIZE: usize = 252;

        if self.window.len() >= WINDOW_SIZE {
            if let Some(old_return) = self.window.pop_front() {
                self.sum -= old_return;
                self.sum_squares -= old_return * old_return;
            }
        }

        self.window.push_back(new_return);
        self.sum += new_return;
        self.sum_squares += new_return * new_return;
        self.count = self.window.len();

        self.volatility = Self::calculate_volatility(self.sum, self.sum_squares, self.count);
        self.last_updated = Instant::now();
    }
}

// -------------------------------
// Key helpers (normalize all keys)
// -------------------------------
#[inline]
fn price_key(portfolio_id: &str, date: NaiveDate) -> String {
    format!("{}:{}", portfolio_id, date)
}

#[inline]
fn daily_return_key(portfolio_id: &str, date: NaiveDate) -> String {
    format!("ret:{}:{}", portfolio_id, date)
}

#[inline]
fn vol_key(portfolio_id: &str, start_date: NaiveDate, end_date: NaiveDate) -> String {
    format!("vol:{}:{}:{}", portfolio_id, start_date, end_date)
}

#[inline]
fn corr_key(
    portfolio_id1: &str,
    portfolio_id2: &str,
    start_date: NaiveDate,
    end_date: NaiveDate,
) -> String {
    let (p1, p2) = if portfolio_id1 < portfolio_id2 {
        (portfolio_id1, portfolio_id2)
    } else {
        (portfolio_id2, portfolio_id1)
    };
    format!("corr:{}:{}:{}:{}", p1, p2, start_date, end_date)
}

#[inline]
fn series_key(portfolio_id: &str, start_date: NaiveDate, end_date: NaiveDate) -> String {
    format!("series:{}:{}:{}", portfolio_id, start_date, end_date)
}

// -------------------------------
// Stats
// -------------------------------
#[derive(Default)]
pub struct CacheStats {
    pub l1_hits: std::sync::atomic::AtomicU64,
    pub l1_misses: std::sync::atomic::AtomicU64,
    pub l2_hits: std::sync::atomic::AtomicU64,
    pub l2_misses: std::sync::atomic::AtomicU64,
    pub total_requests: std::sync::atomic::AtomicU64,
}

impl CacheStats {
    pub fn record_l1_hit(&self) {
        self.l1_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_l1_miss(&self) {
        self.l1_misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_l2_hit(&self) {
        self.l2_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_l2_miss(&self) {
        self.l2_misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get_hit_ratio(&self) -> f64 {
        let hits = self.l1_hits.load(std::sync::atomic::Ordering::Relaxed)
            + self.l2_hits.load(std::sync::atomic::Ordering::Relaxed);
        let total = self
            .total_requests
            .load(std::sync::atomic::Ordering::Relaxed);

        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

// -------------------------------
// NuclearCache
// -------------------------------
#[derive(Clone)]
pub struct NuclearCache {
    // L1 Cache: Ultra-fast in-memory (sub-microsecond access)
    pub portfolio_prices: Arc<DashMap<String, CacheEntry<f64>>>,
    pub daily_returns: Arc<DashMap<String, CacheEntry<f64>>>,
    pub volatilities: Arc<DashMap<String, CacheEntry<RollingVolatility>>>,
    pub correlations: Arc<DashMap<String, CacheEntry<f64>>>,
    pub cumulative_returns: Arc<DashMap<String, CacheEntry<f64>>>,
    pub tracking_errors: Arc<DashMap<String, CacheEntry<f64>>>,

    // L2 Cache: Advanced LRU with TTL (microsecond access)
    pub l2_price_series: Cache<String, Vec<(NaiveDate, f64)>>,
    pub l2_calculations: Cache<String, f64>,

    // Hot data pre-loading for common portfolios
    pub hot_portfolios: Arc<DashMap<String, Instant>>,

    // Cache statistics for monitoring
    pub stats: Arc<CacheStats>,
}

impl NuclearCache {
    pub async fn new() -> Result<Self> {
        info!("⚡ Initializing NUCLEAR cache system for massive datasets...");

        // L2 Cache configuration for massive datasets
        let l2_price_series = Cache::builder()
            .max_capacity(10_000) // 10K price series
            .time_to_live(Duration::from_secs(300)) // 5 minute TTL
            .time_to_idle(Duration::from_secs(60)) // 1 minute idle timeout
            .build();

        let l2_calculations = Cache::builder()
            .max_capacity(50_000) // 50K calculations
            .time_to_live(Duration::from_secs(600)) // 10 minute TTL
            .time_to_idle(Duration::from_secs(120)) // 2 minute idle timeout
            .build();

        Ok(Self {
            // L1 Caches with high capacity for massive datasets
            portfolio_prices: Arc::new(DashMap::with_capacity(100_000)),
            daily_returns: Arc::new(DashMap::with_capacity(100_000)),
            volatilities: Arc::new(DashMap::with_capacity(10_000)),
            correlations: Arc::new(DashMap::with_capacity(50_000)),
            cumulative_returns: Arc::new(DashMap::with_capacity(25_000)),
            tracking_errors: Arc::new(DashMap::with_capacity(25_000)),

            // L2 Caches
            l2_price_series,
            l2_calculations,

            // Hot data tracking
            hot_portfolios: Arc::new(DashMap::with_capacity(1_000)),

            // Statistics
            stats: Arc::new(CacheStats::default()),
        })
    }

    // ---------------------------
    // L1: Prices
    // ---------------------------
    /// Get portfolio price with intelligent caching
    pub async fn get_portfolio_price(&self, portfolio_id: &str, date: NaiveDate) -> Option<f64> {
        let key = price_key(portfolio_id, date);

        // L1 Cache check
        if let Some(mut entry) = self.portfolio_prices.get_mut(&key) {
            self.stats.record_l1_hit();
            return Some(*entry.access());
        }

        self.stats.record_l1_miss();
        None
    }

    /// Cache portfolio price with intelligent eviction
    pub fn cache_portfolio_price(&self, portfolio_id: &str, date: NaiveDate, price: f64) {
        let key = price_key(portfolio_id, date);
        self.portfolio_prices.insert(key, CacheEntry::new(price));

        // Track hot portfolios for pre-loading
        self.hot_portfolios
            .insert(portfolio_id.to_string(), Instant::now());
    }

    /// Get L1 portfolio price from cache (TTL on last_accessed)
    pub async fn get_l1_portfolio_price(&self, portfolio_id: &str, date: NaiveDate) -> Option<f64> {
        let key = price_key(portfolio_id, date);
        if let Some(entry) = self.portfolio_prices.get(&key) {
            if entry.last_accessed.elapsed() < Duration::from_secs(300) {
                return Some(entry.value);
            }
        }
        None
    }

    /// Cache portfolio price async
    pub async fn cache_portfolio_price_async(
        &self,
        portfolio_id: &str,
        date: NaiveDate,
        price: f64,
    ) {
        let key = price_key(portfolio_id, date);
        self.portfolio_prices.insert(key, CacheEntry::new(price));
    }

    // ---------------------------
    // L1: Daily returns
    // ---------------------------
    pub async fn get_l1_daily_return(&self, portfolio_id: &str, date: NaiveDate) -> Option<f64> {
        let key = daily_return_key(portfolio_id, date);
        if let Some(entry) = self.daily_returns.get(&key) {
            if entry.last_accessed.elapsed() < Duration::from_secs(300) {
                return Some(entry.value);
            }
        }
        None
    }

    pub async fn cache_daily_return(&self, portfolio_id: &str, date: NaiveDate, return_value: f64) {
        let key = daily_return_key(portfolio_id, date);
        self.daily_returns
            .insert(key, CacheEntry::new(return_value));
    }

    // ---------------------------
    // L1: Volatility
    // ---------------------------
    /// Get or compute volatility with rolling updates (L1 check only)
    pub async fn get_or_compute_volatility(
        &self,
        portfolio_id: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Option<f64> {
        let key = vol_key(portfolio_id, start_date, end_date);

        if let Some(mut entry) = self.volatilities.get_mut(&key) {
            // If cache was accessed in the last minute, treat as fresh
            if entry.last_accessed.elapsed() < Duration::from_secs(60) {
                self.stats.record_l1_hit();
                return Some(entry.access().volatility);
            }
        }

        self.stats.record_l1_miss();
        None
    }

    /// Cache volatility with rolling statistics for incremental updates
    pub fn cache_volatility(
        &self,
        portfolio_id: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
        rolling_vol: RollingVolatility,
    ) {
        let key = vol_key(portfolio_id, start_date, end_date);
        self.volatilities.insert(key, CacheEntry::new(rolling_vol));
    }

    /// Update volatility incrementally when new data arrives
    pub fn update_volatility_incremental(
        &self,
        portfolio_id: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
        new_return: f64,
    ) -> Option<f64> {
        let key = vol_key(portfolio_id, start_date, end_date);

        if let Some(mut entry) = self.volatilities.get_mut(&key) {
            entry.value.update_with_new_return(new_return);
            return Some(entry.value.volatility);
        }

        None
    }

    /// Get L1 volatility from cache (TTL on last_accessed)
    pub async fn get_l1_volatility(&self, cache_key: &str) -> Option<f64> {
        if let Some(entry) = self.volatilities.get(cache_key) {
            if entry.last_accessed.elapsed() < Duration::from_secs(300) {
                return Some(entry.value.volatility);
            }
        }
        None
    }

    /// Cache volatility (keyed) async — distinct name to avoid duplicate
    pub async fn cache_volatility_keyed_async(&self, cache_key: &str, volatility: f64) {
        let rolling_vol = RollingVolatility {
            volatility,
            sum: 0.0,
            sum_squares: 0.0,
            count: 0,
            window: VecDeque::new(),
            last_updated: Instant::now(),
        };
        self.volatilities
            .insert(cache_key.to_string(), CacheEntry::new(rolling_vol));
    }

    // ---------------------------
    // L1: Correlation
    // ---------------------------
    /// Get correlation from cache
    pub async fn get_correlation(
        &self,
        portfolio_id1: &str,
        portfolio_id2: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Option<f64> {
        let key = corr_key(portfolio_id1, portfolio_id2, start_date, end_date);

        if let Some(mut entry) = self.correlations.get_mut(&key) {
            self.stats.record_l1_hit();
            return Some(*entry.access());
        }

        self.stats.record_l1_miss();
        None
    }

    /// Cache correlation result
    pub fn cache_correlation(
        &self,
        portfolio_id1: &str,
        portfolio_id2: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
        correlation: f64,
    ) {
        let key = corr_key(portfolio_id1, portfolio_id2, start_date, end_date);
        self.correlations.insert(key, CacheEntry::new(correlation));
    }

    /// Get L1 correlation from cache (TTL on last_accessed)
    pub async fn get_l1_correlation(&self, cache_key: &str) -> Option<f64> {
        if let Some(entry) = self.correlations.get(cache_key) {
            if entry.last_accessed.elapsed() < Duration::from_secs(300) {
                return Some(entry.value);
            }
        }
        None
    }

    /// Cache correlation (keyed) async — distinct name to avoid duplicate
    pub async fn cache_correlation_keyed_async(&self, cache_key: &str, correlation: f64) {
        self.correlations
            .insert(cache_key.to_string(), CacheEntry::new(correlation));
    }

    // ---------------------------
    // L2: Series & calculations
    // ---------------------------
    /// Get L2 calculation from cache (f64)
    pub async fn get_l2_calculation(&self, cache_key: &str) -> Option<f64> {
        self.l2_calculations.get(cache_key).await
    }

    /// Cache L2 calculation (f64)
    pub async fn cache_l2_calculation(&self, cache_key: String, value: f64) {
        self.l2_calculations.insert(cache_key, value).await;
    }

    /// Get price series from L2 cache
    pub async fn get_price_series_l2(
        &self,
        portfolio_id: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Option<Vec<(NaiveDate, f64)>> {
        let key = series_key(portfolio_id, start_date, end_date);

        if let Some(series) = self.l2_price_series.get(&key).await {
            self.stats.record_l2_hit();
            return Some(series);
        }

        self.stats.record_l2_miss();
        None
    }

    /// Cache price series in L2
    pub async fn cache_price_series_l2(
        &self,
        portfolio_id: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
        series: Vec<(NaiveDate, f64)>,
    ) {
        let key = series_key(portfolio_id, start_date, end_date);
        self.l2_price_series.insert(key, series).await;
    }

    // ---------------------------
    // Warm & cleanup
    // ---------------------------
    /// Pre-warm cache for common portfolios and date ranges
    pub async fn warm_cache(&self, db: &Database) -> Result<()> {
        info!("🔥 Pre-warming cache for massive dataset optimization...");

        // Get most active portfolios from database
        let hot_portfolios = self.get_hot_portfolios(db).await?;

        for portfolio_id in hot_portfolios {
            // Pre-load recent price data
            let end_date = chrono::Utc::now().naive_utc().date();
            let start_date = end_date - chrono::Duration::days(30); // Last 30 days

            if let Ok(price_series) = db
                .get_portfolio_price_series(&portfolio_id, start_date, end_date)
                .await
            {
                let key = series_key(&portfolio_id, start_date, end_date);
                self.l2_price_series.insert(key, price_series).await;

                debug!("Pre-loaded price series for portfolio {}", portfolio_id);
            }
        }

        info!("✅ Cache warming completed");
        Ok(())
    }

    /// Get hot portfolios for pre-loading (most frequently accessed)
    async fn get_hot_portfolios(&self, _db: &Database) -> Result<Vec<String>> {
        // In production, derive from access logs / metrics
        let portfolios = vec![
            "PORTFOLIO_001".to_string(),
            "PORTFOLIO_002".to_string(),
            "PORTFOLIO_003".to_string(),
            "BENCHMARK_001".to_string(),
            "BENCHMARK_002".to_string(),
        ];
        Ok(portfolios)
    }

    /// Cache cleanup for massive datasets - prevent memory bloat
    pub async fn cleanup_stale_entries(&self) {
        let now = Instant::now();
        let stale_threshold = Duration::from_secs(3600); // 1 hour

        // Clean L1 caches
        self.portfolio_prices
            .retain(|_, entry| now.duration_since(entry.last_accessed) < stale_threshold);

        self.daily_returns
            .retain(|_, entry| now.duration_since(entry.last_accessed) < stale_threshold);

        self.volatilities
            .retain(|_, entry| now.duration_since(entry.last_accessed) < stale_threshold);

        self.correlations
            .retain(|_, entry| now.duration_since(entry.last_accessed) < stale_threshold);

        info!("🧹 Cache cleanup completed - removed stale entries");
    }

    /// Get comprehensive cache statistics
    pub fn get_cache_stats(&self) -> CacheStatistics {
        let total_requests = self
            .stats
            .total_requests
            .load(std::sync::atomic::Ordering::Relaxed);
        let l1_hits = self
            .stats
            .l1_hits
            .load(std::sync::atomic::Ordering::Relaxed);
        let l2_hits = self
            .stats
            .l2_hits
            .load(std::sync::atomic::Ordering::Relaxed);

        CacheStatistics {
            l1_portfolio_prices: self.portfolio_prices.len(),
            l1_daily_returns: self.daily_returns.len(),
            l1_volatilities: self.volatilities.len(),
            l1_correlations: self.correlations.len(),
            l1_cumulative_returns: self.cumulative_returns.len(),
            l1_tracking_errors: self.tracking_errors.len(),
            l2_price_series: self.l2_price_series.entry_count(),
            l2_calculations: self.l2_calculations.entry_count(),
            total_requests,
            l1_hit_ratio: if total_requests > 0 {
                l1_hits as f64 / total_requests as f64
            } else {
                0.0
            },
            l2_hit_ratio: if total_requests > 0 {
                l2_hits as f64 / total_requests as f64
            } else {
                0.0
            },
            overall_hit_ratio: if total_requests > 0 {
                (l1_hits + l2_hits) as f64 / total_requests as f64
            } else {
                0.0
            },
        }
    }

    /// Memory pressure management for massive datasets
    pub async fn manage_memory_pressure(&self) {
        let stats = self.get_cache_stats();

        // If we're using too much memory, aggressively clean caches
        let total_entries = stats.l1_portfolio_prices
            + stats.l1_daily_returns
            + stats.l1_volatilities
            + stats.l1_correlations;

        if total_entries > 500_000 {
            // Threshold for massive datasets
            info!(
                "📊 High cache usage detected: {} entries, cleaning up...",
                total_entries
            );

            // Remove least recently used entries from L1 caches
            let now = Instant::now();
            let cleanup_threshold = Duration::from_secs(300); // 5 minutes

            self.portfolio_prices
                .retain(|_, entry| now.duration_since(entry.last_accessed) < cleanup_threshold);

            self.daily_returns
                .retain(|_, entry| now.duration_since(entry.last_accessed) < cleanup_threshold);

            self.volatilities
                .retain(|_, entry| now.duration_since(entry.last_accessed) < cleanup_threshold);

            self.correlations
                .retain(|_, entry| now.duration_since(entry.last_accessed) < cleanup_threshold);

            info!("🧹 Aggressive cache cleanup completed");
        }
    }
}

// -------------------------------
// Statistics struct for reporting
// -------------------------------
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub l1_portfolio_prices: usize,
    pub l1_daily_returns: usize,
    pub l1_volatilities: usize,
    pub l1_correlations: usize,
    pub l1_cumulative_returns: usize,
    pub l1_tracking_errors: usize,
    pub l2_price_series: u64,
    pub l2_calculations: u64,
    pub total_requests: u64,
    pub l1_hit_ratio: f64,
    pub l2_hit_ratio: f64,
    pub overall_hit_ratio: f64,
}

// -------------------------------
// Background cache management task
// -------------------------------
pub fn start_cache_management_task(cache: NuclearCache) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));

        loop {
            interval.tick().await;

            // Memory pressure management
            cache.manage_memory_pressure().await;

            // Periodic cleanup
            cache.cleanup_stale_entries().await;

            // Log cache performance
            let stats = cache.get_cache_stats();
            let l1_entries_total = stats.l1_portfolio_prices
                + stats.l1_daily_returns
                + stats.l1_volatilities
                + stats.l1_correlations
                + stats.l1_cumulative_returns
                + stats.l1_tracking_errors;

            let l2_entries_total = stats.l2_price_series + stats.l2_calculations;

            info!(
                "📊 Cache Stats: Hit Ratio: {:.2}%, L1 Entries: {}, L2 Entries: {}",
                stats.overall_hit_ratio * 100.0,
                l1_entries_total,
                l2_entries_total
            );
        }
    });
}
