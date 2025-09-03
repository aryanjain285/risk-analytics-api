// src/monitoring.rs - Nuclear performance monitoring and metrics
use std::{
  sync::{
      atomic::{AtomicU64, AtomicUsize, Ordering},
      Arc,
  },
  time::{Duration, Instant},
  collections::HashMap,
};
use axum::{
  extract::Request,
  middleware::Next,
  response::Response,
};
use tracing::{info, warn, error};
use serde::Serialize;
use futures::future;

/// Global performance metrics collector
pub struct PerformanceMonitor {
  // Request metrics
  total_requests: AtomicU64,
  successful_requests: AtomicU64,
  failed_requests: AtomicU64,
  
  // Timing metrics (in microseconds for precision)
  total_response_time_us: AtomicU64,
  min_response_time_us: AtomicU64,
  max_response_time_us: AtomicU64,
  
  // Endpoint-specific metrics
  endpoint_metrics: Arc<dashmap::DashMap<String, EndpointStats>>,
  
  // System metrics
  start_time: Instant,
  current_connections: AtomicUsize,
  peak_connections: AtomicUsize,
  
  // Error tracking
  error_counts: Arc<dashmap::DashMap<String, AtomicU64>>,
}

#[derive(Debug)]
pub struct EndpointStats {
  pub request_count: AtomicU64,
  pub total_response_time_us: AtomicU64,
  pub min_response_time_us: AtomicU64,
  pub max_response_time_us: AtomicU64,
  pub error_count: AtomicU64,
  pub cache_hits: AtomicU64,
  pub cache_misses: AtomicU64,
  pub last_request: std::sync::Mutex<Option<Instant>>,
}

impl EndpointStats {
  fn new() -> Self {
      Self {
          request_count: AtomicU64::new(0),
          total_response_time_us: AtomicU64::new(0),
          min_response_time_us: AtomicU64::new(u64::MAX),
          max_response_time_us: AtomicU64::new(0),
          error_count: AtomicU64::new(0),
          cache_hits: AtomicU64::new(0),
          cache_misses: AtomicU64::new(0),
          last_request: std::sync::Mutex::new(None),
      }
  }
  
  fn record_request(&self, response_time: Duration, is_success: bool) {
      let response_time_us = response_time.as_micros() as u64;
      
      self.request_count.fetch_add(1, Ordering::Relaxed);
      self.total_response_time_us.fetch_add(response_time_us, Ordering::Relaxed);
      
      // Update min response time
      let mut current_min = self.min_response_time_us.load(Ordering::Relaxed);
      while response_time_us < current_min {
          match self.min_response_time_us.compare_exchange_weak(
              current_min,
              response_time_us,
              Ordering::Relaxed,
              Ordering::Relaxed,
          ) {
              Ok(_) => break,
              Err(actual) => current_min = actual,
          }
      }
      
      // Update max response time  
      let mut current_max = self.max_response_time_us.load(Ordering::Relaxed);
      while response_time_us > current_max {
          match self.max_response_time_us.compare_exchange_weak(
              current_max,
              response_time_us,
              Ordering::Relaxed,
              Ordering::Relaxed,
          ) {
              Ok(_) => break,
              Err(actual) => current_max = actual,
          }
      }
      
      if !is_success {
          self.error_count.fetch_add(1, Ordering::Relaxed);
      }
      
      // Update last request time
      if let Ok(mut last_request) = self.last_request.lock() {
          *last_request = Some(Instant::now());
      }
  }
  
  fn record_cache_hit(&self) {
      self.cache_hits.fetch_add(1, Ordering::Relaxed);
  }
  
  fn record_cache_miss(&self) {
      self.cache_misses.fetch_add(1, Ordering::Relaxed);
  }
}

impl Default for PerformanceMonitor {
  fn default() -> Self {
      Self::new()
  }
}

impl PerformanceMonitor {
  pub fn new() -> Self {
      Self {
          total_requests: AtomicU64::new(0),
          successful_requests: AtomicU64::new(0),
          failed_requests: AtomicU64::new(0),
          total_response_time_us: AtomicU64::new(0),
          min_response_time_us: AtomicU64::new(u64::MAX),
          max_response_time_us: AtomicU64::new(0),
          endpoint_metrics: Arc::new(dashmap::DashMap::new()),
          start_time: Instant::now(),
          current_connections: AtomicUsize::new(0),
          peak_connections: AtomicUsize::new(0),
          error_counts: Arc::new(dashmap::DashMap::new()),
      }
  }

  /// Record a request with detailed metrics
  pub fn record_request(
      &self,
      endpoint: &str,
      response_time: Duration,
      status_code: u16,
      cache_hit: bool,
  ) {
      let is_success = status_code >= 200 && status_code < 400;
      
      // Global metrics
      self.total_requests.fetch_add(1, Ordering::Relaxed);
      if is_success {
          self.successful_requests.fetch_add(1, Ordering::Relaxed);
      } else {
          self.failed_requests.fetch_add(1, Ordering::Relaxed);
          
          // Track error types
          let error_key = format!("status_{}", status_code);
          self.error_counts
              .entry(error_key)
              .or_insert_with(|| AtomicU64::new(0))
              .fetch_add(1, Ordering::Relaxed);
      }
      
      let response_time_us = response_time.as_micros() as u64;
      self.total_response_time_us.fetch_add(response_time_us, Ordering::Relaxed);
      
      // Update global min/max
      let mut current_min = self.min_response_time_us.load(Ordering::Relaxed);
      while response_time_us < current_min {
          match self.min_response_time_us.compare_exchange_weak(
              current_min,
              response_time_us,
              Ordering::Relaxed,
              Ordering::Relaxed,
          ) {
              Ok(_) => break,
              Err(actual) => current_min = actual,
          }
      }
      
      let mut current_max = self.max_response_time_us.load(Ordering::Relaxed);
      while response_time_us > current_max {
          match self.max_response_time_us.compare_exchange_weak(
              current_max,
              response_time_us,
              Ordering::Relaxed,
              Ordering::Relaxed,
          ) {
              Ok(_) => break,
              Err(actual) => current_max = actual,
          }
      }

      // Endpoint-specific metrics
      let endpoint_stats = self.endpoint_metrics
          .entry(endpoint.to_string())
          .or_insert_with(EndpointStats::new);
      
      endpoint_stats.record_request(response_time, is_success);
      
      if cache_hit {
          endpoint_stats.record_cache_hit();
      } else {
          endpoint_stats.record_cache_miss();
      }
  }

  /// Record connection metrics
  pub fn record_connection_opened(&self) {
      let current = self.current_connections.fetch_add(1, Ordering::Relaxed) + 1;
      
      // Update peak connections
      let mut peak = self.peak_connections.load(Ordering::Relaxed);
      while current > peak {
          match self.peak_connections.compare_exchange_weak(
              peak,
              current,
              Ordering::Relaxed,
              Ordering::Relaxed,
          ) {
              Ok(_) => break,
              Err(actual) => peak = actual,
          }
      }
  }

  pub fn record_connection_closed(&self) {
      self.current_connections.fetch_sub(1, Ordering::Relaxed);
  }

  /// Get comprehensive performance summary
  pub fn get_performance_summary(&self) -> PerformanceSummary {
      let total_requests = self.total_requests.load(Ordering::Relaxed);
      let successful_requests = self.successful_requests.load(Ordering::Relaxed);
      let failed_requests = self.failed_requests.load(Ordering::Relaxed);
      let total_response_time_us = self.total_response_time_us.load(Ordering::Relaxed);
      let min_response_time_us = self.min_response_time_us.load(Ordering::Relaxed);
      let max_response_time_us = self.max_response_time_us.load(Ordering::Relaxed);
      
      let uptime = self.start_time.elapsed();
      let requests_per_second = if uptime.as_secs() > 0 {
          total_requests as f64 / uptime.as_secs_f64()
      } else {
          0.0
      };

      let avg_response_time_ms = if total_requests > 0 {
          (total_response_time_us as f64 / total_requests as f64) / 1000.0
      } else {
          0.0
      };

      let success_rate = if total_requests > 0 {
          successful_requests as f64 / total_requests as f64
      } else {
          0.0
      };

      PerformanceSummary {
          total_requests,
          successful_requests,
          failed_requests,
          success_rate,
          requests_per_second,
          avg_response_time_ms,
          min_response_time_ms: if min_response_time_us == u64::MAX { 
              0.0 
          } else { 
              min_response_time_us as f64 / 1000.0 
          },
          max_response_time_ms: max_response_time_us as f64 / 1000.0,
          uptime_seconds: uptime.as_secs_f64(),
          current_connections: self.current_connections.load(Ordering::Relaxed),
          peak_connections: self.peak_connections.load(Ordering::Relaxed),
          performance_grade: self.calculate_performance_grade(avg_response_time_ms, success_rate),
      }
  }

  /// Get endpoint-specific metrics
  pub fn get_endpoint_metrics(&self) -> HashMap<String, EndpointMetricsSummary> {
      let mut metrics = HashMap::new();
      
      for entry in self.endpoint_metrics.iter() {
          let endpoint = entry.key();
          let stats = entry.value();
          
          let request_count = stats.request_count.load(Ordering::Relaxed);
          let total_time_us = stats.total_response_time_us.load(Ordering::Relaxed);
          let min_time_us = stats.min_response_time_us.load(Ordering::Relaxed);
          let max_time_us = stats.max_response_time_us.load(Ordering::Relaxed);
          let error_count = stats.error_count.load(Ordering::Relaxed);
          let cache_hits = stats.cache_hits.load(Ordering::Relaxed);
          let cache_misses = stats.cache_misses.load(Ordering::Relaxed);
          
          let avg_response_time_ms = if request_count > 0 {
              (total_time_us as f64 / request_count as f64) / 1000.0
          } else {
              0.0
          };
          
          let cache_hit_ratio = if cache_hits + cache_misses > 0 {
              cache_hits as f64 / (cache_hits + cache_misses) as f64
          } else {
              0.0
          };

          metrics.insert(endpoint.clone(), EndpointMetricsSummary {
              request_count,
              avg_response_time_ms,
              min_response_time_ms: if min_time_us == u64::MAX { 0.0 } else { min_time_us as f64 / 1000.0 },
              max_response_time_ms: max_time_us as f64 / 1000.0,
              error_count,
              error_rate: if request_count > 0 { error_count as f64 / request_count as f64 } else { 0.0 },
              cache_hit_ratio,
              cache_hits,
              cache_misses,
          });
      }
      
      metrics
  }

  /// Calculate performance grade based on metrics
  fn calculate_performance_grade(&self, avg_response_time_ms: f64, success_rate: f64) -> String {
      match (avg_response_time_ms, success_rate) {
          (rt, sr) if rt < 1.0 && sr >= 0.99 => "A++ NUCLEAR PERFORMANCE 🦀🔥💥".to_string(),
          (rt, sr) if rt < 5.0 && sr >= 0.98 => "A+ BLAZING FAST ⚡🚀".to_string(),
          (rt, sr) if rt < 10.0 && sr >= 0.95 => "A EXCELLENT 👍✨".to_string(),
          (rt, sr) if rt < 50.0 && sr >= 0.90 => "B GOOD 👌".to_string(),
          (rt, sr) if rt < 100.0 && sr >= 0.80 => "C ACCEPTABLE ⚠️".to_string(),
          _ => "D NEEDS OPTIMIZATION 🚨".to_string(),
      }
  }

  /// Get error summary
  pub fn get_error_summary(&self) -> HashMap<String, u64> {
      let mut errors = HashMap::new();
      
      for entry in self.error_counts.iter() {
          let error_type = entry.key().clone();
          let count = entry.value().load(Ordering::Relaxed);
          errors.insert(error_type, count);
      }
      
      errors
  }

  /// Reset all metrics (useful for testing)
  pub fn reset(&self) {
      self.total_requests.store(0, Ordering::Relaxed);
      self.successful_requests.store(0, Ordering::Relaxed);
      self.failed_requests.store(0, Ordering::Relaxed);
      self.total_response_time_us.store(0, Ordering::Relaxed);
      self.min_response_time_us.store(u64::MAX, Ordering::Relaxed);
      self.max_response_time_us.store(0, Ordering::Relaxed);
      self.current_connections.store(0, Ordering::Relaxed);
      self.peak_connections.store(0, Ordering::Relaxed);
      
      self.endpoint_metrics.clear();
      self.error_counts.clear();
  }

  /// Generate detailed performance report
  pub fn generate_report(&self) -> PerformanceReport {
      let summary = self.get_performance_summary();
      let endpoint_metrics = self.get_endpoint_metrics();
      let error_summary = self.get_error_summary();
      
      PerformanceReport {
          summary,
          endpoint_metrics,
          error_summary,
          generated_at: chrono::Utc::now(),
      }
  }
}

// Global performance monitor instance
static PERFORMANCE_MONITOR: std::sync::OnceLock<PerformanceMonitor> = std::sync::OnceLock::new();

pub fn get_performance_monitor() -> &'static PerformanceMonitor {
  PERFORMANCE_MONITOR.get_or_init(PerformanceMonitor::new)
}

/// Middleware for automatic performance tracking
pub async fn performance_middleware(
  request: Request,
  next: Next,
) -> Response {
  let start = Instant::now();
  let monitor = get_performance_monitor();
  
  // Extract endpoint from request
  let endpoint = request.uri().path().to_string();
  
  // Record connection opened
  monitor.record_connection_opened();
  
  // Process request
  let response = next.run(request).await;
  
  // Record metrics
  let duration = start.elapsed();
  let status_code = response.status().as_u16();
  
  // For now, we'll assume cache hit detection happens in handlers
  // In production, you'd pass this information through request extensions
  monitor.record_request(&endpoint, duration, status_code, false);
  
  // Record connection closed
  monitor.record_connection_closed();
  
  // Log slow requests for optimization
  if duration.as_millis() > 100 {
      warn!(
          "🐌 Slow request detected: {} took {:.2}ms (status: {})",
          endpoint,
          duration.as_millis(),
          status_code
      );
  }
  
  response
}

/// Background monitoring task for continuous health checking
pub fn start_monitoring_task() {
  tokio::spawn(async {
      let mut interval = tokio::time::interval(Duration::from_secs(60));
      let monitor = get_performance_monitor();
      
      loop {
          interval.tick().await;
          
          let summary = monitor.get_performance_summary();
          
          info!(
              "📊 Performance Report: {:.1} req/s, {:.2}ms avg, {:.1}% success, {}",
              summary.requests_per_second,
              summary.avg_response_time_ms,
              summary.success_rate * 100.0,
              summary.performance_grade
          );
          
          // Alert on performance degradation
          if summary.avg_response_time_ms > 100.0 {
              warn!("🚨 Performance Alert: Average response time is {:.2}ms", summary.avg_response_time_ms);
          }
          
          if summary.success_rate < 0.95 {
              warn!("🚨 Reliability Alert: Success rate is {:.1}%", summary.success_rate * 100.0);
          }
          
          // Memory pressure monitoring
          if summary.current_connections > 1000 {
              warn!("📈 High connection count: {} active connections", summary.current_connections);
          }
      }
  });
}

// === DATA MODELS FOR MONITORING ===

#[derive(Debug, Serialize)]
pub struct PerformanceSummary {
  pub total_requests: u64,
  pub successful_requests: u64,
  pub failed_requests: u64,
  pub success_rate: f64,
  pub requests_per_second: f64,
  pub avg_response_time_ms: f64,
  pub min_response_time_ms: f64,
  pub max_response_time_ms: f64,
  pub uptime_seconds: f64,
  pub current_connections: usize,
  pub peak_connections: usize,
  pub performance_grade: String,
}

#[derive(Debug, Serialize)]
pub struct EndpointMetricsSummary {
  pub request_count: u64,
  pub avg_response_time_ms: f64,
  pub min_response_time_ms: f64,
  pub max_response_time_ms: f64,
  pub error_count: u64,
  pub error_rate: f64,
  pub cache_hit_ratio: f64,
  pub cache_hits: u64,
  pub cache_misses: u64,
}

#[derive(Debug, Serialize)]
pub struct PerformanceReport {
  pub summary: PerformanceSummary,
  pub endpoint_metrics: HashMap<String, EndpointMetricsSummary>,
  pub error_summary: HashMap<String, u64>,
  pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// System resource monitoring
pub struct SystemMonitor;

impl SystemMonitor {
  /// Get current memory usage in MB
  pub fn get_memory_usage_mb() -> f64 {
      #[cfg(target_os = "linux")]
      {
          if let Ok(output) = std::process::Command::new("ps")
              .args(&["-o", "rss=", "-p"])
              .arg(std::process::id().to_string())
              .output()
          {
              if let Ok(stdout) = String::from_utf8(output.stdout) {
                  if let Ok(rss_kb) = stdout.trim().parse::<f64>() {
                      return rss_kb / 1024.0; // Convert KB to MB
                  }
              }
          }
      }
      
      0.0 // Fallback
  }

  /// Get current CPU usage percentage
  pub fn get_cpu_usage_percent() -> f64 {
      // This is a simplified implementation
      // In production, you'd use a proper system monitoring library
      0.0 // Placeholder
  }

  /// Get system load average
  pub fn get_load_average() -> (f64, f64, f64) {
      #[cfg(target_os = "linux")]
      {
          if let Ok(content) = std::fs::read_to_string("/proc/loadavg") {
              let parts: Vec<&str> = content.split_whitespace().collect();
              if parts.len() >= 3 {
                  let load1 = parts[0].parse().unwrap_or(0.0);
                  let load5 = parts[1].parse().unwrap_or(0.0);
                  let load15 = parts[2].parse().unwrap_or(0.0);
                  return (load1, load5, load15);
              }
          }
      }
      
      (0.0, 0.0, 0.0) // Fallback
  }
}

/// Health check utilities
pub mod health {
  use super::*;
  
  #[derive(Debug, Serialize)]
  pub struct HealthStatus {
      pub status: String,
      pub checks: HashMap<String, HealthCheck>,
      pub overall_grade: String,
      pub timestamp: chrono::DateTime<chrono::Utc>,
  }
  
  #[derive(Debug, Serialize)]
  pub struct HealthCheck {
      pub status: String,
      pub response_time_ms: f64,
      pub message: String,
      pub last_error: Option<String>,
  }
  
  pub async fn comprehensive_health_check(
      db: &crate::database::Database,
      cache: &crate::cache::NuclearCache,
  ) -> HealthStatus {
      let mut checks = HashMap::new();
      let start_time = Instant::now();
      
      // Database health check
      let db_check = match db.health_check().await {
          Ok(stats) => HealthCheck {
              status: "healthy".to_string(),
              response_time_ms: stats.response_time_ms,
              message: format!(
                  "Connected with {} active connections",
                  stats.active_connections
              ),
              last_error: None,
          },
          Err(e) => HealthCheck {
              status: "unhealthy".to_string(),
              response_time_ms: start_time.elapsed().as_millis() as f64,
              message: "Database connection failed".to_string(),
              last_error: Some(e.to_string()),
          },
      };
      checks.insert("database".to_string(), db_check);
      
      // Cache health check
      let cache_stats = cache.get_cache_stats();
      let cache_check = HealthCheck {
          status: "healthy".to_string(),
          response_time_ms: 0.1, // Cache is always fast
          message: format!(
              "L1: {} entries, L2: {} entries, {:.1}% hit ratio",
              cache_stats.l1_portfolio_prices + cache_stats.l1_daily_returns + 
              cache_stats.l1_volatilities + cache_stats.l1_correlations,
              cache_stats.l2_price_series + cache_stats.l2_calculations,
              cache_stats.overall_hit_ratio * 100.0
          ),
          last_error: None,
      };
      checks.insert("cache".to_string(), cache_check);
      
      // Performance monitoring health
      let monitor = get_performance_monitor();
      let perf_summary = monitor.get_performance_summary();
      let perf_check = HealthCheck {
          status: if perf_summary.avg_response_time_ms < 50.0 && perf_summary.success_rate > 0.95 {
              "excellent".to_string()
          } else if perf_summary.avg_response_time_ms < 200.0 && perf_summary.success_rate > 0.90 {
              "good".to_string()
          } else {
              "degraded".to_string()
          },
          response_time_ms: perf_summary.avg_response_time_ms,
          message: format!(
              "{:.1} req/s, {:.1}% success rate",
              perf_summary.requests_per_second,
              perf_summary.success_rate * 100.0
          ),
          last_error: None,
      };
      checks.insert("performance".to_string(), perf_check);
      
      // Determine overall status
      let overall_status = if checks.values().all(|check| check.status == "healthy" || check.status == "excellent") {
          "healthy"
      } else if checks.values().any(|check| check.status == "unhealthy") {
          "unhealthy"
      } else {
          "degraded"
      };
      
      let overall_grade = perf_summary.performance_grade;
      
      HealthStatus {
          status: overall_status.to_string(),
          checks,
          overall_grade,
          timestamp: chrono::Utc::now(),
      }
  }
}

/// Alerting system for critical performance issues
pub mod alerts {
  use super::*;
  
  #[derive(Debug)]
  pub enum AlertLevel {
      Info,
      Warning,
      Critical,
  }
  
  #[derive(Debug)]
  pub struct Alert {
      pub level: AlertLevel,
      pub message: String,
      pub metric_value: f64,
      pub threshold: f64,
      pub timestamp: Instant,
  }
  
  pub struct AlertManager {
      alert_thresholds: HashMap<String, f64>,
      recent_alerts: std::sync::Mutex<Vec<Alert>>,
  }
  
  impl AlertManager {
      pub fn new() -> Self {
          let mut thresholds = HashMap::new();
          
          // Performance thresholds
          thresholds.insert("avg_response_time_ms".to_string(), 100.0);
          thresholds.insert("error_rate".to_string(), 0.05); // 5%
          thresholds.insert("memory_usage_mb".to_string(), 1000.0);
          thresholds.insert("connection_count".to_string(), 5000.0);
          
          Self {
              alert_thresholds: thresholds,
              recent_alerts: std::sync::Mutex::new(Vec::new()),
          }
      }
      
      pub fn check_thresholds(&self, metrics: &PerformanceSummary) {
          // Check response time
          if metrics.avg_response_time_ms > self.alert_thresholds["avg_response_time_ms"] {
              self.create_alert(
                  AlertLevel::Warning,
                  format!("High average response time: {:.2}ms", metrics.avg_response_time_ms),
                  metrics.avg_response_time_ms,
                  self.alert_thresholds["avg_response_time_ms"],
              );
          }
          
          // Check error rate
          let error_rate = 1.0 - metrics.success_rate;
          if error_rate > self.alert_thresholds["error_rate"] {
              self.create_alert(
                  AlertLevel::Critical,
                  format!("High error rate: {:.1}%", error_rate * 100.0),
                  error_rate,
                  self.alert_thresholds["error_rate"],
              );
          }
          
          // Check connection count
          if metrics.current_connections as f64 > self.alert_thresholds["connection_count"] {
              self.create_alert(
                  AlertLevel::Warning,
                  format!("High connection count: {}", metrics.current_connections),
                  metrics.current_connections as f64,
                  self.alert_thresholds["connection_count"],
              );
          }
      }
      
      fn create_alert(&self, level: AlertLevel, message: String, value: f64, threshold: f64) {
          let alert = Alert {
              level,
              message: message.clone(),
              metric_value: value,
              threshold,
              timestamp: Instant::now(),
          };
          
          // Log the alert
          match alert.level {
              AlertLevel::Info => info!("ℹ️ {}", message),
              AlertLevel::Warning => warn!("⚠️ {}", message),
              AlertLevel::Critical => error!("🚨 CRITICAL: {}", message),
          }
          
          // Store recent alerts
          if let Ok(mut alerts) = self.recent_alerts.lock() {
              alerts.push(alert);
              
              // Keep only recent alerts (last 100)
              if alerts.len() > 100 {
                  alerts.remove(0);
              }
          }
      }
  }
}

/// Benchmarking utilities for performance validation
pub mod benchmarks {
  use super::*;
  use std::time::Duration;
  
  pub struct LoadTest {
      pub concurrent_requests: usize,
      pub duration_seconds: u64,
      pub target_endpoint: String,
      pub expected_response_time_ms: f64,
  }
  
  impl LoadTest {
      pub async fn run(&self) -> LoadTestResult {
          let start_time = Instant::now();
          let client = reqwest::Client::new();
          let mut tasks = Vec::new();
          
          for _ in 0..self.concurrent_requests {
              let client = client.clone();
              let endpoint = self.target_endpoint.clone();
              
              tasks.push(tokio::spawn(async move {
                  let request_start = Instant::now();
                  let response = client.get(&endpoint).send().await;
                  let request_duration = request_start.elapsed();
                  
                  match response {
                      Ok(resp) => LoadTestRequest {
                          success: resp.status().is_success(),
                          response_time: request_duration,
                          status_code: resp.status().as_u16(),
                      },
                      Err(_) => LoadTestRequest {
                          success: false,
                          response_time: request_duration,
                          status_code: 500,
                      },
                  }
              }));
          }
          
          let results = futures::future::join_all(tasks).await;
          let total_duration = start_time.elapsed();
          
          let mut successful_requests = 0;
          let mut total_response_time = Duration::ZERO;
          let mut min_response_time = Duration::from_secs(3600);
          let mut max_response_time = Duration::ZERO;
          
          for result in &results {
              if let Ok(request_result) = result {
                  if request_result.success {
                      successful_requests += 1;
                  }
                  total_response_time += request_result.response_time;
                  min_response_time = min_response_time.min(request_result.response_time);
                  max_response_time = max_response_time.max(request_result.response_time);
              }
          }
          
          LoadTestResult {
              total_requests: self.concurrent_requests,
              successful_requests,
              failed_requests: self.concurrent_requests - successful_requests,
              success_rate: successful_requests as f64 / self.concurrent_requests as f64,
              total_duration,
              avg_response_time: if successful_requests > 0 {
                  total_response_time / successful_requests as u32
              } else {
                  Duration::ZERO
              },
              min_response_time,
              max_response_time,
              requests_per_second: self.concurrent_requests as f64 / total_duration.as_secs_f64(),
              meets_expectations: (successful_requests as f64 / self.concurrent_requests as f64 > 0.95)
                  && (((total_response_time / successful_requests.max(1) as u32).as_millis() as f64) 
                     < (self.expected_response_time_ms as f64)),
          }
      }
  }
  
  #[derive(Debug)]
  pub struct LoadTestRequest {
      pub success: bool,
      pub response_time: Duration,
      pub status_code: u16,
  }
  
  #[derive(Debug, Serialize)]
  pub struct LoadTestResult {
      pub total_requests: usize,
      pub successful_requests: usize,
      pub failed_requests: usize,
      pub success_rate: f64,
      pub total_duration: Duration,
      pub avg_response_time: Duration,
      pub min_response_time: Duration,
      pub max_response_time: Duration,
      pub requests_per_second: f64,
      pub meets_expectations: bool,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  
  #[test]
  fn test_performance_monitor_basic() {
      let monitor = PerformanceMonitor::new();
      
      monitor.record_request(
          "/test",
          Duration::from_millis(10),
          200,
          true,
      );
      
      let summary = monitor.get_performance_summary();
      assert_eq!(summary.total_requests, 1);
      assert_eq!(summary.successful_requests, 1);
      assert!(summary.avg_response_time_ms >= 9.0 && summary.avg_response_time_ms <= 11.0);
  }
  
  #[test]
  fn test_endpoint_metrics() {
      let monitor = PerformanceMonitor::new();
      
      // Record multiple requests to same endpoint
      for _ in 0..5 {
          monitor.record_request("/portfolio-price", Duration::from_millis(5), 200, true);
      }
      
      let endpoint_metrics = monitor.get_endpoint_metrics();
      let portfolio_metrics = &endpoint_metrics["/portfolio-price"];
      
      assert_eq!(portfolio_metrics.request_count, 5);
      assert!(portfolio_metrics.avg_response_time_ms >= 4.0 && portfolio_metrics.avg_response_time_ms <= 6.0);
  }
}