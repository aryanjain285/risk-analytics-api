// src/config.rs - Conservative configuration for database limits
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub database: DatabaseConfig,
    pub cache: CacheConfig,
    pub performance: PerformanceConfig,
    pub server: ServerConfig,
    pub logging: LoggingConfig,
}

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
    pub statement_cache_capacity: usize,
    pub enable_ssl: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub l1_max_entries: usize,
    pub l2_max_entries: u64,
    pub ttl_seconds: u64,
    pub cleanup_interval_seconds: u64,
    pub memory_threshold_mb: u64,
    pub enable_distributed: bool,
    pub redis_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enable_simd: bool,
    pub enable_parallel: bool,
    pub parallel_threshold: usize,
    pub chunk_size: usize,
    pub max_request_size: usize,
    pub enable_compression: bool,
    pub connection_pool_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: Option<usize>,
    pub max_connections: usize,
    pub timeout_seconds: u64,
    pub keep_alive_seconds: u64,
    pub enable_cors: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub enable_access_logs: bool,
    pub enable_error_logs: bool,
    pub log_format: String, // json, text
    pub log_file: Option<String>,
}

impl AppConfig {
    /// Load configuration from environment variables with conservative defaults
    pub fn load() -> Result<Self> {
        // Database configuration - CONSERVATIVE SETTINGS
        let database = DatabaseConfig {
            host: env::var("DATABASE_HOST")
                .unwrap_or_else(|_| "aws-0-ap-southeast-1.pooler.supabase.com".to_string()),
            port: env::var("DATABASE_PORT")
                .unwrap_or_else(|_| "5432".to_string())
                .parse()
                .context("Invalid DATABASE_PORT")?,
            database: env::var("DATABASE_NAME").unwrap_or_else(|_| "postgres".to_string()),
            username: env::var("DATABASE_USERNAME")
                .unwrap_or_else(|_| "team9dbuser.jdcgkhwtrsdhyysagkwb".to_string()),
            password: env::var("DATABASE_PASSWORD")
                .unwrap_or_else(|_| "e5ci7swfjroiqs4f".to_string()),
            max_connections: env::var("DATABASE_MAX_CONNECTIONS")
                .unwrap_or_else(|_| "10".to_string()) // Reduced from 80
                .parse()
                .context("Invalid DATABASE_MAX_CONNECTIONS")?,
            min_connections: env::var("DATABASE_MIN_CONNECTIONS")
                .unwrap_or_else(|_| "2".to_string()) // Reduced from 40
                .parse()
                .context("Invalid DATABASE_MIN_CONNECTIONS")?,
            connection_timeout_ms: env::var("DATABASE_TIMEOUT_MS")
                .unwrap_or_else(|_| "30000".to_string()) // Increased from 5000
                .parse()
                .context("Invalid DATABASE_TIMEOUT_MS")?,
            statement_cache_capacity: env::var("DATABASE_STATEMENT_CACHE")
                .unwrap_or_else(|_| "1000".to_string()) // Reduced from 5000
                .parse()
                .context("Invalid DATABASE_STATEMENT_CACHE")?,
            enable_ssl: env::var("DATABASE_ENABLE_SSL")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .context("Invalid DATABASE_ENABLE_SSL")?,
        };

        // Cache configuration - moderate settings
        let cache = CacheConfig {
            l1_max_entries: env::var("CACHE_L1_MAX_ENTRIES")
                .unwrap_or_else(|_| "50000".to_string()) // Reduced from 100000
                .parse()
                .context("Invalid CACHE_L1_MAX_ENTRIES")?,
            l2_max_entries: env::var("CACHE_L2_MAX_ENTRIES")
                .unwrap_or_else(|_| "25000".to_string()) // Reduced from 50000
                .parse()
                .context("Invalid CACHE_L2_MAX_ENTRIES")?,
            ttl_seconds: env::var("CACHE_TTL_SECONDS")
                .unwrap_or_else(|_| "600".to_string())
                .parse()
                .context("Invalid CACHE_TTL_SECONDS")?,
            cleanup_interval_seconds: env::var("CACHE_CLEANUP_INTERVAL")
                .unwrap_or_else(|_| "60".to_string())
                .parse()
                .context("Invalid CACHE_CLEANUP_INTERVAL")?,
            memory_threshold_mb: env::var("CACHE_MEMORY_THRESHOLD_MB")
                .unwrap_or_else(|_| "500".to_string())
                .parse()
                .context("Invalid CACHE_MEMORY_THRESHOLD_MB")?,
            enable_distributed: env::var("CACHE_ENABLE_DISTRIBUTED")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .context("Invalid CACHE_ENABLE_DISTRIBUTED")?,
            redis_url: env::var("REDIS_URL").ok(),
        };

        // Performance configuration - balanced settings
        let performance = PerformanceConfig {
            enable_simd: env::var("PERFORMANCE_ENABLE_SIMD")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .context("Invalid PERFORMANCE_ENABLE_SIMD")?,
            enable_parallel: env::var("PERFORMANCE_ENABLE_PARALLEL")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .context("Invalid PERFORMANCE_ENABLE_PARALLEL")?,
            parallel_threshold: env::var("PERFORMANCE_PARALLEL_THRESHOLD")
                .unwrap_or_else(|_| "10000".to_string())
                .parse()
                .context("Invalid PERFORMANCE_PARALLEL_THRESHOLD")?,
            chunk_size: env::var("PERFORMANCE_CHUNK_SIZE")
                .unwrap_or_else(|_| "5000".to_string()) // Reduced from 10000
                .parse()
                .context("Invalid PERFORMANCE_CHUNK_SIZE")?,
            max_request_size: env::var("PERFORMANCE_MAX_REQUEST_SIZE")
                .unwrap_or_else(|_| "500000".to_string()) // Reduced from 1000000
                .parse()
                .context("Invalid PERFORMANCE_MAX_REQUEST_SIZE")?,
            enable_compression: env::var("PERFORMANCE_ENABLE_COMPRESSION")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .context("Invalid PERFORMANCE_ENABLE_COMPRESSION")?,
            connection_pool_size: env::var("PERFORMANCE_CONNECTION_POOL_SIZE")
                .unwrap_or_else(|_| "15".to_string()) // Reduced from 80
                .parse()
                .context("Invalid PERFORMANCE_CONNECTION_POOL_SIZE")?,
        };

        // Server configuration - reasonable settings
        let server = ServerConfig {
            host: env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: env::var("SERVER_PORT")
                .unwrap_or_else(|_| "8000".to_string())
                .parse()
                .context("Invalid SERVER_PORT")?,
            workers: env::var("SERVER_WORKERS").ok().and_then(|s| s.parse().ok()),
            max_connections: env::var("SERVER_MAX_CONNECTIONS")
                .unwrap_or_else(|_| "1000".to_string()) // Reduced from 10000
                .parse()
                .context("Invalid SERVER_MAX_CONNECTIONS")?,
            timeout_seconds: env::var("SERVER_TIMEOUT_SECONDS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .context("Invalid SERVER_TIMEOUT_SECONDS")?,
            keep_alive_seconds: env::var("SERVER_KEEP_ALIVE_SECONDS")
                .unwrap_or_else(|_| "75".to_string())
                .parse()
                .context("Invalid SERVER_KEEP_ALIVE_SECONDS")?,
            enable_cors: env::var("SERVER_ENABLE_CORS")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .context("Invalid SERVER_ENABLE_CORS")?,
        };

        // Logging configuration
        let logging = LoggingConfig {
            level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            enable_access_logs: env::var("LOG_ENABLE_ACCESS")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .context("Invalid LOG_ENABLE_ACCESS")?,
            enable_error_logs: env::var("LOG_ENABLE_ERROR")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .context("Invalid LOG_ENABLE_ERROR")?,
            log_format: env::var("LOG_FORMAT").unwrap_or_else(|_| "json".to_string()),
            log_file: env::var("LOG_FILE").ok(),
        };

        Ok(Self {
            database,
            cache,
            performance,
            server,
            logging,
        })
    }

    /// Create optimized configuration for production deployment
    pub fn production_optimized() -> Self {
        Self {
            database: DatabaseConfig {
                host: "aws-0-ap-southeast-1.pooler.supabase.com".to_string(),
                port: 5432,
                database: "postgres".to_string(),
                username: "team9dbuser.jdcgkhwtrsdhyysagkwb".to_string(),
                password: "e5ci7swfjroiqs4f".to_string(),
                max_connections: 15,            // Max allowed
                min_connections: 3,             // Conservative start
                connection_timeout_ms: 30000,   // 30 second timeout
                statement_cache_capacity: 1000, // Reasonable cache
                enable_ssl: true,
            },
            cache: CacheConfig {
                l1_max_entries: 50_000,       // Moderate L1 cache
                l2_max_entries: 25_000,       // Moderate L2 cache
                ttl_seconds: 600,             // 10 minute TTL
                cleanup_interval_seconds: 60, // Clean every minute
                memory_threshold_mb: 500,     // 500MB threshold
                enable_distributed: false,    // Keep it simple
                redis_url: None,
            },
            performance: PerformanceConfig {
                enable_simd: true,         // Keep SIMD for speed
                enable_parallel: true,     // Use cores efficiently
                parallel_threshold: 5_000, // Lower threshold
                chunk_size: 5_000,         // Smaller chunks
                max_request_size: 500_000, // Reasonable request size
                enable_compression: true,  // Enable compression
                connection_pool_size: 15,  // Match max connections
            },
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8000,             // Required port
                workers: None,          // Auto-detect
                max_connections: 1_000, // Reasonable load
                timeout_seconds: 30,    // Reasonable timeout
                keep_alive_seconds: 75, // Keep connections alive
                enable_cors: true,      // Required for judge app
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                enable_access_logs: false, // Disable for performance
                enable_error_logs: true,   // Keep error logs
                log_format: "json".to_string(),
                log_file: Some("api.log".to_string()),
            },
        }
    }

    /// Development configuration with minimal resources
    pub fn development() -> Self {
        Self {
            database: DatabaseConfig {
                host: "aws-0-ap-southeast-1.pooler.supabase.com".to_string(),
                port: 5432,
                database: "postgres".to_string(),
                username: "team9dbuser.jdcgkhwtrsdhyysagkwb".to_string(),
                password: "e5ci7swfjroiqs4f".to_string(),
                max_connections: 5,            // Very conservative
                min_connections: 1,            // Start with 1
                connection_timeout_ms: 30000,  // Long timeout
                statement_cache_capacity: 100, // Small cache
                enable_ssl: true,
            },
            cache: CacheConfig {
                l1_max_entries: 10_000,
                l2_max_entries: 5_000,
                ttl_seconds: 300,
                cleanup_interval_seconds: 60,
                memory_threshold_mb: 100,
                enable_distributed: false,
                redis_url: None,
            },
            performance: PerformanceConfig {
                enable_simd: true,
                enable_parallel: false, // Single-threaded for dev
                parallel_threshold: 1000,
                chunk_size: 1000,
                max_request_size: 50_000,
                enable_compression: false,
                connection_pool_size: 5,
            },
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 8000,
                workers: Some(1),
                max_connections: 100,
                timeout_seconds: 30,
                keep_alive_seconds: 60,
                enable_cors: true,
            },
            logging: LoggingConfig {
                level: "debug".to_string(),
                enable_access_logs: true,
                enable_error_logs: true,
                log_format: "text".to_string(),
                log_file: None,
            },
        }
    }

    /// Print configuration summary for deployment verification
    pub fn print_summary(&self) {
        println!("🦀 CONSERVATIVE RUST API CONFIGURATION");
        println!("=====================================");
        println!(
            "🗄️  Database: {}:{}",
            self.database.host, self.database.port
        );
        println!(
            "📊 Database Pool: {}-{} connections",
            self.database.min_connections, self.database.max_connections
        );
        println!(
            "⏱️  Connection Timeout: {}ms",
            self.database.connection_timeout_ms
        );
        println!("⚡ Cache L1: {} entries", self.cache.l1_max_entries);
        println!("💾 Cache L2: {} entries", self.cache.l2_max_entries);
        println!("🚀 SIMD Enabled: {}", self.performance.enable_simd);
        println!(
            "🔥 Parallel Processing: {}",
            self.performance.enable_parallel
        );
        println!("🌐 Server: {}:{}", self.server.host, self.server.port);
        println!("📈 Max Server Connections: {}", self.server.max_connections);
        println!("⏱️  Request Timeout: {}s", self.server.timeout_seconds);
        println!("📋 Log Level: {}", self.logging.level);
        println!("=====================================");
    }

    pub fn validate(&self) -> Result<()> {
        if self.database.host.is_empty() {
            return Err(anyhow::anyhow!("Database host cannot be empty"));
        }
        if self.database.max_connections < self.database.min_connections {
            return Err(anyhow::anyhow!(
                "Max connections must be >= min connections"
            ));
        }
        if self.cache.l1_max_entries == 0 {
            return Err(anyhow::anyhow!("L1 cache max entries must be > 0"));
        }
        if self.performance.chunk_size == 0 {
            return Err(anyhow::anyhow!("Chunk size must be > 0"));
        }
        if self.server.port == 0 {
            return Err(anyhow::anyhow!("Server port must be valid"));
        }
        Ok(())
    }
}

impl DatabaseConfig {
    /// Get database URL for connection
    pub fn connection_string(&self) -> String {
        format!(
            "postgresql://{}:{}@{}:{}/{}",
            self.username, self.password, self.host, self.port, self.database
        )
    }

    /// Get connection options with SSL requirement
    pub fn connection_options(&self) -> sqlx::postgres::PgConnectOptions {
        let mut options = sqlx::postgres::PgConnectOptions::new()
            .host(&self.host)
            .port(self.port)
            .username(&self.username)
            .password(&self.password)
            .database(&self.database)
            .application_name("nuclear_risk_api")
            .statement_cache_capacity(self.statement_cache_capacity);

        if self.enable_ssl {
            options = options.ssl_mode(sqlx::postgres::PgSslMode::Require);
        }

        // Conservative performance settings
        options = options.options([
            ("tcp_nodelay", "true"),
            ("tcp_user_timeout", "30000"), // Increased timeout
            ("statement_timeout", "60s"),  // Longer query timeout
            ("idle_in_transaction_session_timeout", "30s"),
            ("application_name", "nuclear_api"),
        ]);

        options
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_config_validation() {
        let config = AppConfig::production_optimized();
        assert!(config.validate().is_ok());
        assert_eq!(config.database.max_connections, 15);
    }

    #[test]
    fn test_development_config() {
        let config = AppConfig::development();
        assert!(config.validate().is_ok());
        assert_eq!(config.database.max_connections, 5);
    }
}
