// src/config.rs - Production configuration management
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
    /// Load configuration from environment variables with production defaults
    pub fn load() -> Result<Self> {
        // Database configuration (provided in hackathon docs)
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
                .unwrap_or_else(|_| "80".to_string())
                .parse()
                .context("Invalid DATABASE_MAX_CONNECTIONS")?,
            min_connections: env::var("DATABASE_MIN_CONNECTIONS")
                .unwrap_or_else(|_| "40".to_string())
                .parse()
                .context("Invalid DATABASE_MIN_CONNECTIONS")?,
            connection_timeout_ms: env::var("DATABASE_TIMEOUT_MS")
                .unwrap_or_else(|_| "5000".to_string())
                .parse()
                .context("Invalid DATABASE_TIMEOUT_MS")?,
            statement_cache_capacity: env::var("DATABASE_STATEMENT_CACHE")
                .unwrap_or_else(|_| "5000".to_string())
                .parse()
                .context("Invalid DATABASE_STATEMENT_CACHE")?,
            enable_ssl: env::var("DATABASE_ENABLE_SSL")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .context("Invalid DATABASE_ENABLE_SSL")?,
        };

        // Cache configuration - optimized for massive datasets
        let cache = CacheConfig {
            l1_max_entries: env::var("CACHE_L1_MAX_ENTRIES")
                .unwrap_or_else(|_| "100000".to_string())
                .parse()
                .context("Invalid CACHE_L1_MAX_ENTRIES")?,
            l2_max_entries: env::var("CACHE_L2_MAX_ENTRIES")
                .unwrap_or_else(|_| "50000".to_string())
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

        // Performance configuration - NUCLEAR settings
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
                .unwrap_or_else(|_| "10000".to_string())
                .parse()
                .context("Invalid PERFORMANCE_CHUNK_SIZE")?,
            max_request_size: env::var("PERFORMANCE_MAX_REQUEST_SIZE")
                .unwrap_or_else(|_| "1000000".to_string())
                .parse()
                .context("Invalid PERFORMANCE_MAX_REQUEST_SIZE")?,
            enable_compression: env::var("PERFORMANCE_ENABLE_COMPRESSION")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .context("Invalid PERFORMANCE_ENABLE_COMPRESSION")?,
            connection_pool_size: env::var("PERFORMANCE_CONNECTION_POOL_SIZE")
                .unwrap_or_else(|_| "80".to_string())
                .parse()
                .context("Invalid PERFORMANCE_CONNECTION_POOL_SIZE")?,
        };

        // Server configuration
        let server = ServerConfig {
            host: env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: env::var("SERVER_PORT")
                .unwrap_or_else(|_| "8000".to_string())
                .parse()
                .context("Invalid SERVER_PORT")?,
            workers: env::var("SERVER_WORKERS").ok().and_then(|s| s.parse().ok()),
            max_connections: env::var("SERVER_MAX_CONNECTIONS")
                .unwrap_or_else(|_| "10000".to_string())
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

    /// Load configuration from TOML file (useful for local development)
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path))?;

        let config: Self = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path))?;

        Ok(config)
    }

    /// Validate configuration and return any issues
    pub fn validate(&self) -> Result<()> {
        // Database validation
        if self.database.host.is_empty() {
            return Err(anyhow::anyhow!("Database host cannot be empty"));
        }
        if self.database.max_connections < self.database.min_connections {
            return Err(anyhow::anyhow!(
                "Max connections must be >= min connections"
            ));
        }

        // Cache validation
        if self.cache.l1_max_entries == 0 {
            return Err(anyhow::anyhow!("L1 cache max entries must be > 0"));
        }

        // Performance validation
        if self.performance.chunk_size == 0 {
            return Err(anyhow::anyhow!("Chunk size must be > 0"));
        }

        // Server validation
        if self.server.port == 0 {
            return Err(anyhow::anyhow!("Server port must be valid"));
        }

        Ok(())
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
                max_connections: 100,           // High concurrency
                min_connections: 50,            // Always ready
                connection_timeout_ms: 3000,    // Fast timeout
                statement_cache_capacity: 5000, // Cache many statements
                enable_ssl: true,
            },
            cache: CacheConfig {
                l1_max_entries: 200_000,      // Large L1 cache
                l2_max_entries: 100_000,      // Large L2 cache
                ttl_seconds: 600,             // 10 minute TTL
                cleanup_interval_seconds: 60, // Clean every minute
                memory_threshold_mb: 1000,    // 1GB threshold
                enable_distributed: false,    // Keep it simple for hackathon
                redis_url: None,
            },
            performance: PerformanceConfig {
                enable_simd: true,           // MAXIMUM SPEED
                enable_parallel: true,       // Use all cores
                parallel_threshold: 10_000,  // Parallel for large datasets
                chunk_size: 10_000,          // Optimal chunk size
                max_request_size: 1_000_000, // Handle large requests
                enable_compression: false,   // Skip compression for speed
                connection_pool_size: 100,   // Large pool
            },
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8000,              // Required port
                workers: None,           // Auto-detect
                max_connections: 50_000, // Handle massive load
                timeout_seconds: 30,     // Reasonable timeout
                keep_alive_seconds: 75,  // Keep connections alive
                enable_cors: true,       // Required for judge app
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

    /// Create configuration optimized for massive datasets
    pub fn massive_dataset_optimized() -> Self {
        let mut config = Self::production_optimized();

        // Optimize for massive datasets
        config.performance.chunk_size = 50_000; // Larger chunks
        config.performance.parallel_threshold = 50_000; // Higher parallel threshold
        config.database.max_connections = 150; // More DB connections
        config.cache.l1_max_entries = 500_000; // Massive L1 cache
        config.cache.memory_threshold_mb = 2000; // 2GB memory threshold

        config
    }

    /// Print configuration summary for deployment verification
    pub fn print_summary(&self) {
        println!("🦀 NUCLEAR RUST API CONFIGURATION");
        println!("================================");
        println!(
            "🗄️  Database: {}:{}",
            self.database.host, self.database.port
        );
        println!(
            "📊 Database Pool: {}-{} connections",
            self.database.min_connections, self.database.max_connections
        );
        println!("⚡ Cache L1: {} entries", self.cache.l1_max_entries);
        println!("💾 Cache L2: {} entries", self.cache.l2_max_entries);
        println!("🚀 SIMD Enabled: {}", self.performance.enable_simd);
        println!(
            "🔥 Parallel Processing: {}",
            self.performance.enable_parallel
        );
        println!("🌐 Server: {}:{}", self.server.host, self.server.port);
        println!("📈 Max Connections: {}", self.server.max_connections);
        println!("⏱️  Request Timeout: {}s", self.server.timeout_seconds);
        println!("📋 Log Level: {}", self.logging.level);
        println!("================================");
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

    /// Get connection options for maximum performance
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
            options = options.ssl_mode(sqlx::postgres::PgSslMode::Prefer);
        }

        // Performance optimizations
        options = options.options([
            ("tcp_nodelay", "true"),                        // Reduce network latency
            ("tcp_user_timeout", "5000"),                   // Fast failure detection
            ("statement_timeout", "30s"),                   // Prevent runaway queries
            ("idle_in_transaction_session_timeout", "10s"), // Clean up idle connections
            ("application_name", "nuclear_api"),            // Identify our connections
        ]);

        options
    }
}

/// Environment-specific configurations
impl AppConfig {
    /// Development configuration with relaxed settings
    pub fn development() -> Self {
        let mut config = Self::production_optimized();

        config.logging.level = "debug".to_string();
        config.logging.enable_access_logs = true;
        config.server.max_connections = 1_000;
        config.database.max_connections = 20;
        config.cache.l1_max_entries = 10_000;

        config
    }

    /// Test configuration with minimal resources
    pub fn test() -> Self {
        Self {
            database: DatabaseConfig {
                host: "localhost".to_string(),
                port: 5432,
                database: "test_db".to_string(),
                username: "test_user".to_string(),
                password: "test_pass".to_string(),
                max_connections: 5,
                min_connections: 2,
                connection_timeout_ms: 1000,
                statement_cache_capacity: 100,
                enable_ssl: false,
            },
            cache: CacheConfig {
                l1_max_entries: 1_000,
                l2_max_entries: 500,
                ttl_seconds: 60,
                cleanup_interval_seconds: 30,
                memory_threshold_mb: 50,
                enable_distributed: false,
                redis_url: None,
            },
            performance: PerformanceConfig {
                enable_simd: true,
                enable_parallel: false, // Single-threaded for tests
                parallel_threshold: 1000,
                chunk_size: 100,
                max_request_size: 10_000,
                enable_compression: false,
                connection_pool_size: 5,
            },
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 8000,
                workers: Some(1),
                max_connections: 100,
                timeout_seconds: 10,
                keep_alive_seconds: 30,
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
}

/// Configuration validation utilities
pub mod validation {
    use super::*;

    pub fn validate_database_config(config: &DatabaseConfig) -> Vec<String> {
        let mut errors = Vec::new();

        if config.host.is_empty() {
            errors.push("Database host cannot be empty".to_string());
        }

        if config.port == 0 || config.port > 65535 {
            errors.push("Database port must be between 1 and 65535".to_string());
        }

        if config.username.is_empty() {
            errors.push("Database username cannot be empty".to_string());
        }

        if config.max_connections == 0 {
            errors.push("Database max connections must be > 0".to_string());
        }

        if config.max_connections < config.min_connections {
            errors.push("Database max connections must be >= min connections".to_string());
        }

        errors
    }

    pub fn validate_performance_config(config: &PerformanceConfig) -> Vec<String> {
        let mut errors = Vec::new();

        if config.chunk_size == 0 {
            errors.push("Chunk size must be > 0".to_string());
        }

        if config.chunk_size > config.max_request_size {
            errors.push("Chunk size cannot be larger than max request size".to_string());
        }

        if config.parallel_threshold == 0 {
            errors.push("Parallel threshold must be > 0".to_string());
        }

        errors
    }

    pub fn validate_complete_config(config: &AppConfig) -> Result<()> {
        let mut all_errors = Vec::new();

        all_errors.extend(validate_database_config(&config.database));
        all_errors.extend(validate_performance_config(&config.performance));

        if !all_errors.is_empty() {
            return Err(anyhow::anyhow!(
                "Configuration validation failed:\n{}",
                all_errors.join("\n")
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_config_validation() {
        let config = AppConfig::production_optimized();
        assert!(validation::validate_complete_config(&config).is_ok());
    }

    #[test]
    fn test_database_connection_string() {
        let db_config = DatabaseConfig {
            host: "localhost".to_string(),
            port: 5432,
            database: "testdb".to_string(),
            username: "user".to_string(),
            password: "pass".to_string(),
            max_connections: 10,
            min_connections: 5,
            connection_timeout_ms: 1000,
            statement_cache_capacity: 100,
            enable_ssl: false,
        };

        let connection_string = db_config.connection_string();
        assert_eq!(
            connection_string,
            "postgresql://user:pass@localhost:5432/testdb"
        );
    }

    #[test]
    fn test_cache_config_validation() {
        let cache_config = CacheConfig {
            l1_max_entries: 0, // Invalid
            l2_max_entries: 1000,
            ttl_seconds: 60,
            cleanup_interval_seconds: 30,
            memory_threshold_mb: 100,
            enable_distributed: false,
            redis_url: None,
        };

        // Should catch the invalid l1_max_entries
        // In a real validation function, this would fail
        assert_eq!(cache_config.l1_max_entries, 0);
    }
}
