// src/main.rs - Nuclear Performance Entry Point
use axum::{Router, middleware, routing::get};
use std::sync::Arc;
use std::time::Duration;
use tower_http::{compression::CompressionLayer, cors::CorsLayer, trace::TraceLayer};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod cache;
mod calculations;
mod challenge_handlers;
mod config;
mod database;
mod handlers;
mod models;
mod monitoring;

use crate::{
    cache::NuclearCache, config::AppConfig, database::Database, 
    handlers::*, challenge_handlers::*, monitoring::performance_middleware,
};

#[derive(Clone)]
pub struct AppState {
    pub db: Database,
    pub cache: NuclearCache,
    pub config: AppConfig,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize high-performance allocator
    #[cfg(feature = "mimalloc")]
    {
        mimalloc::MiMalloc;
    }

    // Setup tracing for production monitoring
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "risk_analytics_api=info,sqlx=warn,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("🦀 Starting NUCLEAR PERFORMANCE Risk Analytics API...");

    // Load configuration
    let config = AppConfig::load()?;
    info!("📋 Configuration loaded successfully");

    // Initialize database with massive dataset optimizations
    let database = Database::new(&config).await?;
    info!("🗄️  Database connection pool established");

    // Initialize multi-tier cache system
    let cache = NuclearCache::new().await?;
    info!("⚡ Nuclear cache system initialized");

    // Pre-warm cache for common queries
    tokio::spawn({
        let db = database.clone();
        let cache = cache.clone();
        async move {
            info!("🔥 Pre-warming cache for optimal performance...");
            cache.warm_cache_intelligently(&db).await.unwrap_or_else(|e| {
                warn!("Cache warming failed: {}", e);
            });
            info!("✅ Cache warming completed");
        }
    });

    // Create application state
    let state = Arc::new(AppState {
        db: database,
        cache,
        config,
    });

    // Build the nuclear-powered router with exact challenge endpoints
    let app = Router::new()
        // Challenge specification endpoints (exact paths from Swagger)
        .route("/portfolio-price", get(get_portfolio_price))
        .route("/daily-return", get(get_daily_return))
        .route("/cumulative-return", get(get_cumulative_return))
        .route("/daily-volatility", get(get_daily_volatility))
        .route("/correlation", get(get_correlation))
        .route("/tracking-error", get(get_tracking_error))
        // Health and monitoring endpoints
        .route("/health", get(health_check))
        .route("/metrics", get(get_metrics))
        .route("/stats", get(get_performance_stats))
        // Middleware stack for maximum performance
        .layer(middleware::from_fn(performance_middleware))
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        // Application state
        .with_state(state.clone());

    // Create optimized listener using config
    let bind_address = format!("{}:{}", state.config.server.host, state.config.server.port);
    let listener = tokio::net::TcpListener::bind(&bind_address).await?;

    // Configure socket options for maximum performance
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = listener.as_raw_fd();

        // Enable TCP_NODELAY (disable Nagle's algorithm)
        unsafe {
            let optval: libc::c_int = 1;
            libc::setsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_NODELAY,
                &optval as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );

            // Set SO_REUSEADDR
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_REUSEADDR,
                &optval as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
        }
    }

    info!("🚀 Nuclear-powered API launching on http://0.0.0.0:8000");
    info!("💥 Ready to DOMINATE the competition!");
    info!("🎯 Endpoints available:");
    info!("   - GET /health");
    info!("   - GET /portfolio-price?portfolioId=X&date=YYYY-MM-DD");
    info!("   - GET /daily-return?portfolioId=X&date=YYYY-MM-DD");
    info!("   - GET /cumulative-return?portfolioId=X&startDate=X&endDate=Y");
    info!("   - GET /daily-volatility?portfolioId=X&startDate=X&endDate=Y");
    info!("   - GET /correlation?portfolioId1=X&portfolioId2=Y&startDate=A&endDate=B");
    info!("   - GET /tracking-error?portfolioId=X&benchmarkId=Y&startDate=A&endDate=B");

    // Start the server with maximum performance settings
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("🛑 Received Ctrl+C, shutting down gracefully...");
        },
        _ = terminate => {
            info!("🛑 Received terminate signal, shutting down gracefully...");
        },
    }
}
