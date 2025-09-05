use sqlx::postgres::{PgConnectOptions, PgConnection};
use sqlx::{Connection, Row};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = PgConnectOptions::new()
        .host("aws-0-ap-southeast-1.pooler.supabase.com")
        .port(5432)
        .username("team9dbuser.jdcgkhwtrsdhyysagkwb")
        .password("e5ci7swfjroiqs4f")
        .database("postgres")
        .ssl_mode(sqlx::postgres::PgSslMode::Require);

    println!("Connecting to database...");
    let mut conn = PgConnection::connect_with(&options).await?;
    
    println!("✅ Connected successfully!");
    
    // Check what tables exist
    println!("\n🔍 Checking available tables...");
    let tables: Vec<(String,)> = sqlx::query_as(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )
    .fetch_all(&mut conn)
    .await?;
    
    if tables.is_empty() {
        println!("❌ No tables found in the public schema");
    } else {
        println!("📋 Available tables:");
        for (table_name,) in &tables {
            println!("  - {}", table_name);
        }
    }
    
    // Check if portfolios table exists and what data it has
    if tables.iter().any(|(name,)| name == "portfolios") {
        println!("\n📊 Checking portfolios table...");
        let portfolio_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM portfolios")
            .fetch_one(&mut conn)
            .await?;
        println!("  - Total portfolios: {}", portfolio_count.0);
        
        // Check what columns exist in portfolios table
        println!("\n📝 Portfolios table columns:");
        let columns: Vec<(String, String)> = sqlx::query_as(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'portfolios' ORDER BY ordinal_position"
        )
        .fetch_all(&mut conn)
        .await?;
        
        for (col_name, col_type) in columns {
            println!("  - {}: {}", col_name, col_type);
        }
        
        if portfolio_count.0 > 0 {
            println!("\n📝 Sample portfolio data:");
            let sample_portfolios: Vec<(i32, String, String, Option<i32>)> = sqlx::query_as(
                "SELECT portfolio_id, portfolio_name, portfolio_type, parent_portfolio_id FROM portfolios LIMIT 5"
            )
            .fetch_all(&mut conn)
            .await?;
            
            for (id, name, ptype, parent) in sample_portfolios {
                println!("  - ID: {}, Name: {}, Type: {}, Parent: {:?}", id, name, ptype, parent);
            }
        }
    } else {
        println!("❌ portfolios table not found");
    }
    
    // Check if holdings table exists
    if tables.iter().any(|(name,)| name == "holdings") {
        println!("\n💰 Checking holdings table...");
        let holdings_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM holdings")
            .fetch_one(&mut conn)
            .await?;
        println!("  - Total holdings: {}", holdings_count.0);
        
        // Check available dates in holdings
        println!("\n📅 Available dates in holdings (first 5):");
        let sample_dates: Vec<(chrono::NaiveDate,)> = sqlx::query_as(
            "SELECT DISTINCT date FROM holdings ORDER BY date LIMIT 5"
        )
        .fetch_all(&mut conn)
        .await?;
        
        for (date,) in sample_dates {
            println!("  - {}", date);
        }
        
        // Test the actual query used by the API for portfolio 3
        println!("\n🧪 Testing portfolio price query for portfolio 3 on 2024-07-27:");
        let test_result: Option<(Option<f64>,)> = sqlx::query_as(
            r#"
                SELECT SUM(h.quantity * p.closing_price) as total_value
                FROM holdings h
                JOIN prices p ON h.symbol = p.symbol AND h.date = p.date
                WHERE h.portfolio_id = $1 AND h.date = $2
            "#
        )
        .bind(3i32)
        .bind(chrono::NaiveDate::from_ymd_opt(2024, 7, 27).unwrap())
        .fetch_optional(&mut conn)
        .await?;
        
        match test_result {
            Some((Some(value),)) => println!("  ✅ Portfolio 3 value on 2024-07-27: {}", value),
            Some((None,)) => println!("  ❌ Query returned NULL (no matching records)"),
            None => println!("  ❌ Query returned no rows"),
        }
        
        // Debug: Check if portfolio 3 has any holdings
        println!("\n🔍 Checking holdings for portfolio 3:");
        let holdings_for_p3: Vec<(String, chrono::NaiveDate, f64)> = sqlx::query_as(
            "SELECT symbol, date, quantity FROM holdings WHERE portfolio_id = 3 LIMIT 3"
        )
        .fetch_all(&mut conn)
        .await?;
        
        if holdings_for_p3.is_empty() {
            println!("  ❌ No holdings found for portfolio 3");
        } else {
            println!("  📋 Sample holdings for portfolio 3:");
            for (symbol, date, qty) in holdings_for_p3 {
                println!("    - Symbol: {}, Date: {}, Quantity: {}", symbol, date, qty);
            }
        }
        
        // Debug: Check if prices table has any data for those dates/symbols
        println!("\n🔍 Checking sample prices data:");
        let sample_prices: Vec<(String, chrono::NaiveDate, f64)> = sqlx::query_as(
            "SELECT symbol, date, closing_price FROM prices LIMIT 3"
        )
        .fetch_all(&mut conn)
        .await?;
        
        for (symbol, date, price) in sample_prices {
            println!("  - Symbol: {}, Date: {}, Price: {}", symbol, date, price);
        }
        
        // Find a working combination
        println!("\n🎯 Finding a portfolio and date with both holdings and prices:");
        let working_combo: Option<(i32, chrono::NaiveDate, String)> = sqlx::query_as(
            r#"
            SELECT DISTINCT h.portfolio_id, h.date, h.symbol
            FROM holdings h
            INNER JOIN prices p ON h.symbol = p.symbol AND h.date = p.date
            LIMIT 1
            "#
        )
        .fetch_optional(&mut conn)
        .await?;
        
        match working_combo {
            Some((portfolio_id, date, symbol)) => {
                println!("  ✅ Found working combination: Portfolio {}, Date {}, Symbol {}", 
                         portfolio_id, date, symbol);
                
                // Test the actual portfolio price calculation
                let price_test: Option<(Option<f64>,)> = sqlx::query_as(
                    r#"
                        SELECT SUM(h.quantity * p.closing_price) as total_value
                        FROM holdings h
                        JOIN prices p ON h.symbol = p.symbol AND h.date = p.date
                        WHERE h.portfolio_id = $1 AND h.date = $2
                    "#
                )
                .bind(portfolio_id)
                .bind(date)
                .fetch_optional(&mut conn)
                .await?;
                
                match price_test {
                    Some((Some(value),)) => println!("  💰 Portfolio {} value on {}: ${:.2}", 
                                                   portfolio_id, date, value),
                    _ => println!("  ❌ Still no value calculated"),
                }
            }
            None => println!("  ❌ No matching holdings and prices found"),
        }
    } else {
        println!("❌ holdings table not found");
    }
    
    // Check if prices table exists
    if tables.iter().any(|(name,)| name == "prices") {
        println!("\n💵 Checking prices table...");
        let prices_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM prices")
            .fetch_one(&mut conn)
            .await?;
        println!("  - Total price records: {}", prices_count.0);
    } else {
        println!("❌ prices table not found");
    }
    
    // Check if benchmark table exists and analyze it thoroughly
    if tables.iter().any(|(name,)| name == "benchmark") {
        println!("\n📊 COMPREHENSIVE BENCHMARK TABLE ANALYSIS");
        println!("========================================");
        
        let benchmark_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM benchmark")
            .fetch_one(&mut conn)
            .await?;
        println!("  - Total benchmark records: {}", benchmark_count.0);
        
        // Check benchmark table columns
        println!("\n📝 Benchmark table columns:");
        let benchmark_columns: Vec<(String, String)> = sqlx::query_as(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'benchmark' ORDER BY ordinal_position"
        )
        .fetch_all(&mut conn)
        .await?;
        
        for (col_name, col_type) in benchmark_columns {
            println!("  - {}: {}", col_name, col_type);
        }
        
        if benchmark_count.0 > 0 {
            println!("\n📝 Sample benchmark data (first 10 rows):");
            let sample_benchmarks: Vec<(i32, chrono::NaiveDate, i32)> = sqlx::query_as(
                "SELECT bmk_id, date, bmk_returns FROM benchmark ORDER BY bmk_id, date LIMIT 10"
            )
            .fetch_all(&mut conn)
            .await?;
            
            for (bmk_id, date, returns) in sample_benchmarks {
                println!("  - Benchmark ID: {}, Date: {}, Returns: {}", bmk_id, date, returns);
            }
            
            // Get date range for benchmarks
            println!("\n📅 Benchmark date range:");
            let benchmark_date_range: (Option<chrono::NaiveDate>, Option<chrono::NaiveDate>) = sqlx::query_as(
                "SELECT MIN(date), MAX(date) FROM benchmark"
            )
            .fetch_one(&mut conn)
            .await?;
            
            if let (Some(min_date), Some(max_date)) = benchmark_date_range {
                println!("  - Date range: {} to {}", min_date, max_date);
                println!("  - Total days: {}", (max_date - min_date).num_days() + 1);
            }
            
            // Get unique benchmark IDs
            println!("\n🔢 Unique benchmark IDs:");
            let unique_benchmarks: Vec<(i32,)> = sqlx::query_as(
                "SELECT DISTINCT bmk_id FROM benchmark ORDER BY bmk_id"
            )
            .fetch_all(&mut conn)
            .await?;
            
            for (bmk_id,) in unique_benchmarks {
                println!("  - Benchmark ID: {}", bmk_id);
            }
        }
    } else {
        println!("\n❌ benchmark table not found");
    }
    
    // Comprehensive analysis of ALL tables
    println!("\n🔍 COMPREHENSIVE DATABASE ANALYSIS");
    println!("========================================");
    
    for (table_name,) in &tables {
        println!("\n📊 TABLE: {}", table_name);
        println!("----------------------------------------");
        
        // Get row count
        let row_count: (i64,) = sqlx::query_as(&format!("SELECT COUNT(*) FROM {}", table_name))
            .fetch_one(&mut conn)
            .await?;
        println!("  📈 Total rows: {}", row_count.0);
        
        // Get columns
        let table_columns: Vec<(String, String, String)> = sqlx::query_as(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = $1 ORDER BY ordinal_position"
        )
        .bind(table_name)
        .fetch_all(&mut conn)
        .await?;
        
        println!("  📋 Columns:");
        for (col_name, col_type, is_nullable) in table_columns {
            println!("    - {}: {} (nullable: {})", col_name, col_type, is_nullable);
        }
        
        if row_count.0 > 0 {
            // Show first few rows of each table for understanding data structure
            println!("  📝 Sample data (first 3 rows):");
            let sample_query = format!("SELECT * FROM {} LIMIT 3", table_name);
            let rows = sqlx::query(&sample_query)
                .fetch_all(&mut conn)
                .await?;
            
            for (i, row) in rows.iter().enumerate() {
                println!("    Row {}: {} columns", i + 1, row.len());
                // Note: We can't easily print all column values without knowing types
                // This gives us structure information
            }
        }
    }
    
    // Cross-table analysis for understanding relationships
    println!("\n🔗 CROSS-TABLE RELATIONSHIP ANALYSIS");
    println!("========================================");
    
    // Check which portfolios have holdings
    println!("\n📊 Portfolios with holdings data:");
    let portfolios_with_holdings: Vec<(i32, i64)> = sqlx::query_as(
        "SELECT h.portfolio_id, COUNT(*) as holding_count FROM holdings h GROUP BY h.portfolio_id ORDER BY holding_count DESC LIMIT 10"
    )
    .fetch_all(&mut conn)
    .await?;
    
    for (portfolio_id, count) in portfolios_with_holdings {
        println!("  - Portfolio {}: {} holdings", portfolio_id, count);
    }
    
    // Check date overlap between holdings and prices
    println!("\n📅 Date overlap analysis:");
    let holdings_dates: (Option<chrono::NaiveDate>, Option<chrono::NaiveDate>) = sqlx::query_as(
        "SELECT MIN(date), MAX(date) FROM holdings"
    )
    .fetch_one(&mut conn)
    .await?;
    
    let prices_dates: (Option<chrono::NaiveDate>, Option<chrono::NaiveDate>) = sqlx::query_as(
        "SELECT MIN(date), MAX(date) FROM prices"
    )
    .fetch_one(&mut conn)
    .await?;
    
    println!("  📊 Holdings date range: {:?} to {:?}", holdings_dates.0, holdings_dates.1);
    println!("  💰 Prices date range: {:?} to {:?}", prices_dates.0, prices_dates.1);
    
    // Check symbol overlap between holdings and prices
    println!("\n🔤 Symbol analysis:");
    let holdings_symbols: (i64,) = sqlx::query_as("SELECT COUNT(DISTINCT symbol) FROM holdings")
        .fetch_one(&mut conn)
        .await?;
    let prices_symbols: (i64,) = sqlx::query_as("SELECT COUNT(DISTINCT symbol) FROM prices")
        .fetch_one(&mut conn)
        .await?;
    
    println!("  📊 Distinct symbols in holdings: {}", holdings_symbols.0);
    println!("  💰 Distinct symbols in prices: {}", prices_symbols.0);
    
    // Check common symbols
    let common_symbols: (i64,) = sqlx::query_as(
        "SELECT COUNT(DISTINCT h.symbol) FROM holdings h INNER JOIN prices p ON h.symbol = p.symbol"
    )
    .fetch_one(&mut conn)
    .await?;
    
    println!("  🔗 Common symbols (holdings ∩ prices): {}", common_symbols.0);
    
    println!("\n✅ COMPREHENSIVE SCHEMA ANALYSIS COMPLETE!");
    println!("========================================");
    Ok(())
}