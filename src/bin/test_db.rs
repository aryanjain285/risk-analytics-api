use sqlx::postgres::{PgConnectOptions, PgConnection};
use sqlx::Connection;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = PgConnectOptions::new()
        .host("aws-0-ap-southeast-1.pooler.supabase.com")
        .port(5432)
        .username("team9dbuser.jdcgkhwtrsdhyysagkwb")
        .password("e5ci7swfjroiqs4f")
        .database("postgres")
        .ssl_mode(sqlx::postgres::PgSslMode::Require);

    println!("Testing connection...");
    let _conn = PgConnection::connect_with(&options).await?;
    println!("Connection successful!");
    Ok(())
}
