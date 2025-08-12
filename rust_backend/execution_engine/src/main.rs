//! Order execution engine
//! Places orders with brokers at lightning speed

use anyhow::Result;
use tracing::info;
use shared::{Config, Order, OrderSide};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    info!("⚡ Starting Execution Engine");
    
    let _config = Config::from_env()?;
    info!("✅ Configuration loaded");
    
    // Create a sample order
    let mut order = Order::market_order(
        "TSLA".to_string(),
        OrderSide::Buy,
        100
    );
    
    info!("📋 Created order: {:?}", order);
    
    // Simulate filling the order
    order.mark_filled();
    info!("✅ Order filled: {:?}", order);
    
    info!("🚀 Execution engine ready for trading!");
    
    Ok(())
}
