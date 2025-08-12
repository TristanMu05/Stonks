use std::time::Duration;
use anyhow::Result;
use tracing::info;
use rand::{Rng, rngs::ThreadRng};
use tokio::sync::broadcast::Sender;
use shared::MarketTick;
use chrono::Utc;

pub struct MarketSimulator {
    symbols: Vec<String>,
    rng: ThreadRng,
    sender: Sender<MarketTick>,
}

impl MarketSimulator {
    pub fn new(symbols: Vec<String>, sender: Sender<MarketTick>) -> Self {
        Self { symbols, rng: rand::thread_rng(), sender }
    }

    pub async fn run(&mut self) -> Result<()> {
        info!("🧪 Starting market simulator...");
        loop {
            for s in &self.symbols {
                let price: f64 = self.rng.gen_range(50.0..500.0);
                let volume: u64 = self.rng.gen_range(1..1000);
                let tick = MarketTick {
                    id: uuid::Uuid::new_v4(),
                    symbol: s.clone(),
                    price,
                    volume,
                    timestamp: Utc::now(),
                    exchange: "Simulator".to_string(),
                };
                let _ = self.sender.send(tick.clone());
                info!("🎭 {}: ${:.2} (vol: {}) [Simulator]", s, price, volume);
            }
            tokio::time::sleep(Duration::from_millis(750)).await;
        }
    }
}

