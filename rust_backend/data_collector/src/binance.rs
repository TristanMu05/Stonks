//! Binance WebSocket client for crypto trades (24/7)

use anyhow::{anyhow, Result};
use futures_util::{StreamExt};
use serde_json::Value;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn, error};
use shared::MarketTick;
use chrono::{Utc, TimeZone};
use tokio::sync::broadcast::Sender;

pub struct BinanceWebSocket {
    symbols: Vec<String>, // e.g., ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT"]
    sender: Sender<MarketTick>,
}

impl BinanceWebSocket {
    pub fn new(symbols: Vec<String>, sender: Sender<MarketTick>) -> Self {
        Self { symbols, sender }
    }

    fn build_stream_url(&self) -> Result<String> {
        // multi-stream: /stream?streams=btcusdt@trade/ethusdt@trade
        let mut streams: Vec<String> = Vec::new();
        for s in &self.symbols {
            if let Some(pair) = s.strip_prefix("BINANCE:") {
                streams.push(format!("{}@trade", pair.to_ascii_lowercase()));
            }
        }
        if streams.is_empty() {
            return Err(anyhow!("No BINANCE:* symbols provided"));
        }
        Ok(format!(
            "wss://stream.binance.com:9443/stream?streams={}",
            streams.join("/")
        ))
    }

    pub async fn run(&self) -> Result<()> {
        let url = self.build_stream_url()?;
        info!("🔌 Connecting to Binance WebSocket: {}", url);
        let (ws_stream, _) = connect_async(&url).await?;
        info!("✅ Connected to Binance WebSocket");

        let (_write, mut read) = ws_stream.split();

        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = self.process_message(&text).await {
                        warn!("Binance message error: {}", e);
                    }
                }
                Ok(Message::Close(_)) => {
                    warn!("Binance WebSocket closed");
                    break;
                }
                Err(e) => {
                    error!("Binance WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    async fn process_message(&self, text: &str) -> Result<()> {
        // Multi-stream wrapper: { "stream": "btcusdt@trade", "data": { ... } }
        let v: Value = serde_json::from_str(text)?;
        let data = v.get("data").unwrap_or(&v);

        if data.get("e").and_then(|e| e.as_str()) == Some("trade") {
            let symbol = data.get("s").and_then(|s| s.as_str()).unwrap_or("");
            let price = data.get("p").and_then(|p| p.as_str()).and_then(|s| s.parse::<f64>().ok())
                .ok_or_else(|| anyhow!("missing price"))?;
            let qty = data.get("q").and_then(|p| p.as_str()).and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0);
            let event_time = data.get("E").and_then(|t| t.as_i64()).unwrap_or_else(|| chrono::Utc::now().timestamp_millis());

            let ts = Utc
                .timestamp_millis_opt(event_time)
                .single()
                .ok_or_else(|| anyhow!("invalid timestamp"))?;

            let tick = MarketTick {
                id: uuid::Uuid::new_v4(),
                symbol: format!("BINANCE:{}", symbol),
                price,
                volume: qty as u64,
                timestamp: ts,
                exchange: "Binance".to_string(),
            };

            let _ = self.sender.send(tick);
        }

        Ok(())
    }
}


