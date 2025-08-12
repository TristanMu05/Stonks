//! Finnhub WebSocket client
//! Free real-time stock data with WebSocket access

use anyhow::{Result, anyhow};
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, error, warn};
use url::Url;
use shared::{MarketTick, Config};
use std::sync::Arc;
use chrono::{Utc, TimeZone};
use tokio::sync::broadcast::Sender;

pub struct FinnhubWebSocket {
    config: Arc<Config>,
    symbols: Vec<String>,
    sender: Sender<MarketTick>,
}

impl FinnhubWebSocket {
    /// Create new Finnhub WebSocket client
    pub fn new(config: Arc<Config>, symbols: Vec<String>, sender: Sender<MarketTick>) -> Self {
        Self { config, symbols, sender }
    }
    
    /// Connect and start receiving data
    pub async fn run(&self) -> Result<()> {
        info!("🔌 Connecting to Finnhub WebSocket...");
        
        // Construct WebSocket URL with API token
        let url = format!("wss://ws.finnhub.io?token={}", self.config.finnhub_api_key);
        let url = Url::parse(&url)?;
        
        info!("📡 Connecting to: wss://ws.finnhub.io");
        
        // Connect to WebSocket
        let (ws_stream, _) = connect_async(url).await
            .map_err(|e| anyhow!("Failed to connect: {}", e))?;
            
        info!("✅ Connected to Finnhub WebSocket");
        
        // Split into sender and receiver
        let (mut write, mut read) = ws_stream.split();
        
        // Subscribe to symbols
        self.subscribe_to_symbols(&mut write).await?;
        
        // Process incoming messages
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = self.process_message(&text).await {
                        error!("Error processing message: {}", e);
                    }
                }
                Ok(Message::Close(_)) => {
                    warn!("WebSocket connection closed");
                    break;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
                _ => {} // Ignore other message types
            }
        }
        
        warn!("WebSocket connection ended");
        Ok(())
    }
    
    /// Subscribe to trades for our symbols
    async fn subscribe_to_symbols(&self, write: &mut futures_util::stream::SplitSink<tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>, Message>) -> Result<()> {
        info!("📋 Subscribing to symbols: {:?}", self.symbols);
        
        for symbol in &self.symbols {
            let subscribe_msg = serde_json::json!({
                "type": "subscribe",
                "symbol": symbol
            });
            
            write.send(Message::Text(subscribe_msg.to_string())).await
                .map_err(|e| anyhow!("Failed to subscribe to {}: {}", symbol, e))?;
                
            info!("✅ Subscribed to {}", symbol);
            
            // Small delay to avoid rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        
        Ok(())
    }
    
    /// Process incoming WebSocket message
    async fn process_message(&self, text: &str) -> Result<()> {
        // Parse JSON message
        let message: Value = serde_json::from_str(text)?;
        
        match message["type"].as_str() {
            Some("trade") => {
                // Trade data array
                if let Some(data) = message["data"].as_array() {
                    for trade in data {
                        if let Ok(tick) = self.parse_trade_message(trade) {
                            let _ = self.sender.send(tick.clone());
                            info!("💰 {}: ${:.4} (vol: {}) [{}]", 
                                  tick.symbol, tick.price, tick.volume, tick.exchange);
                        }
                    }
                }
            }
            Some("ping") => {
                // Heartbeat - connection is alive
                info!("💓 Heartbeat received");
            }
            _ => {
                info!("📨 Other message: {}", text);
            }
        }
        
        Ok(())
    }
    
    /// Parse trade message into MarketTick
    fn parse_trade_message(&self, trade: &Value) -> Result<MarketTick> {
        let symbol = trade["s"].as_str()
            .ok_or_else(|| anyhow!("Missing symbol"))?;
        let price = trade["p"].as_f64()
            .ok_or_else(|| anyhow!("Missing price"))?;
        let volume = trade["v"].as_u64()
            .ok_or_else(|| anyhow!("Missing volume"))?;
        let timestamp_ms = trade["t"].as_u64()
            .ok_or_else(|| anyhow!("Missing timestamp"))?;
            
        // Convert timestamp from milliseconds to DateTime<Utc>
        let timestamp = Utc
            .timestamp_millis_opt(timestamp_ms as i64)
            .single()
            .ok_or_else(|| anyhow!("Invalid timestamp"))?;
        
        Ok(MarketTick {
            id: uuid::Uuid::new_v4(),
            symbol: symbol.to_string(),
            price,
            volume,
            timestamp,
            exchange: "Finnhub".to_string(),
        })
    }
}

