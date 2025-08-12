//! Polygon.io WebSocket client
//! Connects to real-time market data stream

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

pub struct PolygonWebSocket {
    config: Arc<Config>,
    symbols: Vec<String>,
    sender: Sender<MarketTick>,
}

impl PolygonWebSocket {
    /// Create new Polygon WebSocket client
    pub fn new(config: Arc<Config>, symbols: Vec<String>, sender: Sender<MarketTick>) -> Self {
        Self { config, symbols, sender }
    }
    
    /// Connect and start receiving data
    pub async fn run(&self) -> Result<()> {
        info!("🔌 Connecting to Polygon.io WebSocket...");
        
        // Construct WebSocket URL (auth is sent as a message, not query param)
        let url = Url::parse("wss://socket.polygon.io/stocks")?;
        info!("📡 Connecting to: {}", url);
        
        // Connect to WebSocket
        let (ws_stream, _) = connect_async(url).await
            .map_err(|e| anyhow!("Failed to connect: {}", e))?;
            
        info!("✅ Connected to Polygon.io WebSocket");
        
        // Split into sender and receiver
        let (mut write, mut read) = ws_stream.split();
        
        // Authenticate and then subscribe to symbols
        self.authenticate(&mut write).await?;
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

    async fn authenticate(&self, write: &mut futures_util::stream::SplitSink<tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>, Message>) -> Result<()> {
        info!("🔐 Authenticating with Polygon...");

        let auth_msg = serde_json::json!({
            "action": "auth",
            "params": self.config.polygon_api_key
        });

        write
            .send(Message::Text(auth_msg.to_string()))
            .await
            .map_err(|e| anyhow!("Failed to send auth: {}", e))?;

        info!("✅ Auth message sent");
        Ok(())
    }

    async fn subscribe_to_symbols(&self, write: &mut futures_util::stream::SplitSink<tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>, Message>) -> Result<()> {
        info!("🔍 Subscribing to symbols: {:?}", self.symbols);

        // create subscription message
        let subscribe_msg = serde_json::json!({
            "action": "subscribe",
            "params": format!("T.{}", self.symbols.join(",T."))
        });

        // send subscription
        write.send(Message::Text(subscribe_msg.to_string())).await
            .map_err(|e| anyhow!("Failed to send subscription: {}", e))?;

        info!("✅ Subscribed to symbols: {:?}", self.symbols);
        Ok(())
    }

    /// Process incoming messages
    async fn process_message(&self, text: &str) -> Result<()> {
        // parse JSON message
        let messages: Vec<Value> = serde_json::from_str(text)?;
        for message in messages {
            match message["ev"].as_str() {
                Some("T") => {
                    // trade message - conver to market tick
                    if let Ok(tick) = self.parse_trade_message(&message){
                        info!("📊 Trade: {} @ ${:.2} (vol: {})", 
                              tick.symbol, tick.price, tick.volume);

                        let _ = self.sender.send(tick.clone());
                    }
                }
                Some("status") => {
                    // connection status message
                    let status = message["message"].as_str().unwrap_or("Unknown");
                    info!("🔄 Connection status: {}", status);
                }
                _ => {
                    // unknown message type
                    warn!("Unknown message type: {}", text);
                }
            }
        }
        Ok(())
    }

    /// Parse trade message into MarketTick
    fn parse_trade_message(&self, message: &Value) -> Result<MarketTick> {
        let symbol = message["sym"].as_str()
            .ok_or_else(|| anyhow!("Missing symbol"))?;
        let price = message["p"].as_f64()
            .ok_or_else(|| anyhow!("Missing price"))?;
        let volume = message["s"].as_u64()
            .ok_or_else(|| anyhow!("Missing volume"))?;
        let timestamp_ms = message["t"].as_u64()
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
            exchange: "Polygon".to_string(),
        })
    }
}
