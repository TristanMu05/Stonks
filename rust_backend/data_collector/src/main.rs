//! High-performance market data collector
//! Connects to Finnhub for FREE real-time stock data

mod polygon;
mod finnhub;
mod simulator;
mod binance;

use anyhow::Result;
use tracing::{info, warn};
use shared::{Config, MarketTick};
use std::sync::Arc;
use finnhub::FinnhubWebSocket;
use simulator::MarketSimulator;
use polygon::PolygonWebSocket;
use binance::BinanceWebSocket;
use tokio::signal;
use tokio::sync::broadcast::{self, Sender};
use tokio::sync::RwLock;

// axum + sse
use axum::{Router, routing::get, extract::{State, Query}};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::http::HeaderMap;
use axum::Json;
use axum::http::Method;
use tower_http::cors::{Any, CorsLayer};
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use std::convert::Infallible;
use std::collections::{HashMap, VecDeque};
use serde::{Deserialize, Serialize};
use chrono::Duration;

#[derive(Clone)]
struct AppState {
    sender: Sender<MarketTick>,
    history: Arc<RwLock<HashMap<String, VecDeque<MarketTick>>>>,
}

const MAX_HISTORY_PER_SYMBOL: usize = 1000;

async fn sse_handler(
    State(app_state): State<Arc<AppState>>,
) -> (HeaderMap, Sse<impl tokio_stream::Stream<Item = std::result::Result<Event, Infallible>>>) {
    let rx = app_state.sender.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|res| {
        match res {
            Ok(tick) => match serde_json::to_string(&tick) {
                Ok(json) => Some(Ok(Event::default().data(json))),
                Err(_) => None,
            },
            Err(_) => None,
        }
    });

    let sse = Sse::new(stream).keep_alive(KeepAlive::new());

    let mut headers = HeaderMap::new();
    headers.insert(axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN, "*".parse().unwrap());
    headers.insert(axum::http::header::CACHE_CONTROL, "no-cache".parse().unwrap());

    (headers, sse)
}

#[derive(Deserialize)]
struct HistoryQuery {
    symbol: Option<String>,
    limit: Option<usize>,
}

async fn history_handler(
    State(app_state): State<Arc<AppState>>,
    Query(params): Query<HistoryQuery>,
) -> (HeaderMap, Json<Vec<MarketTick>>) {
    let mut headers = HeaderMap::new();
    headers.insert(axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN, "*".parse().unwrap());
    headers.insert(axum::http::header::CACHE_CONTROL, "no-cache".parse().unwrap());

    let history_map = app_state.history.read().await;

    let result: Vec<MarketTick> = if let Some(symbol) = params.symbol.as_ref() {
        let limit = params.limit.unwrap_or(500);
        history_map
            .get(symbol)
            .map(|dq| {
                let len = dq.len();
                let start = len.saturating_sub(limit);
                dq.iter().skip(start).cloned().collect::<Vec<_>>()
            })
            .unwrap_or_default()
    } else {
        // Return the latest one per symbol (flattened)
        history_map
            .values()
            .filter_map(|dq| dq.back().cloned())
            .collect::<Vec<_>>()
    };

    (headers, Json(result))
}

#[derive(Deserialize)]
struct ChartQuery {
    symbol: String,
    timeframe: String, // "1m" or "1d"
}

#[derive(Serialize)]
struct ChartPoint {
    ts: i64,   // epoch ms
    price: f64,
}

async fn chart_handler(
    State(app_state): State<Arc<AppState>>,
    Query(params): Query<ChartQuery>,
) -> (HeaderMap, Json<Vec<ChartPoint>>) {
    let mut headers = HeaderMap::new();
    headers.insert(axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN, "*".parse().unwrap());
    headers.insert(axum::http::header::CACHE_CONTROL, "no-cache".parse().unwrap());

    let history_map = app_state.history.read().await;
    let deque = match history_map.get(&params.symbol) {
        Some(dq) => dq,
        None => return (headers, Json(vec![])),
    };

    let now = chrono::Utc::now();
    let (cutoff, bucket_secs): (chrono::DateTime<chrono::Utc>, i64) = match params.timeframe.as_str() {
        "1m" => (now - Duration::seconds(60), 1),
        _ => (now - Duration::hours(24), 60), // default to 1d with per-minute buckets
    };

    let mut last_per_bucket: HashMap<i64, &MarketTick> = HashMap::new();
    for t in deque.iter().rev() { // newest first to keep last per bucket
        if t.timestamp < cutoff { break; }
        let ts_ms = t.timestamp.timestamp_millis();
        let bucket_start_sec = (ts_ms / 1000) / bucket_secs * bucket_secs;
        last_per_bucket.entry(bucket_start_sec).or_insert(t);
    }

    let mut buckets: Vec<i64> = last_per_bucket.keys().cloned().collect();
    buckets.sort_unstable();
    let points = buckets
        .into_iter()
        .filter_map(|sec| last_per_bucket.get(&sec).map(|tick| ChartPoint { ts: sec * 1000, price: tick.price }))
        .collect::<Vec<_>>();

    (headers, Json(points))
}

#[derive(Deserialize)]
struct CandleQuery {
    symbol: String,
    timeframe: String,
}

#[derive(Serialize)]
struct CandleDto {
    ts: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
}

async fn candles_handler(
    State(app_state): State<Arc<AppState>>,
    Query(params): Query<CandleQuery>,
) -> (HeaderMap, Json<Vec<CandleDto>>) {
    let mut headers = HeaderMap::new();
    headers.insert(axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN, "*".parse().unwrap());
    headers.insert(axum::http::header::CACHE_CONTROL, "no-cache".parse().unwrap());

    let history_map = app_state.history.read().await;
    let Some(dq) = history_map.get(&params.symbol) else { return (headers, Json(vec![])); };

    let now = chrono::Utc::now();
    let (cutoff, bucket_secs) = match params.timeframe.as_str() {
        // last 60 seconds with 1-second buckets
        "1m" => (now - Duration::seconds(60), 1_i64),
        // last 24 hours with 1-minute buckets
        _ => (now - Duration::hours(24), 60_i64),
    };

    #[derive(Clone, Copy)]
    struct Agg { open_price: f64, open_ts: i64, high: f64, low: f64, close_price: f64, close_ts: i64, volume: u64 }

    let mut map: HashMap<i64, Agg> = HashMap::new();
    for t in dq.iter() {
        if t.timestamp < cutoff { continue; }
        let ts_ms = t.timestamp.timestamp_millis();
        let sec = ts_ms / 1000;
        let bucket = (sec / bucket_secs) * bucket_secs;
        let price = t.price;
        let vol = t.volume;

        map.entry(bucket).and_modify(|a| {
            if t.timestamp.timestamp_millis() < a.open_ts { a.open_price = price; a.open_ts = t.timestamp.timestamp_millis(); }
            if t.timestamp.timestamp_millis() > a.close_ts { a.close_price = price; a.close_ts = t.timestamp.timestamp_millis(); }
            if price > a.high { a.high = price; }
            if price < a.low { a.low = price; }
            a.volume = a.volume.saturating_add(vol);
        }).or_insert(Agg { open_price: price, open_ts: ts_ms, high: price, low: price, close_price: price, close_ts: ts_ms, volume: vol });
    }

    let mut keys: Vec<i64> = map.keys().cloned().collect();
    keys.sort_unstable();
    let mut out = Vec::with_capacity(keys.len());
    for k in keys {
        let a = map[&k];
        out.push(CandleDto { ts: k * 1000, open: a.open_price, high: a.high, low: a.low, close: a.close_price, volume: a.volume });
    }

    (headers, Json(out))
}
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🚀 Starting Quant Trading Data Collector");
    
    // Load configuration
    let config = Arc::new(Config::from_env()?);
    info!("✅ Configuration loaded - Finnhub key: {}...", &config.finnhub_api_key[..8]);
    
    // Define symbols to track (configurable via SYMBOLS env, comma separated)
    let symbols: Vec<String> = std::env::var("SYMBOLS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
        .unwrap_or_else(|| vec![
            "AAPL".to_string(),
            "MSFT".to_string(),
            "GOOGL".to_string(),
            "TSLA".to_string(),
            "NVDA".to_string(),
            // 24/7 feed example
            "BINANCE:BTCUSDT".to_string(),
        ]);
    
    info!("📊 Tracking symbols: {:?}", symbols);
    info!("Press Ctrl+C to stop");
    
    // Try Finnhub WebSocket first, fall back to Polygon, then simulator
    info!("🌐 Attempting Finnhub WebSocket connection...");

    // Create broadcast channel and start HTTP server for SSE
    let (tx, _rx) = broadcast::channel::<MarketTick>(1024);
    let history: Arc<RwLock<HashMap<String, VecDeque<MarketTick>>>> = Arc::new(RwLock::new(HashMap::new()));

    // Fan-in ticks into in-memory history for backlogs
    let hist_writer_tx = tx.clone();
    let hist_map_for_task = history.clone();
    tokio::spawn(async move {
        let mut rx = hist_writer_tx.subscribe();
        while let Ok(tick) = rx.recv().await {
            let mut map = hist_map_for_task.write().await;
            let dq = map.entry(tick.symbol.clone()).or_insert_with(VecDeque::new);
            dq.push_back(tick.clone());
            if dq.len() > MAX_HISTORY_PER_SYMBOL { let _ = dq.pop_front(); }
        }
    });

    let http_state = Arc::new(AppState { sender: tx.clone(), history: history.clone() });
    tokio::spawn(async move {
        let cors = CorsLayer::new()
            .allow_methods([Method::GET])
            .allow_origin(Any)
            .allow_headers(Any);

        let app = Router::new()
            .route("/events", get(sse_handler))
            .route("/history", get(history_handler))
            .route("/chart", get(chart_handler))
            .route("/candles", get(candles_handler))
            .with_state(http_state)
            .layer(cors);

        let port: u16 = std::env::var("SSE_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8080);
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::UNSPECIFIED, port)).await.expect("bind http listener");
        info!("🌎 SSE server listening on http://0.0.0.0:{}/events", port);
        if let Err(err) = axum::serve(listener, app).await {
            warn!("HTTP server stopped: {}", err);
        }
    });
    
    let finnhub_client = FinnhubWebSocket::new(config.clone(), symbols.clone(), tx.clone());

    // If any BINANCE:* symbols present, spawn a Binance WS task for those
    let binance_symbols: Vec<String> = symbols.iter().filter(|s| s.starts_with("BINANCE:")).cloned().collect();
    if !binance_symbols.is_empty() {
        let binance_tx = tx.clone();
        tokio::spawn(async move {
            let client = BinanceWebSocket::new(binance_symbols, binance_tx);
            if let Err(e) = client.run().await {
                warn!("Binance WS failed: {}", e);
            }
        });
    }

    tokio::select! {
        res = finnhub_client.run() => {
            match res {
                Ok(_) => {
                    info!("✅ Finnhub WebSocket completed successfully");
                }
                Err(e) => {
                    warn!("❌ Finnhub WebSocket failed: {}", e);
                    
                    // Try Polygon if API key is available
                    if config.polygon_api_key != "not_set" {
                        info!("🌐 Attempting Polygon WebSocket connection...");
                        let polygon_client = PolygonWebSocket::new(config.clone(), symbols.clone(), tx.clone());

                        tokio::select! {
                            res2 = polygon_client.run() => {
                                match res2 {
                                    Ok(_) => info!("✅ Polygon WebSocket completed successfully"),
                                    Err(e2) => {
                                        warn!("❌ Polygon WebSocket failed: {}", e2);
                                        warn!("🎭 Falling back to market data simulator...");
                                        let mut simulator = MarketSimulator::new(symbols, tx.clone());
                                        tokio::select! {
                                            res3 = simulator.run() => { res3?; }
                                            _ = signal::ctrl_c() => {
                                                info!("🛑 Received Ctrl+C, exiting");
                                                return Ok(());
                                            }
                                        }
                                    }
                                }
                            }
                            _ = signal::ctrl_c() => {
                                info!("🛑 Received Ctrl+C, exiting");
                                return Ok(());
                            }
                        }
                    } else {
                        warn!("⚠️ POLYGON_API_KEY not set; skipping Polygon WebSocket");
                        warn!("🎭 Falling back to market data simulator...");
                        let mut simulator = MarketSimulator::new(symbols, tx.clone());
                        tokio::select! {
                            res3 = simulator.run() => { res3?; }
                            _ = signal::ctrl_c() => {
                                info!("🛑 Received Ctrl+C, exiting");
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }
        _ = signal::ctrl_c() => {
            info!("🛑 Received Ctrl+C, exiting");
            return Ok(());
        }
    }
    
    Ok(())
}
