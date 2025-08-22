//! High-performance market data collector
//! Connects to Finnhub for FREE real-time stock data
//! 
//! Features:
//! - Real-time WebSocket connections to Finnhub, Polygon, and Binance
//! - Persistent tick history storage (saves to JSON files every 30 seconds)
//! - HTTP API with SSE for real-time data streaming
//! - Chart data endpoints for candlesticks and historical prices

mod polygon;
mod finnhub;
mod simulator;
mod binance;
mod alt_data;
mod congress_free;
mod congress_sources;
mod historical;

use anyhow::Result;
use tracing::{info, warn};
use shared::{Config, MarketTick};
use std::sync::Arc;
use finnhub::FinnhubWebSocket;
use simulator::MarketSimulator;
use polygon::PolygonWebSocket;
use binance::BinanceWebSocket;
use alt_data::AltDataCollector;
use congress_free::FreeCongressTracker;
use historical::HistoricalDataCollector;
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
use std::fs;
use std::path::Path;
use std::time::SystemTime;

#[derive(Clone)]
struct AppState {
    sender: Sender<MarketTick>,
    history: Arc<RwLock<HashMap<String, VecDeque<MarketTick>>>>,
}

const MAX_HISTORY_PER_SYMBOL: usize = 1000;
const HISTORY_SAVE_INTERVAL_SECS: u64 = 30; // Save every 30 seconds
const HISTORY_FILE_PREFIX: &str = "tick_history";

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

/// Save target symbols to JSON file
async fn save_target_symbols(symbols: &[String], filepath: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(symbols)?;
    fs::write(filepath, json)?;
    info!("💾 Saved {} target symbols to {}", symbols.len(), filepath);
    Ok(())
}

/// Load target symbols from JSON file
async fn load_target_symbols(filepath: &str) -> Result<Vec<String>> {
    if !Path::new(filepath).exists() {
        return Ok(Vec::new());
    }
    
    let content = fs::read_to_string(filepath)?;
    let symbols: Vec<String> = serde_json::from_str(&content)?;
    info!("📂 Loaded {} target symbols from {}", symbols.len(), filepath);
    Ok(symbols)
}

/// Save tick history to JSON file
async fn save_tick_history(history: &HashMap<String, VecDeque<MarketTick>>) -> Result<()> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    for (symbol, ticks) in history.iter() {
        if ticks.is_empty() { continue; }
        
        let filename = format!("{}_{}.json", HISTORY_FILE_PREFIX, symbol.replace(":", "_"));
        let ticks_vec: Vec<MarketTick> = ticks.iter().cloned().collect();
        
        match serde_json::to_string_pretty(&ticks_vec) {
            Ok(json) => {
                if let Err(e) = fs::write(&filename, json) {
                    warn!("Failed to save history for {}: {}", symbol, e);
                } else {
                    info!("💾 Saved {} ticks for {} to {}", ticks_vec.len(), symbol, filename);
                }
            }
            Err(e) => warn!("Failed to serialize history for {}: {}", symbol, e),
        }
    }
    
    Ok(())
}

/// Load tick history from JSON files
async fn load_tick_history() -> Result<HashMap<String, VecDeque<MarketTick>>> {
    let mut history = HashMap::new();
    
    // Find all tick history files
    let entries = match fs::read_dir(".") {
        Ok(entries) => entries,
        Err(_) => return Ok(history),
    };
    
    for entry in entries.flatten() {
        let filename = entry.file_name();
        let filename_str = filename.to_string_lossy();
        
        if filename_str.starts_with(HISTORY_FILE_PREFIX) && filename_str.ends_with(".json") {
            // Extract symbol from filename
            let symbol_part = filename_str
                .strip_prefix(&format!("{}_", HISTORY_FILE_PREFIX))
                .and_then(|s| s.strip_suffix(".json"))
                .unwrap_or("");
            
            let symbol = symbol_part.replace("_", ":");
            
            match fs::read_to_string(entry.path()) {
                Ok(content) => {
                    match serde_json::from_str::<Vec<MarketTick>>(&content) {
                        Ok(ticks) => {
                            let mut deque = VecDeque::new();
                            for tick in ticks.into_iter().take(MAX_HISTORY_PER_SYMBOL) {
                                deque.push_back(tick);
                            }
                            history.insert(symbol.clone(), deque);
                            info!("📂 Loaded {} ticks for {} from {}", history.get(&symbol).unwrap().len(), symbol, filename_str);
                        }
                        Err(e) => warn!("Failed to parse history file {}: {}", filename_str, e),
                    }
                }
                Err(e) => warn!("Failed to read history file {}: {}", filename_str, e),
            }
        }
    }
    
    Ok(history)
}

/// Phase 1: Collect detailed government data and save target symbols
async fn run_phase_1(config: Arc<Config>) -> Result<()> {
    info!("🏛️ === PHASE 1: COLLECTING DETAILED GOVERNMENT DATA ===");
    
    // Collect detailed congressional trades
    let mut congress_tracker = FreeCongressTracker::new();
    let _events = congress_tracker.collect_all_trades().await?;
    
    info!("✅ Phase 1 completed. Detailed files saved:");
    info!("   📄 congressional_trades_detailed.json");
    info!("   📊 symbol_analysis.json");
    info!("   🎯 target_symbols.json");
    
    // Also collect government contracts data in background
    tokio::spawn(async move {
        let alt_collector = AltDataCollector::new(config);
        if let Err(e) = alt_collector.run().await {
            warn!("Alt data collector error: {}", e);
        }
    });
    
    Ok(())
}

/// Phase 2: Load target symbols, fetch historical data, and perform event correlation
async fn run_phase_2(config: Arc<Config>) -> Result<()> {
    info!("📈 === PHASE 2: COLLECTING HISTORICAL DATA & EVENT CORRELATION ===");
    
    // Load target symbols from Phase 1
    let target_symbols = load_target_symbols("target_symbols.json").await?;
    
    if target_symbols.is_empty() {
        warn!("⚠️ No target symbols found. Run Phase 1 first with COLLECT_GOVERNMENT_DATA=1");
        return Ok(());
    }
    
    info!("📊 Fetching historical data for {} symbols: {:?}", target_symbols.len(), target_symbols);
    
    // Load detailed congressional trades from Phase 1
    let detailed_trades_content = fs::read_to_string("congressional_trades_detailed.json")?;
    let detailed_trades: Vec<congress_free::DetailedCongressTrade> = serde_json::from_str(&detailed_trades_content)?;
    info!("📋 Loaded {} detailed congressional trades", detailed_trades.len());
    
    // Initialize historical data collector
    let historical_collector = HistoricalDataCollector::new(&config);
    
    // Process each target symbol
    for symbol in &target_symbols {
        info!("📈 Processing historical data for {}", symbol);
        
        // Check if analysis already exists (can be overridden with FORCE_REFRESH=1)
        let analysis_filename = format!("analysis_{}.json", symbol);
        if Path::new(&analysis_filename).exists() && std::env::var("FORCE_REFRESH").is_err() {
            info!("📁 Analysis for {} already exists, skipping", symbol);
            continue;
        }
        
        // Fetch historical price data
        match historical_collector.fetch_daily_data(symbol).await {
            Ok(price_data) => {
                if price_data.is_empty() {
                    warn!("⚠️ No price data received for {} (API limit or error)", symbol);
                    
                    // Still create an empty analysis file
                    let empty_analysis = historical::SymbolAnalysis {
                        symbol: symbol.clone(),
                        analysis_date: chrono::Utc::now().date_naive().to_string(),
                        price_data_points: 0,
                        events_analyzed: 0,
                        correlations: Vec::new(),
                        summary: historical::AnalysisSummary {
                            avg_return_1d_on_events: 0.0,
                            avg_return_3d_on_events: 0.0,
                            avg_return_7d_on_events: 0.0,
                            positive_events_1d: 0,
                            negative_events_1d: 0,
                            max_single_day_impact: 0.0,
                            min_single_day_impact: 0.0,
                        },
                    };
                    
                    let analysis_json = serde_json::to_string_pretty(&empty_analysis)?;
                    fs::write(&analysis_filename, analysis_json)?;
                    info!("📄 Saved empty analysis for {} to {}", symbol, analysis_filename);
                } else {
                    info!("📊 Fetched {} price points for {}", price_data.len(), symbol);
                    
                    // Correlate with congressional events
                    let analysis = historical_collector.correlate_events_with_prices(symbol, &detailed_trades, &price_data)?;
                    
                    // Save analysis to file
                    let analysis_json = serde_json::to_string_pretty(&analysis)?;
                    fs::write(&analysis_filename, analysis_json)?;
                    
                    info!("💾 Saved analysis for {} to {} ({} events correlated)", 
                          symbol, analysis_filename, analysis.events_analyzed);
                    
                    if !analysis.correlations.is_empty() {
                        info!("📈 Average returns for {}: 1d={:.2}%, 3d={:.2}%, 7d={:.2}%", 
                              symbol, 
                              analysis.summary.avg_return_1d_on_events,
                              analysis.summary.avg_return_3d_on_events,
                              analysis.summary.avg_return_7d_on_events);
                    }
                }
            }
            Err(e) => {
                warn!("❌ Failed to fetch historical data for {}: {}", symbol, e);
                continue;
            }
        }
        
        // Alpha Vantage free tier: 5 requests per minute
        info!("⏳ Waiting 15 seconds for API rate limit...");
        tokio::time::sleep(tokio::time::Duration::from_secs(15)).await;
    }
    
    info!("✅ Phase 2 completed. Historical analysis files saved to analysis_SYMBOL.json");
    Ok(())
}

/// Phase 3: Real-time monitoring with target symbols
async fn run_phase_3(config: Arc<Config>) -> Result<()> {
    info!("🔴 === PHASE 3: REAL-TIME MONITORING ===");
    
    // Load target symbols if available, otherwise use defaults
    let mut symbols = load_target_symbols("target_symbols.json").await.unwrap_or_default();
    
    // If no target symbols, use default tracking symbols
    if symbols.is_empty() {
        symbols = std::env::var("SYMBOLS")
            .ok()
            .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
            .unwrap_or_else(|| vec![
                "AAPL".to_string(),
                "MSFT".to_string(),
                "GOOGL".to_string(),
                "TSLA".to_string(),
                "NVDA".to_string(),
                "BINANCE:BTCUSDT".to_string(),
            ]);
    }
    
    info!("📊 Real-time tracking symbols: {:?}", symbols);
    info!("Press Ctrl+C to stop");
    
    // Create broadcast channel and start HTTP server for SSE
    let (tx, _rx) = broadcast::channel::<MarketTick>(1024);
    
    // Load existing tick history from files
    info!("📂 Loading existing tick history...");
    let existing_history = load_tick_history().await.unwrap_or_default();
    let history: Arc<RwLock<HashMap<String, VecDeque<MarketTick>>>> = Arc::new(RwLock::new(existing_history));

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

    // Periodic history saving task
    let save_history_map = history.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(HISTORY_SAVE_INTERVAL_SECS));
        loop {
            interval.tick().await;
            let history_snapshot = {
                let map = save_history_map.read().await;
                map.clone()
            };
            
            if !history_snapshot.is_empty() {
                if let Err(e) = save_tick_history(&history_snapshot).await {
                    warn!("Failed to save tick history: {}", e);
                }
            }
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
    
    // Start Alternative Data collector in background for real-time monitoring
    let alt_config = config.clone();
    tokio::spawn(async move {
        let collector = AltDataCollector::new(alt_config);
        if let Err(e) = collector.run().await { warn!("Alt data collector error: {}", e); }
    });

    // Start congressional tracker for real-time monitoring
    tokio::spawn(async move {
        let mut tracker = FreeCongressTracker::new();
        // Check for new congressional trades every hour
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600));
        loop {
            interval.tick().await;
            if let Err(e) = tracker.get_recent_trades(1).await {
                warn!("Congressional tracker error: {}", e);
            }
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
            info!("🛑 Received Ctrl+C, saving history and exiting");
            
            // Final save of tick history before exiting
            let final_history = {
                let map = history.read().await;
                map.clone()
            };
            
            if !final_history.is_empty() {
                info!("💾 Performing final save of tick history...");
                if let Err(e) = save_tick_history(&final_history).await {
                    warn!("Failed to perform final save: {}", e);
                } else {
                    info!("✅ Final history save completed");
                }
            }
            
            return Ok(());
        }
    }
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🚀 Starting Quant Trading Data Collector");
    
    // Load configuration
    let config = Arc::new(Config::from_env()?);
    info!("✅ Configuration loaded - Finnhub key: {}...", &config.finnhub_api_key[..8]);
    
    // Check environment flags to determine which phase to run
    let collect_government = std::env::var("COLLECT_GOVERNMENT_DATA")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
        
    let collect_historical = std::env::var("COLLECT_HISTORICAL")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    match (collect_government, collect_historical) {
        (true, _) => {
            // Phase 1: Collect government data
            run_phase_1(config).await
        }
        (false, true) => {
            // Phase 2: Collect historical data
            run_phase_2(config).await
        }
        (false, false) => {
            // Phase 3: Real-time monitoring (default)
            run_phase_3(config).await
        }
    }
}