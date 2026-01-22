//! Historical stock data collection
//! Targets symbols with congressional activity

use anyhow::{Result, anyhow};
use shared::Config;
use crate::database::TradingDatabase;
use tracing::{info, warn, error};
use sqlx::Row;
use std::sync::Arc;
use chrono::{NaiveDate, Utc, Duration};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct AlphaVantageResponse {
    #[serde(rename = "Time Series (Daily)")]
    time_series: Option<std::collections::HashMap<String, DailyData>>,
    #[serde(rename = "Error Message")]
    error_message: Option<String>,
    #[serde(rename = "Note")]
    note: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DailyData {
    #[serde(rename = "1. open")]
    open: String,
    #[serde(rename = "2. high")]
    high: String,
    #[serde(rename = "3. low")]
    low: String,
    #[serde(rename = "4. close")]
    close: String,
    #[serde(rename = "5. volume")]
    volume: String,
}

pub struct HistoricalDataCollector {
    config: Arc<Config>,
    database: Arc<TradingDatabase>,
    client: reqwest::Client,
}

impl HistoricalDataCollector {
    pub fn new(config: Arc<Config>, database: Arc<TradingDatabase>) -> Self {
        Self {
            config,
            database,
            client: reqwest::Client::new(),
        }
    }
    
    /// Collect historical data for all symbols with congressional activity
    pub async fn collect_targeted_historical_data(&self) -> Result<()> {
        info!("📈 Starting targeted historical data collection...");
        
        // Get symbols that need historical data
        let symbols = self.database.get_target_symbols().await?;
        info!("🎯 Found {} symbols with congressional activity", symbols.len());
        
        for (i, symbol) in symbols.iter().enumerate() {
            info!("🔄 Processing {}/{}: {}", i + 1, symbols.len(), symbol);
            
            match self.collect_symbol_data(symbol).await {
                Ok(days_collected) => {
                    info!("✅ Collected {} days of data for {}", days_collected, symbol);
                }
                Err(e) => {
                    error!("❌ Failed to collect data for {}: {}", symbol, e);
                }
            }
            
            // Rate limiting for Alpha Vantage (5 calls per minute)
            if i < symbols.len() - 1 {
                info!("⏳ Rate limiting... waiting 15 seconds");
                tokio::time::sleep(tokio::time::Duration::from_secs(15)).await;
            }
        }
        
        info!("🎉 Historical data collection complete!");
        Ok(())
    }
    
    /// Collect historical data for a single symbol
    async fn collect_symbol_data(&self, symbol: &str) -> Result<usize> {
        let url = format!(
            "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey={}",
            symbol, self.config.alpha_vantage_key
        );
        
        let response = self.client.get(&url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("API request failed: {}", response.status()));
        }
        
        let data: AlphaVantageResponse = response.json().await?;
        
        if let Some(error) = data.error_message {
            return Err(anyhow!("Alpha Vantage error: {}", error));
        }
        
        if let Some(note) = data.note {
            warn!("⚠️ Rate limited: {}", note);
            return Ok(0);
        }
        
        let time_series = data.time_series.ok_or_else(|| anyhow!("No time series data"))?;
        
        let mut stored_count = 0;
        let cutoff_date = Utc::now().date_naive() - Duration::days(365 * 2); // 2 years
        
        for (date_str, daily_data) in time_series {
            let date = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")?;
            
            if date < cutoff_date {
                continue; // Skip old data
            }
            
            let open: f64 = daily_data.open.parse()?;
            let high: f64 = daily_data.high.parse()?;
            let low: f64 = daily_data.low.parse()?;
            let close: f64 = daily_data.close.parse()?;
            let volume: i64 = daily_data.volume.parse()?;
            
            if let Err(e) = self.database.store_historical_price(symbol, date, open, high, low, close, volume).await {
                warn!("Failed to store price data for {} on {}: {}", symbol, date, e);
            } else {
                stored_count += 1;
            }
        }
        
        Ok(stored_count)
    }
    
    /// Run correlation analysis after data collection
    pub async fn run_correlation_analysis(&self) -> Result<()> {
        info!("🔗 Running correlation analysis...");
        
        // Get all congressional trades that need correlation analysis
        let trades = sqlx::query(
            r#"
            SELECT DISTINCT ct.ticker, ct.transaction_date, ct.representative
            FROM congressional_trades ct
            LEFT JOIN trade_correlations tc ON ct.ticker = tc.symbol 
                AND ct.transaction_date = tc.transaction_date
                AND ct.representative = tc.representative
            WHERE ct.ticker IS NOT NULL 
                AND ct.ticker != 'N/A'
                AND tc.id IS NULL  -- Not already analyzed
            "#
        )
        .fetch_all(&self.database.pool)
        .await?;
        
        info!("📊 Analyzing {} trade correlations...", trades.len());
        
        for trade in trades {
            let ticker: Option<String> = trade.try_get("ticker")?;
            let transaction_date: NaiveDate = trade.try_get("transaction_date")?;
            let representative: String = trade.try_get("representative")?;
            if let Some(ticker) = ticker {
                match self.database.calculate_trade_correlation(&ticker, transaction_date, &representative).await {
                    Ok(Some(_correlation_id)) => {
                        info!("✅ Analyzed: {} {} on {}", representative, ticker, transaction_date);
                    }
                    Ok(None) => {
                        warn!("⚠️ No historical data for: {} on {}", ticker, transaction_date);
                    }
                    Err(e) => {
                        error!("❌ Correlation failed for {} {}: {}", ticker, transaction_date, e);
                    }
                }
            }
        }
        
        info!("🎉 Correlation analysis complete!");
        Ok(())
    }
    
    /// Fetch daily data for a symbol (for compatibility with main.rs)
    pub async fn fetch_daily_data(&self, symbol: &str) -> Result<Vec<DailyPriceData>> {
        match self.collect_symbol_data(symbol).await {
            Ok(count) => {
                info!("📊 Collected {} days of data for {}", count, symbol);
                // Return mock data for now - in a real implementation you'd return actual data
                Ok(Vec::new())
            }
            Err(e) => {
                warn!("⚠️ Failed to collect data for {}: {}", symbol, e);
                Ok(Vec::new())
            }
        }
    }
    
    /// Correlate events with prices (for compatibility with main.rs)
    pub async fn correlate_events_with_prices(
        &self,
        symbol: &str,
        _trades: &[crate::congress_free::DetailedCongressTrade],
        _price_data: &[DailyPriceData]
    ) -> Result<SymbolAnalysis> {
        // Return a mock analysis for now
        Ok(SymbolAnalysis {
            symbol: symbol.to_string(),
            analysis_date: Utc::now().date_naive().to_string(),
            price_data_points: 0,
            events_analyzed: 0,
            correlations: Vec::new(),
            summary: AnalysisSummary {
                avg_return_1d_on_events: 0.0,
                avg_return_3d_on_events: 0.0,
                avg_return_7d_on_events: 0.0,
                positive_events_1d: 0,
                negative_events_1d: 0,
                max_single_day_impact: 0.0,
                min_single_day_impact: 0.0,
            },
        })
    }
}

/// Daily price data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyPriceData {
    pub date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
}

/// Symbol analysis structure for compatibility with main.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolAnalysis {
    pub symbol: String,
    pub analysis_date: String,
    pub price_data_points: usize,
    pub events_analyzed: usize,
    pub correlations: Vec<EventCorrelation>,
    pub summary: AnalysisSummary,
}

/// Analysis summary structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisSummary {
    pub avg_return_1d_on_events: f64,
    pub avg_return_3d_on_events: f64,
    pub avg_return_7d_on_events: f64,
    pub positive_events_1d: i32,
    pub negative_events_1d: i32,
    pub max_single_day_impact: f64,
    pub min_single_day_impact: f64,
}

/// Event correlation structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventCorrelation {
    pub event_date: String,
    pub representative: String,
    pub transaction_type: String,
    pub return_1d: f64,
    pub return_3d: f64,
    pub return_7d: f64,
}
