//! Historical data collection and analysis
//! Fetches OHLCV data from Alpha Vantage and correlates with congressional events

use anyhow::{Result, anyhow};
use chrono::{NaiveDate, Duration};
use serde::{Deserialize, Serialize};
use shared::Config;
use std::collections::HashMap;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventCorrelation {
    pub event_date: String,
    pub event_type: String,
    pub event_description: String,
    pub impact_score: f64,
    pub return_1d: Option<f64>,
    pub return_3d: Option<f64>,
    pub return_7d: Option<f64>,
    pub volume_change_1d: Option<f64>,
    pub price_at_event: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolAnalysis {
    pub symbol: String,
    pub analysis_date: String,
    pub price_data_points: usize,
    pub events_analyzed: usize,
    pub correlations: Vec<EventCorrelation>,
    pub summary: AnalysisSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisSummary {
    pub avg_return_1d_on_events: f64,
    pub avg_return_3d_on_events: f64,
    pub avg_return_7d_on_events: f64,
    pub positive_events_1d: usize,
    pub negative_events_1d: usize,
    pub max_single_day_impact: f64,
    pub min_single_day_impact: f64,
}

#[derive(Debug, Deserialize)]
struct AlphaVantageResponse {
    #[serde(rename = "Time Series (Daily)")]
    time_series: Option<HashMap<String, DailyData>>,
    #[serde(rename = "Note")]
    note: Option<String>,
    #[serde(rename = "Information")]
    information: Option<String>,
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
    client: reqwest::Client,
    api_key: String,
}

impl HistoricalDataCollector {
    pub fn new(config: &Config) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("QuantTrader/1.0")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();
        
        Self {
            client,
            api_key: config.alpha_vantage_key.clone(),
        }
    }

    /// Fetch daily OHLCV data for a symbol from Alpha Vantage
    pub async fn fetch_daily_data(&self, symbol: &str) -> Result<Vec<PricePoint>> {
        if self.api_key == "not_set" {
            return Err(anyhow!("Alpha Vantage API key not configured"));
        }

        let url = format!(
            "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey={}",
            symbol, self.api_key
        );

        info!("📈 Fetching historical data for {} from Alpha Vantage", symbol);
        
        let response = self.client.get(&url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Alpha Vantage API error: {}", response.status()));
        }

        let data: AlphaVantageResponse = response.json().await?;

        // Check for API limits or errors
        if let Some(note) = data.note {
            warn!("Alpha Vantage note: {}", note);
            return Ok(Vec::new()); // Return empty data for rate limits
        }

        if let Some(info) = data.information {
            warn!("Alpha Vantage info: {}", info);
            return Ok(Vec::new());
        }

        let time_series = data.time_series.ok_or_else(|| anyhow!("No time series data in response"))?;

        let mut price_points = Vec::new();
        for (date, daily) in time_series {
            let price_point = PricePoint {
                date: date.clone(),
                open: daily.open.parse().unwrap_or(0.0),
                high: daily.high.parse().unwrap_or(0.0),
                low: daily.low.parse().unwrap_or(0.0),
                close: daily.close.parse().unwrap_or(0.0),
                volume: daily.volume.parse().unwrap_or(0),
            };
            price_points.push(price_point);
        }

        // Sort by date (most recent first)
        price_points.sort_by(|a, b| b.date.cmp(&a.date));

        info!("✅ Fetched {} data points for {}", price_points.len(), symbol);
        Ok(price_points)
    }

    /// Correlate events with price data to calculate returns
    pub fn correlate_events_with_prices(
        &self,
        symbol: &str,
        events: &[crate::congress_free::DetailedCongressTrade],
        price_data: &[PricePoint],
    ) -> Result<SymbolAnalysis> {
        info!("🔍 Correlating {} events with {} price points for {}", 
              events.len(), price_data.len(), symbol);

        // Create a map of date -> price for fast lookup
        let price_map: HashMap<String, &PricePoint> = price_data
            .iter()
            .map(|p| (p.date.clone(), p))
            .collect();

        let mut correlations = Vec::new();

        for event in events {
            if event.ticker != symbol {
                continue;
            }

            let correlation = self.calculate_event_correlation(event, &price_map)?;
            correlations.push(correlation);
        }

        // Calculate summary statistics
        let summary = self.calculate_summary(&correlations);

        let analysis = SymbolAnalysis {
            symbol: symbol.to_string(),
            analysis_date: chrono::Utc::now().date_naive().to_string(),
            price_data_points: price_data.len(),
            events_analyzed: correlations.len(),
            correlations,
            summary,
        };

        Ok(analysis)
    }

    fn calculate_event_correlation(
        &self,
        event: &crate::congress_free::DetailedCongressTrade,
        price_map: &HashMap<String, &PricePoint>,
    ) -> Result<EventCorrelation> {
        let event_date = &event.transaction_date;
        
        // Find price at event date (or next trading day)
        let (event_price, actual_date) = self.find_price_on_or_after(event_date, price_map)?;

        // Calculate returns
        let return_1d = self.calculate_return(&actual_date, 1, price_map);
        let return_3d = self.calculate_return(&actual_date, 3, price_map);
        let return_7d = self.calculate_return(&actual_date, 7, price_map);

        // Calculate volume change
        let volume_change_1d = self.calculate_volume_change(&actual_date, price_map);

        Ok(EventCorrelation {
            event_date: event_date.clone(),
            event_type: event.transaction_type.clone(),
            event_description: format!("{} {} {}", 
                event.representative, event.transaction_type, event.amount),
            impact_score: event.impact_score,
            return_1d,
            return_3d,
            return_7d,
            volume_change_1d,
            price_at_event: Some(event_price),
        })
    }

    fn find_price_on_or_after(
        &self,
        target_date: &str,
        price_map: &HashMap<String, &PricePoint>,
    ) -> Result<(f64, String)> {
        // Try the exact date first
        if let Some(price_point) = price_map.get(target_date) {
            return Ok((price_point.close, target_date.to_string()));
        }

        // If not found, try the next few days (markets might be closed)
        let target = NaiveDate::parse_from_str(target_date, "%Y-%m-%d")?;
        
        for i in 1..=5 {
            let next_date = target + Duration::days(i);
            let next_date_str = next_date.to_string();
            
            if let Some(price_point) = price_map.get(&next_date_str) {
                return Ok((price_point.close, next_date_str));
            }
        }

        Err(anyhow!("No price data found for {} or following days", target_date))
    }

    fn calculate_return(
        &self,
        base_date: &str,
        days: i64,
        price_map: &HashMap<String, &PricePoint>,
    ) -> Option<f64> {
        let base_date_parsed = NaiveDate::parse_from_str(base_date, "%Y-%m-%d").ok()?;
        let future_date = base_date_parsed + Duration::days(days);
        let future_date_str = future_date.to_string();

        let base_price = price_map.get(base_date)?.close;
        
        // Try to find price on target date or nearby
        for i in 0..=3 {
            let check_date = future_date + Duration::days(i);
            let check_date_str = check_date.to_string();
            
            if let Some(future_price_point) = price_map.get(&check_date_str) {
                let future_price = future_price_point.close;
                return Some((future_price - base_price) / base_price * 100.0);
            }
        }

        None
    }

    fn calculate_volume_change(
        &self,
        base_date: &str,
        price_map: &HashMap<String, &PricePoint>,
    ) -> Option<f64> {
        let base_date_parsed = NaiveDate::parse_from_str(base_date, "%Y-%m-%d").ok()?;
        let prev_date = base_date_parsed - Duration::days(1);
        let prev_date_str = prev_date.to_string();

        let base_volume = price_map.get(base_date)?.volume as f64;
        let prev_volume = price_map.get(&prev_date_str)?.volume as f64;

        if prev_volume > 0.0 {
            Some((base_volume - prev_volume) / prev_volume * 100.0)
        } else {
            None
        }
    }

    fn calculate_summary(&self, correlations: &[EventCorrelation]) -> AnalysisSummary {
        let mut return_1d_sum = 0.0;
        let mut return_3d_sum = 0.0;
        let mut return_7d_sum = 0.0;
        let mut return_1d_count = 0;
        let mut return_3d_count = 0;
        let mut return_7d_count = 0;
        let mut positive_1d = 0;
        let mut negative_1d = 0;
        let mut max_impact = f64::MIN;
        let mut min_impact = f64::MAX;

        for correlation in correlations {
            if let Some(ret_1d) = correlation.return_1d {
                return_1d_sum += ret_1d;
                return_1d_count += 1;
                if ret_1d > 0.0 { positive_1d += 1; } else { negative_1d += 1; }
                max_impact = max_impact.max(ret_1d);
                min_impact = min_impact.min(ret_1d);
            }

            if let Some(ret_3d) = correlation.return_3d {
                return_3d_sum += ret_3d;
                return_3d_count += 1;
            }

            if let Some(ret_7d) = correlation.return_7d {
                return_7d_sum += ret_7d;
                return_7d_count += 1;
            }
        }

        AnalysisSummary {
            avg_return_1d_on_events: if return_1d_count > 0 { return_1d_sum / return_1d_count as f64 } else { 0.0 },
            avg_return_3d_on_events: if return_3d_count > 0 { return_3d_sum / return_3d_count as f64 } else { 0.0 },
            avg_return_7d_on_events: if return_7d_count > 0 { return_7d_sum / return_7d_count as f64 } else { 0.0 },
            positive_events_1d: positive_1d,
            negative_events_1d: negative_1d,
            max_single_day_impact: if max_impact != f64::MIN { max_impact } else { 0.0 },
            min_single_day_impact: if min_impact != f64::MAX { min_impact } else { 0.0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_return_calculation() {
        // This would test the return calculation logic
        // Implementation depends on your specific needs
    }

    #[test]
    fn test_event_correlation() {
        // This would test event correlation logic
        // Implementation depends on your specific needs
    }
}


