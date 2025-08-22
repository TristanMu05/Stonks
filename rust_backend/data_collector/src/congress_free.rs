//! Detailed Congressional trading data collection and analysis

use anyhow::Result;
use chrono::{DateTime, Utc, NaiveDate};
use serde::{Serialize, Deserialize};
use shared::{AltDataEvent, AltDataType};
use std::collections::HashMap;
use tracing::{info, warn};
use uuid::Uuid;

use crate::congress_sources::{CongressSource, CapitolTradesSource, SenateGithubSource, NormalizedTrade};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedCongressTrade {
    pub representative: String,
    pub district: String,
    pub transaction_date: String,
    pub disclosure_date: String,
    pub transaction_type: String,
    pub owner: String,
    pub ticker: String,
    pub asset_description: String,
    pub amount: String,
    pub amount_min: f64,
    pub amount_max: f64,
    pub disclosure_lag_days: i32,
    pub impact_score: f64,
    pub ptr_link: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolAnalysis {
    pub symbol: String,
    pub total_trades: usize,
    pub purchase_count: usize,
    pub sale_count: usize,
    pub avg_disclosure_lag: f64,
    pub avg_impact_score: f64,
    pub top_representatives: Vec<RepresentativeActivity>,
    pub amount_distribution: AmountDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepresentativeActivity {
    pub name: String,
    pub trade_count: usize,
    pub avg_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmountDistribution {
    pub under_15k: usize,
    pub from_15k_to_50k: usize,
    pub from_50k_to_100k: usize,
    pub from_100k_to_250k: usize,
    pub over_250k: usize,
}

pub struct FreeCongressTracker {
    max_pages: usize,
    primary: CapitolTradesSource,
    fallback: SenateGithubSource,
    all_trades: Vec<DetailedCongressTrade>,
}

impl FreeCongressTracker {
    pub fn new() -> Self {
        let max_pages = std::env::var("CONGRESS_MAX_PAGES")
            .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(50);
        Self {
            max_pages,
            primary: CapitolTradesSource::new(),
            fallback: SenateGithubSource::new(),
            all_trades: Vec::new(),
        }
    }

    pub async fn collect_all_trades(&mut self) -> Result<Vec<AltDataEvent>> {
        info!("🏛️ Fetching congressional trades from Capitol Trades + GitHub fallback...");
        
        // Fetch normalized trades from sources
        let normalized = match self.primary.fetch_all(self.max_pages).await {
            Ok(v) if !v.is_empty() => v,
            Ok(_) => {
                warn!("Primary returned 0 trades. Using Senate GitHub fallback.");
                self.fallback.fetch_all(self.max_pages).await?
            }
            Err(e) => {
                warn!("Primary source error: {e}. Using Senate GitHub fallback.");
                self.fallback.fetch_all(self.max_pages).await?
            }
        };

        info!("📊 Converting {} normalized trades to detailed format", normalized.len());

        // Convert normalized trades to detailed trades
        self.all_trades = normalized.into_iter().filter_map(|n| self.convert_to_detailed(n)).collect();

        info!("✅ Converted to {} detailed trades", self.all_trades.len());

        // Analyze symbols
        let symbol_analysis = self.analyze_symbols();

        // Save detailed files
        self.save_detailed_files(&symbol_analysis).await?;

        // Convert to AltDataEvents for compatibility
        let events = self.all_trades.iter()
            .map(|trade| self.to_alt_data_event(trade))
            .collect();

        info!("🎯 Generated {} AltDataEvents from detailed trades", self.all_trades.len());
        Ok(events)
    }

    fn convert_to_detailed(&self, n: NormalizedTrade) -> Option<DetailedCongressTrade> {
        // Keep trades even if they don't have a ticker (private companies, etc.)
        let ticker = n.ticker.clone().unwrap_or_else(|| "N/A".to_string());
        let transaction_type = n.transaction_type.clone().unwrap_or_else(|| "trade".to_string());
        let owner = n.owner.clone().unwrap_or_else(|| "unknown".to_string());

        let amount_str = n.amount.clone().unwrap_or_else(|| "$1,001 - $15,000".to_string());
        let (amount_min, amount_max) = Self::parse_amount_range_static(&amount_str);

        let tx = n.traded_date.map(|d| d.to_string()).unwrap_or_else(|| Utc::now().date_naive().to_string());
        let disc = n.disclosed_date.map(|d| d.to_string()).unwrap_or_else(|| tx.clone());
        let lag_days = {
            let a = chrono::NaiveDate::parse_from_str(&tx, "%Y-%m-%d").ok()?;
            let b = chrono::NaiveDate::parse_from_str(&disc, "%Y-%m-%d").ok()?;
            (b - a).num_days() as i32
        };

        let mut detailed_trade = DetailedCongressTrade {
            representative: n.representative.clone(),
            district: "".to_string(),
            transaction_date: tx,
            disclosure_date: disc,
            transaction_type,
            owner,
            ticker,
            asset_description: "".to_string(),
            amount: amount_str.clone(),
            amount_min,
            amount_max,
            disclosure_lag_days: lag_days,
            impact_score: 0.0,
            ptr_link: n.detail_url.unwrap_or_default(),
        };

        detailed_trade.impact_score = self.calculate_impact_score(&detailed_trade);
        Some(detailed_trade)
    }

    fn parse_amount_range_static(amount_str: &str) -> (f64, f64) {
        use tracing::debug;
        
        let clean = amount_str.replace('$', "").replace(',', "");
        debug!("Parsing amount: '{}' -> '{}'", amount_str, clean);
        
        // Handle K notation (e.g., "15K–50K")
        if clean.contains('K') {
            let k_clean = clean.replace('K', "000");
            if let Some((a, b)) = k_clean.split_once(" - ") {
                let min = a.trim().parse().unwrap_or(1000.0);
                let max = b.trim().parse().unwrap_or(15000.0);
                debug!("Parsed K format with ' - ': {} -> ({}, {})", amount_str, min, max);
                return (min, max);
            } else if let Some((a, b)) = k_clean.split_once("–") {
                let min = a.trim().parse().unwrap_or(1000.0);
                let max = b.trim().parse().unwrap_or(15000.0);
                debug!("Parsed K format with '–': {} -> ({}, {})", amount_str, min, max);
                return (min, max);
            } else if let Some((a, b)) = k_clean.split_once("-") {
                let min = a.trim().parse().unwrap_or(1000.0);
                let max = b.trim().parse().unwrap_or(15000.0);
                debug!("Parsed K format with '-': {} -> ({}, {})", amount_str, min, max);
                return (min, max);
            }
        }
        
        // Handle standard dollar ranges (e.g., "$1,001 - $15,000")
        if let Some((a, b)) = clean.split_once(" - ") {
            let min = a.trim().parse().unwrap_or(1000.0);
            let max = b.trim().parse().unwrap_or(15000.0);
            debug!("Parsed standard format with ' - ': {} -> ({}, {})", amount_str, min, max);
            return (min, max);
        } else if let Some((a, b)) = clean.split_once("–") {
            let min = a.trim().parse().unwrap_or(1000.0);
            let max = b.trim().parse().unwrap_or(15000.0);
            debug!("Parsed standard format with '–': {} -> ({}, {})", amount_str, min, max);
            return (min, max);
        } else if let Some((a, b)) = clean.split_once("-") {
            // Handle the problematic format like "6648-93318"
            let min = a.trim().parse().unwrap_or(1000.0);
            let max = b.trim().parse().unwrap_or(15000.0);
            debug!("Parsed raw number format: {} -> ({}, {})", amount_str, min, max);
            return (min, max);
        }
        
        // Handle single value - treat as point estimate and create a range
        let val: f64 = clean.trim().parse().unwrap_or(15000.0);
        let min = val * 0.8; // 20% below
        let max = val * 1.2; // 20% above
        debug!("Parsed single value: {} -> ({}, {})", amount_str, min, max);
        (min, max)
    }

    fn calculate_impact_score(&self, trade: &DetailedCongressTrade) -> f64 {
        let mut score: f64 = 0.3;
        let rep = trade.representative.to_lowercase();
        
        if rep.contains("pelosi") || rep.contains("mccarthy") || rep.contains("schumer") || rep.contains("mcconnell") {
            score += 0.4;
        }
        
        if rep.contains("speaker") || rep.contains("leader") || rep.contains("whip") || rep.contains("chair") {
            score += 0.3;
        }
        
        let avg_amount = (trade.amount_min + trade.amount_max) / 2.0;
        if avg_amount >= 1_000_000.0 {
            score += 0.3;
        } else if avg_amount >= 500_000.0 {
            score += 0.2;
        } else if avg_amount >= 100_000.0 {
            score += 0.1;
        } else if avg_amount >= 50_000.0 {
            score += 0.05;
        }
        
        match trade.transaction_type.to_lowercase().as_str() {
            "buy" | "purchase" => score += 0.15,
            "sell" | "sale" => score += 0.10,
            _ => {}
        }
        
        match trade.owner.to_lowercase().as_str() {
            "self" => score += 0.10,
            "spouse" => score += 0.08,
            "child" => score += 0.05,
            _ => {}
        }
        
        if trade.disclosure_lag_days <= 15 {
            score += 0.05;
        } else if trade.disclosure_lag_days > 45 {
            score += 0.10;
        }
        
        score.min(1.0)
    }

    fn analyze_symbols(&self) -> HashMap<String, SymbolAnalysis> {
        let mut symbol_map: HashMap<String, Vec<&DetailedCongressTrade>> = HashMap::new();
        
        for trade in &self.all_trades {
            // Skip N/A tickers (private companies) in symbol analysis
            if trade.ticker != "N/A" {
                symbol_map.entry(trade.ticker.clone()).or_default().push(trade);
            }
        }
        
        let mut analysis_map = HashMap::new();
        
        for (symbol, trades) in symbol_map {
            let total_trades = trades.len();
            let purchase_count = trades.iter().filter(|t| t.transaction_type.to_lowercase().contains("buy") || t.transaction_type.to_lowercase().contains("purchase")).count();
            let sale_count = trades.iter().filter(|t| t.transaction_type.to_lowercase().contains("sell") || t.transaction_type.to_lowercase().contains("sale")).count();
            
            let avg_disclosure_lag = trades.iter().map(|t| t.disclosure_lag_days as f64).sum::<f64>() / total_trades as f64;
            let avg_impact_score = trades.iter().map(|t| t.impact_score).sum::<f64>() / total_trades as f64;
            
            let mut rep_activity: HashMap<String, (usize, f64)> = HashMap::new();
            for trade in &trades {
                let (count, score_sum) = rep_activity.entry(trade.representative.clone()).or_insert((0, 0.0));
                *count += 1;
                *score_sum += trade.impact_score;
            }
            
            let mut top_representatives: Vec<RepresentativeActivity> = rep_activity
                .into_iter()
                .map(|(name, (count, score_sum))| RepresentativeActivity {
                    name,
                    trade_count: count,
                    avg_impact: score_sum / count as f64,
                })
                .collect();
            
            top_representatives.sort_by(|a, b| b.trade_count.cmp(&a.trade_count));
            top_representatives.truncate(5);
            
            let mut amount_dist = AmountDistribution {
                under_15k: 0,
                from_15k_to_50k: 0,
                from_50k_to_100k: 0,
                from_100k_to_250k: 0,
                over_250k: 0,
            };
            
            for trade in &trades {
                let avg_amount = (trade.amount_min + trade.amount_max) / 2.0;
                if avg_amount < 15_000.0 {
                    amount_dist.under_15k += 1;
                } else if avg_amount < 50_000.0 {
                    amount_dist.from_15k_to_50k += 1;
                } else if avg_amount < 100_000.0 {
                    amount_dist.from_50k_to_100k += 1;
                } else if avg_amount < 250_000.0 {
                    amount_dist.from_100k_to_250k += 1;
                } else {
                    amount_dist.over_250k += 1;
                }
            }
            
            let analysis = SymbolAnalysis {
                symbol: symbol.clone(),
                total_trades,
                purchase_count,
                sale_count,
                avg_disclosure_lag,
                avg_impact_score,
                top_representatives,
                amount_distribution: amount_dist,
            };
            
            analysis_map.insert(symbol, analysis);
        }
        
        analysis_map
    }

    async fn save_detailed_files(&self, symbol_analysis: &HashMap<String, SymbolAnalysis>) -> Result<()> {
        use std::fs;
        
        let detailed_json = serde_json::to_string_pretty(&self.all_trades)?;
        fs::write("congressional_trades_detailed.json", detailed_json)?;
        info!("💾 Saved {} detailed trades to congressional_trades_detailed.json", self.all_trades.len());
        
        let analysis_json = serde_json::to_string_pretty(symbol_analysis)?;
        fs::write("symbol_analysis.json", analysis_json)?;
        info!("💾 Saved analysis for {} symbols to symbol_analysis.json", symbol_analysis.len());
        
        let target_symbols: Vec<String> = symbol_analysis
            .iter()
            .filter(|(_, analysis)| {
                analysis.avg_impact_score > 0.5 && analysis.total_trades >= 3
            })
            .map(|(symbol, _)| symbol.clone())
            .collect();
        
        let targets_json = serde_json::to_string_pretty(&target_symbols)?;
        fs::write("target_symbols.json", targets_json)?;
        info!("💾 Saved {} target symbols to target_symbols.json", target_symbols.len());
        
        Ok(())
    }

    fn to_alt_data_event(&self, trade: &DetailedCongressTrade) -> AltDataEvent {
        let timestamp = NaiveDate::parse_from_str(&trade.transaction_date, "%Y-%m-%d")
            .map(|date| date.and_hms_opt(16, 0, 0).unwrap().and_utc())
            .unwrap_or_else(|_| Utc::now());

        let description = format!(
            "{} {} {} {} ({})",
            trade.representative,
            trade.transaction_type,
            trade.ticker,
            trade.owner,
            trade.amount
        );

        let metadata = serde_json::json!({
            "representative": trade.representative,
            "transaction_type": trade.transaction_type,
            "owner": trade.owner,
            "amount": trade.amount,
            "amount_min": trade.amount_min,
            "amount_max": trade.amount_max,
            "transaction_date": trade.transaction_date,
            "disclosure_date": trade.disclosure_date,
            "disclosure_lag_days": trade.disclosure_lag_days,
            "ptr_link": trade.ptr_link,
            "district": trade.district
        });

        AltDataEvent {
            id: Uuid::new_v4(),
            event_type: AltDataType::CongressionalTrade,
            symbol: Some(trade.ticker.clone()),
            description,
            impact_score: trade.impact_score,
            timestamp,
            metadata,
        }
    }

    pub async fn get_target_symbols(&mut self) -> Result<Vec<String>> {
        self.collect_all_trades().await?;
        
        let target_symbols: Vec<String> = self.all_trades
            .iter()
            .filter(|trade| trade.impact_score > 0.5)
            .map(|trade| trade.ticker.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        
        Ok(target_symbols)
    }

    pub async fn get_recent_trades(&mut self, days: i64) -> Result<Vec<AltDataEvent>> {
        let events = self.collect_all_trades().await?;
        let cutoff = Utc::now() - chrono::Duration::days(days);
        
        let recent_events: Vec<AltDataEvent> = events
            .into_iter()
            .filter(|event| event.timestamp > cutoff)
            .collect();
        
        info!("🕒 Found {} trades within last {} days", recent_events.len(), days);
        Ok(recent_events)
    }

    pub async fn get_important_dates(&mut self, symbol: &str) -> Result<Vec<DateTime<Utc>>> {
        let events = self.collect_all_trades().await?;
        
        let mut dates = Vec::new();
        for event in events {
            if let Some(event_symbol) = &event.symbol {
                if event_symbol == symbol && event.impact_score > 0.5 {
                    dates.push(event.timestamp);
                }
            }
        }
        
        dates.sort();
        info!("📅 Found {} important dates for {}", dates.len(), symbol);
        Ok(dates)
    }

    pub fn get_all_detailed_trades(&self) -> &[DetailedCongressTrade] {
        &self.all_trades
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_amount_range() {
        // Test standard dollar range format
        assert_eq!(FreeCongressTracker::parse_amount_range_static("$1,001 - $15,000"), (1001.0, 15000.0));
        
        // Test K notation format
        assert_eq!(FreeCongressTracker::parse_amount_range_static("50K - 100K"), (50000.0, 100000.0));
        assert_eq!(FreeCongressTracker::parse_amount_range_static("15K–50K"), (15000.0, 50000.0));
        
        // Test the problematic raw number format like "6648-93318"
        assert_eq!(FreeCongressTracker::parse_amount_range_static("6648-93318"), (6648.0, 93318.0));
        assert_eq!(FreeCongressTracker::parse_amount_range_static("1000-15000"), (1000.0, 15000.0));
        
        // Test single value (creates range around it)
        assert_eq!(FreeCongressTracker::parse_amount_range_static("$100,000"), (80000.0, 120000.0));
        assert_eq!(FreeCongressTracker::parse_amount_range_static("50000"), (40000.0, 60000.0));
    }
}
