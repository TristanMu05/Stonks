//! Alternative data collection
//! Congressional trades, government contracts, news sentiment

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use shared::{Config, AltDataEvent, AltDataType};
use tracing::{info, error, warn};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Congressional trade from QuiverQuant
#[derive(Debug, Deserialize, Serialize)]
pub struct CongressionalTrade {
    #[serde(rename = "Representative")]
    pub politician: String,
    #[serde(rename = "Ticker")]
    pub ticker: String,
    #[serde(rename = "Transaction")]
    pub transaction_type: String,  // "Purchase" or "Sale"
    #[serde(rename = "Amount")]
    pub amount: String,           // "$1,001 - $15,000"
    #[serde(rename = "TransactionDate")]
    pub transaction_date: String,
    #[serde(rename = "ReportDate")]
    pub report_date: String,
}

/// Government contract from USAspending.gov
#[derive(Debug, Deserialize, Serialize)]
pub struct GovernmentContract {
    pub recipient_name: String,
    pub award_amount: f64,
    pub description: String,
    pub award_date: String,
    pub awarding_agency: String,
}

pub struct AltDataCollector {
    config: Arc<Config>,
    client: reqwest::Client,
}

impl AltDataCollector {
    /// Create new alternative data collector
    pub fn new(config: Arc<Config>) -> Self {
        let client = reqwest::Client::new();
        Self { config, client }
    }
    
    /// Start collecting alternative data
    pub async fn run(&self) -> Result<()> {
        info!("🔍 Starting Alternative Data Collection");
        
        // Set up collection intervals
        let mut congressional_timer = tokio::time::interval(
            tokio::time::Duration::from_secs(300)  // Every 5 minutes
        );
        let mut contracts_timer = tokio::time::interval(
            tokio::time::Duration::from_secs(600)  // Every 10 minutes
        );
        
        loop {
            tokio::select! {
                _ = congressional_timer.tick() => {
                    if let Err(e) = self.collect_congressional_trades().await {
                        error!("Failed to collect congressional trades: {}", e);
                    }
                }
                _ = contracts_timer.tick() => {
                    if let Err(e) = self.collect_government_contracts().await {
                        error!("Failed to collect government contracts: {}", e);
                    }
                }
            }
        }
    }
    
    /// Collect recent congressional trades
    async fn collect_congressional_trades(&self) -> Result<()> {
        info!("🏛️ Fetching congressional trades...");
        
        // QuiverQuant API endpoint
        let url = "https://api.quiverquant.com/beta/live/congresstrading";
        
        let response = self.client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.config.quiver_api_key))
            .send()
            .await?;
            
        if !response.status().is_success() {
            return Err(anyhow!("QuiverQuant API error: {}", response.status()));
        }
        
        let trades: Vec<CongressionalTrade> = response.json().await?;
        
        info!("📊 Found {} recent congressional trades", trades.len());
        
        // Process each trade
        for trade in trades {
            let event = self.process_congressional_trade(trade).await?;
            info!("🎯 Congressional Trade: {} {} {} ({})", 
                  event.description, 
                  trade.ticker,
                  trade.transaction_type,
                  trade.amount);
            
            // TODO: Send to database and ML pipeline
        }
        
        Ok(())
    }
    
    /// Process congressional trade into AltDataEvent
    async fn process_congressional_trade(&self, trade: CongressionalTrade) -> Result<AltDataEvent> {
        // Calculate impact score based on politician importance and amount
        let impact_score = self.calculate_trade_impact(&trade);
        
        let description = format!(
            "{} {} {} shares ({})", 
            trade.politician, 
            trade.transaction_type.to_lowercase(),
            trade.ticker,
            trade.amount
        );
        
        let metadata = serde_json::json!({
            "politician": trade.politician,
            "transaction_type": trade.transaction_type,
            "amount": trade.amount,
            "transaction_date": trade.transaction_date,
            "report_date": trade.report_date,
            "days_to_disclosure": self.calculate_disclosure_lag(&trade)
        });
        
        Ok(AltDataEvent {
            id: Uuid::new_v4(),
            event_type: AltDataType::CongressionalTrade,
            symbol: Some(trade.ticker),
            description,
            impact_score,
            timestamp: Utc::now(),
            metadata,
        })
    }
    
    /// Calculate impact score for congressional trade
    fn calculate_trade_impact(&self, trade: &CongressionalTrade) -> f64 {
        let mut score = 0.5; // Base score
        
        // High-profile politicians get higher scores
        if trade.politician.contains("Pelosi") 
            || trade.politician.contains("McCarthy")
            || trade.politician.contains("Schumer") {
            score += 0.3;
        }
        
        // Larger amounts get higher scores
        if trade.amount.contains("$50,000") || trade.amount.contains("$100,000") {
            score += 0.2;
        }
        
        // Purchase vs Sale
        if trade.transaction_type == "Purchase" {
            score += 0.1; // Purchases often more bullish signal
        }
        
        score.min(1.0) // Cap at 1.0
    }
    
    /// Calculate how long it took to disclose the trade
    fn calculate_disclosure_lag(&self, _trade: &CongressionalTrade) -> i32 {
        // TODO: Parse dates and calculate difference
        // For now, return average disclosure time
        35 // ~35 days average
    }
    
    /// Collect recent government contracts
    async fn collect_government_contracts(&self) -> Result<()> {
        info!("📋 Fetching government contracts...");
        
        // USAspending.gov API
        let url = "https://api.usaspending.gov/api/v2/search/spending_by_award/";
        
        let payload = serde_json::json!({
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"], // Contract types
                "time_period": [
                    {
                        "start_date": "2024-01-01",
                        "end_date": "2024-12-31"
                    }
                ]
            },
            "fields": [
                "Award Amount",
                "Recipient Name", 
                "Award Description",
                "Start Date"
            ],
            "sort": "Award Amount",
            "order": "desc",
            "limit": 50
        });
        
        let response = self.client
            .post(url)
            .json(&payload)
            .send()
            .await?;
            
        if !response.status().is_success() {
            warn!("USAspending API error: {}", response.status());
            return Ok(());
        }
        
        let data: serde_json::Value = response.json().await?;
        
        if let Some(results) = data["results"].as_array() {
            info!("📊 Found {} recent contracts", results.len());
            
            for contract in results.iter().take(10) { // Process top 10
                if let Ok(event) = self.process_government_contract(contract).await {
                    info!("🎯 Contract: {} - ${:.1}M", 
                          event.description,
                          contract["Award Amount"].as_f64().unwrap_or(0.0) / 1_000_000.0);
                }
            }
        }
        
        Ok(())
    }
    
    /// Process government contract into AltDataEvent
    async fn process_government_contract(&self, contract: &serde_json::Value) -> Result<AltDataEvent> {
        let recipient = contract["Recipient Name"].as_str().unwrap_or("Unknown");
        let amount = contract["Award Amount"].as_f64().unwrap_or(0.0);
        let description = contract["Award Description"].as_str().unwrap_or("Unknown");
        
        // Try to map company name to stock symbol
        let symbol = self.map_company_to_symbol(recipient);
        
        let impact_score = (amount / 1_000_000_000.0).min(1.0); // $1B = score of 1.0
        
        let event_description = format!("{} awarded ${:.1}M contract", recipient, amount / 1_000_000.0);
        
        Ok(AltDataEvent {
            id: Uuid::new_v4(),
            event_type: AltDataType::GovernmentContract,
            symbol,
            description: event_description,
            impact_score,
            timestamp: Utc::now(),
            metadata: serde_json::json!({
                "recipient": recipient,
                "amount": amount,
                "contract_description": description
            }),
        })
    }
    
    /// Map company name to stock symbol
    fn map_company_to_symbol(&self, company: &str) -> Option<String> {
        // Simple mapping - in production you'd use a comprehensive database
        let company_lower = company.to_lowercase();
        
        if company_lower.contains("lockheed") {
            Some("LMT".to_string())
        } else if company_lower.contains("boeing") {
            Some("BA".to_string())
        } else if company_lower.contains("raytheon") || company_lower.contains("rtx") {
            Some("RTX".to_string())
        } else if company_lower.contains("general dynamics") {
            Some("GD".to_string())
        } else if company_lower.contains("northrop") {
            Some("NOC".to_string())
        } else {
            None // Unknown company
        }
    }
}
EOF# Create the alt_data module
cat > src/alt_data.rs << 'EOF'
//! Alternative data collection
//! Congressional trades, government contracts, news sentiment

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use shared::{Config, AltDataEvent, AltDataType};
use tracing::{info, error, warn};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Congressional trade from QuiverQuant
#[derive(Debug, Deserialize, Serialize)]
pub struct CongressionalTrade {
    #[serde(rename = "Representative")]
    pub politician: String,
    #[serde(rename = "Ticker")]
    pub ticker: String,
    #[serde(rename = "Transaction")]
    pub transaction_type: String,  // "Purchase" or "Sale"
    #[serde(rename = "Amount")]
    pub amount: String,           // "$1,001 - $15,000"
    #[serde(rename = "TransactionDate")]
    pub transaction_date: String,
    #[serde(rename = "ReportDate")]
    pub report_date: String,
}

/// Government contract from USAspending.gov
#[derive(Debug, Deserialize, Serialize)]
pub struct GovernmentContract {
    pub recipient_name: String,
    pub award_amount: f64,
    pub description: String,
    pub award_date: String,
    pub awarding_agency: String,
}

pub struct AltDataCollector {
    config: Arc<Config>,
    client: reqwest::Client,
}

impl AltDataCollector {
    /// Create new alternative data collector
    pub fn new(config: Arc<Config>) -> Self {
        let client = reqwest::Client::new();
        Self { config, client }
    }
    
    /// Start collecting alternative data
    pub async fn run(&self) -> Result<()> {
        info!("🔍 Starting Alternative Data Collection");
        
        // Set up collection intervals
        let mut congressional_timer = tokio::time::interval(
            tokio::time::Duration::from_secs(300)  // Every 5 minutes
        );
        let mut contracts_timer = tokio::time::interval(
            tokio::time::Duration::from_secs(600)  // Every 10 minutes
        );
        
        loop {
            tokio::select! {
                _ = congressional_timer.tick() => {
                    if let Err(e) = self.collect_congressional_trades().await {
                        error!("Failed to collect congressional trades: {}", e);
                    }
                }
                _ = contracts_timer.tick() => {
                    if let Err(e) = self.collect_government_contracts().await {
                        error!("Failed to collect government contracts: {}", e);
                    }
                }
            }
        }
    }
    
    /// Collect recent congressional trades
    async fn collect_congressional_trades(&self) -> Result<()> {
        info!("🏛️ Fetching congressional trades...");
        
        // QuiverQuant API endpoint
        let url = "https://api.quiverquant.com/beta/live/congresstrading";
        
        let response = self.client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.config.quiver_api_key))
            .send()
            .await?;
            
        if !response.status().is_success() {
            return Err(anyhow!("QuiverQuant API error: {}", response.status()));
        }
        
        let trades: Vec<CongressionalTrade> = response.json().await?;
        
        info!("📊 Found {} recent congressional trades", trades.len());
        
        // Process each trade
        for trade in trades {
            let event = self.process_congressional_trade(trade).await?;
            info!("🎯 Congressional Trade: {} {} {} ({})", 
                  event.description, 
                  trade.ticker,
                  trade.transaction_type,
                  trade.amount);
            
            // TODO: Send to database and ML pipeline
        }
        
        Ok(())
    }
    
    /// Process congressional trade into AltDataEvent
    async fn process_congressional_trade(&self, trade: CongressionalTrade) -> Result<AltDataEvent> {
        // Calculate impact score based on politician importance and amount
        let impact_score = self.calculate_trade_impact(&trade);
        
        let description = format!(
            "{} {} {} shares ({})", 
            trade.politician, 
            trade.transaction_type.to_lowercase(),
            trade.ticker,
            trade.amount
        );
        
        let metadata = serde_json::json!({
            "politician": trade.politician,
            "transaction_type": trade.transaction_type,
            "amount": trade.amount,
            "transaction_date": trade.transaction_date,
            "report_date": trade.report_date,
            "days_to_disclosure": self.calculate_disclosure_lag(&trade)
        });
        
        Ok(AltDataEvent {
            id: Uuid::new_v4(),
            event_type: AltDataType::CongressionalTrade,
            symbol: Some(trade.ticker),
            description,
            impact_score,
            timestamp: Utc::now(),
            metadata,
        })
    }
    
    /// Calculate impact score for congressional trade
    fn calculate_trade_impact(&self, trade: &CongressionalTrade) -> f64 {
        let mut score = 0.5; // Base score
        
        // High-profile politicians get higher scores
        if trade.politician.contains("Pelosi") 
            || trade.politician.contains("McCarthy")
            || trade.politician.contains("Schumer") {
            score += 0.3;
        }
        
        // Larger amounts get higher scores
        if trade.amount.contains("$50,000") || trade.amount.contains("$100,000") {
            score += 0.2;
        }
        
        // Purchase vs Sale
        if trade.transaction_type == "Purchase" {
            score += 0.1; // Purchases often more bullish signal
        }
        
        score.min(1.0) // Cap at 1.0
    }
    
    /// Calculate how long it took to disclose the trade
    fn calculate_disclosure_lag(&self, _trade: &CongressionalTrade) -> i32 {
        // TODO: Parse dates and calculate difference
        // For now, return average disclosure time
        35 // ~35 days average
    }
    
    /// Collect recent government contracts
    async fn collect_government_contracts(&self) -> Result<()> {
        info!("📋 Fetching government contracts...");
        
        // USAspending.gov API
        let url = "https://api.usaspending.gov/api/v2/search/spending_by_award/";
        
        let payload = serde_json::json!({
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"], // Contract types
                "time_period": [
                    {
                        "start_date": "2024-01-01",
                        "end_date": "2024-12-31"
                    }
                ]
            },
            "fields": [
                "Award Amount",
                "Recipient Name", 
                "Award Description",
                "Start Date"
            ],
            "sort": "Award Amount",
            "order": "desc",
            "limit": 50
        });
        
        let response = self.client
            .post(url)
            .json(&payload)
            .send()
            .await?;
            
        if !response.status().is_success() {
            warn!("USAspending API error: {}", response.status());
            return Ok(());
        }
        
        let data: serde_json::Value = response.json().await?;
        
        if let Some(results) = data["results"].as_array() {
            info!("📊 Found {} recent contracts", results.len());
            
            for contract in results.iter().take(10) { // Process top 10
                if let Ok(event) = self.process_government_contract(contract).await {
                    info!("🎯 Contract: {} - ${:.1}M", 
                          event.description,
                          contract["Award Amount"].as_f64().unwrap_or(0.0) / 1_000_000.0);
                }
            }
        }
        
        Ok(())
    }
    
    /// Process government contract into AltDataEvent
    async fn process_government_contract(&self, contract: &serde_json::Value) -> Result<AltDataEvent> {
        let recipient = contract["Recipient Name"].as_str().unwrap_or("Unknown");
        let amount = contract["Award Amount"].as_f64().unwrap_or(0.0);
        let description = contract["Award Description"].as_str().unwrap_or("Unknown");
        
        // Try to map company name to stock symbol
        let symbol = self.map_company_to_symbol(recipient);
        
        let impact_score = (amount / 1_000_000_000.0).min(1.0); // $1B = score of 1.0
        
        let event_description = format!("{} awarded ${:.1}M contract", recipient, amount / 1_000_000.0);
        
        Ok(AltDataEvent {
            id: Uuid::new_v4(),
            event_type: AltDataType::GovernmentContract,
            symbol,
            description: event_description,
            impact_score,
            timestamp: Utc::now(),
            metadata: serde_json::json!({
                "recipient": recipient,
                "amount": amount,
                "contract_description": description
            }),
        })
    }
    
    /// Map company name to stock symbol
    fn map_company_to_symbol(&self, company: &str) -> Option<String> {
        // Simple mapping - in production you'd use a comprehensive database
        let company_lower = company.to_lowercase();
        
        if company_lower.contains("lockheed") {
            Some("LMT".to_string())
        } else if company_lower.contains("boeing") {
            Some("BA".to_string())
        } else if company_lower.contains("raytheon") || company_lower.contains("rtx") {
            Some("RTX".to_string())
        } else if company_lower.contains("general dynamics") {
            Some("GD".to_string())
        } else if company_lower.contains("northrop") {
            Some("NOC".to_string())
        } else {
            None // Unknown company
        }
    }
}
