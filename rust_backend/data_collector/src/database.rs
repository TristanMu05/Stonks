//! Database operations for trading system
//! Handles congressional trades, historical prices, and correlation analysis

use anyhow::Result;
use shared::Config;
use sqlx::{PgPool, Row};
use tracing::info;
use std::sync::Arc;
use chrono::NaiveDate;
use uuid::Uuid;
use serde_json::Value;

pub struct TradingDatabase {
    pub pool: PgPool,
}

impl TradingDatabase {
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        let pool = PgPool::connect(&config.database_url).await?;
        
        info!("✅ Database connected");
        Ok(Self { pool })
    }
    
    /// Store congressional trade
    pub async fn store_congressional_trade(&self, trade: &CongressionalTradeRecord) -> Result<Uuid> {
        let id = sqlx::query_scalar::<_, Uuid>(
            r#"
            INSERT INTO congressional_trades (
                representative, district, transaction_date, disclosure_date,
                transaction_type, owner, ticker, asset_description,
                amount, amount_min, amount_max, disclosure_lag_days,
                impact_score, ptr_link
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            RETURNING id
            "#
        )
        .bind(&trade.representative)
        .bind(&trade.district)
        .bind(trade.transaction_date)
        .bind(trade.disclosure_date)
        .bind(&trade.transaction_type)
        .bind(&trade.owner)
        .bind(&trade.ticker)
        .bind(&trade.asset_description)
        .bind(&trade.amount)
        .bind(trade.amount_min)
        .bind(trade.amount_max)
        .bind(trade.disclosure_lag_days)
        .bind(trade.impact_score)
        .bind(&trade.ptr_link)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(id)
    }
    
    /// Store symbol analysis
    pub async fn store_symbol_analysis(&self, analysis: &SymbolAnalysis) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO symbol_analysis (
                symbol, total_trades, purchase_count, sale_count,
                avg_disclosure_lag, avg_impact_score, amount_distribution,
                top_representatives
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (symbol) DO UPDATE SET
                total_trades = $2,
                purchase_count = $3,
                sale_count = $4,
                avg_disclosure_lag = $5,
                avg_impact_score = $6,
                amount_distribution = $7,
                top_representatives = $8,
                last_updated = NOW()
            "#
        )
        .bind(&analysis.symbol)
        .bind(analysis.total_trades)
        .bind(analysis.purchase_count)
        .bind(analysis.sale_count)
        .bind(analysis.avg_disclosure_lag)
        .bind(analysis.avg_impact_score)
        .bind(serde_json::to_value(&analysis.amount_distribution)?)
        .bind(serde_json::to_value(&analysis.top_representatives)?)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    /// Store historical price data
    pub async fn store_historical_price(&self, symbol: &str, date: NaiveDate, 
                                       open: f64, high: f64, low: f64, close: f64, volume: i64) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO historical_prices (symbol, date, open_price, high_price, low_price, close_price, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (symbol, date) DO UPDATE SET
                open_price = $3,
                high_price = $4,
                low_price = $5,
                close_price = $6,
                volume = $7
            "#
        )
        .bind(symbol)
        .bind(date)
        .bind(open)
        .bind(high)
        .bind(low)
        .bind(close)
        .bind(volume)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    /// Get symbols that need historical data
    pub async fn get_target_symbols(&self) -> Result<Vec<String>> {
        let symbols = sqlx::query_scalar::<_, Option<String>>(
            "SELECT DISTINCT ticker FROM congressional_trades WHERE ticker IS NOT NULL AND ticker != 'N/A'"
        )
        .fetch_all(&self.pool)
        .await?;
        
        Ok(symbols.into_iter().filter_map(|s| s).collect())
    }
    
    /// Calculate correlation for a specific trade
    pub async fn calculate_trade_correlation(&self, symbol: &str, transaction_date: NaiveDate, representative: &str) -> Result<Option<Uuid>> {
        let correlation_id = sqlx::query_scalar::<_, Option<Uuid>>(
            "SELECT calculate_trade_correlation($1, $2, $3)",
        )
        .bind(symbol)
        .bind(transaction_date)
        .bind(representative)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(correlation_id)
    }
    
    /// Get top performing representatives
    pub async fn get_representative_performance(&self) -> Result<Vec<RepresentativePerformance>> {
        let results = sqlx::query(
            r#"
            SELECT representative,
                total_trades,
                avg_7d_return::float8 as avg_7d_return,
                return_volatility::float8 as return_volatility,
                win_rate::float8 as win_rate,
                avg_impact_score::float8 as avg_impact_score
            FROM representative_performance
            ORDER BY avg_7d_return DESC
            LIMIT 20
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        let mut performances = Vec::with_capacity(results.len());
        for row in results {
            performances.push(RepresentativePerformance {
                representative: row.try_get::<Option<String>, _>("representative")?.unwrap_or_default(),
                total_trades: row.try_get("total_trades")?,
                avg_7d_return: row.try_get("avg_7d_return")?,
                return_volatility: row.try_get("return_volatility")?,
                win_rate: row.try_get("win_rate")?,
                avg_impact_score: row.try_get("avg_impact_score")?,
            });
        }

        Ok(performances)
    }
    
    /// Find trading patterns
    pub async fn analyze_patterns(&self) -> Result<Vec<TradingPattern>> {
        // High-impact representatives with consistent performance
        let patterns = sqlx::query(
            r#"
            SELECT 
                'high_impact_rep' as pattern_type,
                ct.representative,
                NULL::text as symbol,
                AVG(tc.return_7d)::float8 as avg_return_7d,
                COUNT(*) as sample_size,
                (COUNT(*) FILTER (WHERE tc.return_7d > 0) * 100.0 / COUNT(*))::float8 as win_rate,
                AVG(ct.impact_score)::float8 as confidence_score
            FROM trade_correlations tc
            JOIN congressional_trades ct ON tc.symbol = ct.ticker 
                AND tc.transaction_date = ct.transaction_date
                AND tc.representative = ct.representative
            WHERE ct.impact_score > 0.7
            GROUP BY ct.representative
            HAVING COUNT(*) >= 5 AND AVG(tc.return_7d) > 0.02
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        let mut analyzed = Vec::with_capacity(patterns.len());
        for row in patterns {
            analyzed.push(TradingPattern {
                pattern_type: row.try_get::<Option<String>, _>("pattern_type")?.unwrap_or_default(),
                symbol: row.try_get("symbol")?,
                representative: row.try_get("representative")?,
                avg_return_7d: row.try_get("avg_return_7d")?,
                win_rate: row.try_get("win_rate")?,
                confidence_score: row.try_get("confidence_score")?,
                sample_size: row.try_get("sample_size")?,
            });
        }

        Ok(analyzed)
    }
}

// Data structures
#[derive(Debug)]
pub struct CongressionalTradeRecord {
    pub representative: String,
    pub district: Option<String>,
    pub transaction_date: NaiveDate,
    pub disclosure_date: NaiveDate,
    pub transaction_type: String,
    pub owner: String,
    pub ticker: Option<String>,
    pub asset_description: Option<String>,
    pub amount: String,
    pub amount_min: Option<f64>,
    pub amount_max: Option<f64>,
    pub disclosure_lag_days: Option<i32>,
    pub impact_score: Option<f64>,
    pub ptr_link: Option<String>,
}

#[derive(Debug)]
pub struct SymbolAnalysis {
    pub symbol: String,
    pub total_trades: i32,
    pub purchase_count: i32,
    pub sale_count: i32,
    pub avg_disclosure_lag: Option<f64>,
    pub avg_impact_score: Option<f64>,
    pub amount_distribution: Value,
    pub top_representatives: Value,
}

#[derive(Debug)]
pub struct RepresentativePerformance {
    pub representative: String,
    pub total_trades: Option<i64>,
    pub avg_7d_return: Option<f64>,
    pub return_volatility: Option<f64>,
    pub win_rate: Option<f64>,
    pub avg_impact_score: Option<f64>,
}

#[derive(Debug)]
pub struct TradingPattern {
    pub pattern_type: String,
    pub symbol: Option<String>,
    pub representative: Option<String>,
    pub avg_return_7d: Option<f64>,
    pub win_rate: Option<f64>,
    pub confidence_score: Option<f64>,
    pub sample_size: Option<i64>,
}
