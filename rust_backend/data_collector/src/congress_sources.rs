//! Congressional data sources - pluggable system for different data providers
//! Primary: Capitol Trades (HTML scraping with detail page parsing)
//! Fallback: Senate GitHub (JSON from GitHub mirror)

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use chrono::NaiveDate;
use regex::Regex;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, ACCEPT_LANGUAGE, CACHE_CONTROL, PRAGMA, USER_AGENT};
use scraper::{Html, Selector};
use serde::Deserialize;
use std::{collections::HashSet, time::Duration};
use tracing::{info, warn, debug};

#[derive(Debug, Clone)]
pub struct NormalizedTrade {
    pub representative: String,
    pub ticker: Option<String>,
    pub transaction_type: Option<String>, // buy or sell if available
    pub amount: Option<String>,           // "$1K–$15K" etc
    pub owner: Option<String>,            // "self", "spouse" etc
    pub traded_date: Option<NaiveDate>,
    pub disclosed_date: Option<NaiveDate>,
    pub source: &'static str,
    pub detail_url: Option<String>,
}

#[async_trait]
pub trait CongressSource: Send + Sync {
    async fn fetch_all(&self, max_pages: usize) -> Result<Vec<NormalizedTrade>>;
}

fn default_headers() -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert(USER_AGENT, HeaderValue::from_static("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"));
    h.insert(ACCEPT, HeaderValue::from_static("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"));
    h.insert(ACCEPT_LANGUAGE, HeaderValue::from_static("en-US,en;q=0.9"));
    h.insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    h.insert(PRAGMA, HeaderValue::from_static("no-cache"));
    h
}

/* ----------------------- CapitolTradesSource ------------------------ */

pub struct CapitolTradesSource {
    client: reqwest::Client,
}

impl CapitolTradesSource {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .default_headers(default_headers())
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap();
        Self { client }
    }

    async fn fetch_list_page(&self, page: usize) -> Result<String> {
        let url = format!("https://www.capitoltrades.com/trades?page={page}");
        let res = self.client.get(&url).send().await?;
        if !res.status().is_success() {
            return Err(anyhow!("CapitolTrades list status {}", res.status()));
        }
        Ok(res.text().await?)
    }

    /// Extract detail links like /trades/20003789699 from the list HTML.
    fn extract_detail_links(&self, html: &str) -> Vec<String> {
        // Use regex so we are resilient to CSS changes
        let re = Regex::new(r#"href\s*=\s*["'](/trades/\d{6,})["']"#).unwrap();
        let mut uniq = HashSet::new();
        let mut out = Vec::new();
        for cap in re.captures_iter(html) {
            if let Some(m) = cap.get(1) {
                let link = m.as_str().to_string();
                if uniq.insert(link.clone()) {
                    out.push(link);
                }
            }
        }
        out
    }

    async fn parse_detail(&self, url_path: &str) -> Result<NormalizedTrade> {
        let url = format!("https://www.capitoltrades.com{url_path}");
        let res = self.client.get(&url).send().await?;
        if !res.status().is_success() {
            return Err(anyhow!("CapitolTrades detail status {} {}", res.status(), url));
        }
        let body = res.text().await?;
        let doc = Html::parse_document(&body);

        debug!("Parsing detail page: {}", url);

        // Extract representative name - try multiple selectors
        let representative = self.extract_representative(&doc, &body);
        
        // Extract ticker - try multiple patterns
        let ticker = self.extract_ticker(&body);
        
        // Extract transaction type
        let transaction_type = self.extract_transaction_type(&body);
        
        // Extract amount with improved parsing
        let amount = self.extract_amount(&body);
        
        // Extract owner
        let owner = self.extract_owner(&body);
        
        // Extract dates with improved parsing
        let (traded_date, disclosed_date) = self.extract_dates(&body);

        debug!("Extracted data for {}: rep={}, ticker={:?}, type={:?}, amount={:?}, owner={:?}, traded={:?}, disclosed={:?}", 
               url, representative, ticker, transaction_type, amount, owner, traded_date, disclosed_date);

        Ok(NormalizedTrade {
            representative,
            ticker,
            transaction_type,
            amount,
            owner,
            traded_date,
            disclosed_date,
            source: "capitol_trades",
            detail_url: Some(url),
        })
    }

    fn extract_representative(&self, doc: &Html, body: &str) -> String {
        // Try multiple selectors for representative name
        let selectors = [
            "a[href^=\"/politicians/\"]",
            ".politician-name",
            ".representative",
            "[class*=\"politician\"]",
        ];

        for selector_str in &selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = doc.select(&selector).next() {
                    let text = element.text().collect::<String>().trim().to_string();
                    if !text.is_empty() && text != "Unknown" {
                        return text;
                    }
                }
            }
        }

        // Fallback: look for common patterns in the raw HTML
        let patterns = [
            r"politician[^>]*>([^<]+)",
            r"representative[^>]*>([^<]+)",
            "/politicians/[^\"]*\">([^<]+)",
        ];

        for pattern in &patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(cap) = re.captures(body) {
                    if let Some(m) = cap.get(1) {
                        let name = m.as_str().trim().to_string();
                        if !name.is_empty() {
                            return name;
                        }
                    }
                }
            }
        }

        "Unknown".to_string()
    }

    fn extract_ticker(&self, body: &str) -> Option<String> {
        // First check if this is a private company that shouldn't have a ticker
        if self.is_private_company(body) {
            debug!("Detected private company, no ticker available");
            return None;
        }

        // Try exchange-specific patterns first (most reliable)
        let exchange_patterns = [
            r"([A-Z]{1,5}):US",           // NVDA:US format
            r"([A-Z]{1,5}):NASDAQ",      // NVDA:NASDAQ format  
            r"([A-Z]{1,5}):NYSE",        // NVDA:NYSE format
            r"([A-Z]{1,5}):AMEX",        // AMEX format
        ];

        for pattern in &exchange_patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(cap) = re.captures(body) {
                    if let Some(m) = cap.get(1) {
                        let ticker = m.as_str().to_string();
                        if self.is_valid_ticker(&ticker) {
                            debug!("Found exchange-formatted ticker: {}", ticker);
                            return Some(ticker);
                        }
                    }
                }
            }
        }

        // Try ticker field patterns
        let field_patterns = [
            r"ticker[^>]*>([A-Z]{1,5})",  // ticker field
            r"symbol[^>]*>([A-Z]{1,5})",  // symbol field
        ];

        for pattern in &field_patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(cap) = re.captures(body) {
                    if let Some(m) = cap.get(1) {
                        let ticker = m.as_str().to_string();
                        if self.is_valid_ticker(&ticker) && !self.is_company_name_part(body, &ticker) {
                            debug!("Found field ticker: {}", ticker);
                            return Some(ticker);
                        }
                    }
                }
            }
        }

        // REMOVED: Standalone ticker pattern as it's too prone to false positives
        // We're being conservative - only extract tickers when we're confident

        debug!("No reliable ticker found");
        None
    }

    fn is_valid_ticker(&self, ticker: &str) -> bool {
        // Basic validation: 2-5 uppercase letters
        ticker.len() >= 2 && ticker.len() <= 5 && ticker.chars().all(|c| c.is_ascii_uppercase())
    }

    fn is_company_name_part(&self, body: &str, potential_ticker: &str) -> bool {
        // Check if the potential ticker appears in the context of a company name
        let company_suffixes = [
            "INC", "LLC", "CORP", "LTD", "GROUP", "HOLDINGS", "COMPANY", "CO", 
            "PARTNERS", "FUND", "TRUST", "REIT", "TECHNOLOGIES", "TECH", "SYSTEMS"
        ];

        // Check if the ticker appears adjacent to company suffixes
        for suffix in &company_suffixes {
            let patterns = [
                format!(r"{}\s*-?\s*{}", potential_ticker, suffix),
                format!(r"{}\s*-?\s*{}", suffix, potential_ticker),
                format!(r"{}\s+{}\s", potential_ticker, suffix),
                format!(r"{}\s+{}\s", suffix, potential_ticker),
            ];

            for pattern in &patterns {
                if let Ok(re) = Regex::new(&pattern) {
                    if re.is_match(body) {
                        debug!("Rejected {} - appears to be part of company name", potential_ticker);
                        return true;
                    }
                }
            }
        }

        false
    }

    fn is_private_company(&self, body: &str) -> bool {
        // Detect patterns that indicate private companies
        let private_indicators = [
            r"(?i)(private|privately.held)",
            r"(?i)(pre.ipo|pre-ipo)",
            r"(?i)(not.publicly.traded)",
            r"(?i)(startup|early.stage)",
            r"INC\s*-[A-Z]+",  // Pattern like "FIGMA INC -REDH"
            r"LLC\s*-[A-Z]+",  // Pattern like "COMPANY LLC -ABCD"
            r"CORP\s*-[A-Z]+", // Pattern like "COMPANY CORP -ABCD"
        ];

        for pattern in &private_indicators {
            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(body) {
                    debug!("Detected private company indicator: {}", pattern);
                    return true;
                }
            }
        }

        false
    }

    fn extract_transaction_type(&self, body: &str) -> Option<String> {
        let patterns = [
            r"(?i)\b(buy|purchase|bought)\b",
            r"(?i)\b(sell|sale|sold)\b",
            r"(?i)\b(exchange)\b",
            r"transaction[^>]*>(buy|sell|purchase|sale)",
        ];

        for pattern in &patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(cap) = re.captures(body) {
                    if let Some(m) = cap.get(1) {
                        let tx_type = m.as_str().to_lowercase();
                        return Some(match tx_type.as_str() {
                            "bought" | "purchase" => "buy".to_string(),
                            "sold" | "sale" => "sell".to_string(),
                            _ => tx_type,
                        });
                    }
                }
            }
        }

        None
    }

    fn extract_amount(&self, body: &str) -> Option<String> {
        let patterns = [
            // Standard formats like $1,001 - $15,000
            r"(\$[\d,]+\s*(?:–|-|to)\s*\$[\d,]+)",
            // K notation like 1K–15K  
            r"(\d+K\s*(?:–|-|to)\s*\d+K)",
            // Number ranges like 6648-93318 (the problematic format)
            r"(\d{4,}-\d{4,})",
            // Dollar ranges without commas
            r"(\$\d+\s*(?:–|-|to)\s*\$\d+)",
            // Size or amount fields
            r"(?:size|amount)[^>]*>([^<]+)",
        ];

        for pattern in &patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(cap) = re.captures(body) {
                    if let Some(m) = cap.get(1) {
                        let amount = m.as_str().trim().to_string();
                        if !amount.is_empty() {
                            return Some(amount);
                        }
                    }
                }
            }
        }

        None
    }

    fn extract_owner(&self, body: &str) -> Option<String> {
        let patterns = [
            r"(?i)\b(spouse|self|child|undisclosed\s+owner)\b",
            r"owner[^>]*>([^<]+)",
        ];

        for pattern in &patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(cap) = re.captures(body) {
                    if let Some(m) = cap.get(1) {
                        return Some(m.as_str().trim().to_string());
                    }
                }
            }
        }

        None
    }

    fn extract_dates(&self, body: &str) -> (Option<NaiveDate>, Option<NaiveDate>) {
        let traded = self.extract_date_flexible(body, &["traded", "transaction"]);
        let disclosed = self.extract_date_flexible(body, &["published", "disclosed", "disclosure"]);
        (traded, disclosed)
    }

    fn extract_date_flexible(&self, body: &str, labels: &[&str]) -> Option<NaiveDate> {
        for label in labels {
            // Try multiple date formats and patterns
            let patterns = [
                // ISO format: label followed by 2023-12-25
                format!(r"(?i){}[^>]*>([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})", label),
                // Readable format: label followed by 21 Jul 2025
                format!(r"(?i){}[^>]*>([0-3]?\d\s+[A-Za-z]{{3}}\s+20\d\d)", label),
                // Date in nearby text: 2023-12-25 near the label
                format!(r"(?i){}.*?([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})", label),
                // Alternative readable format
                format!(r"(?i){}.*?([0-3]?\d\s+[A-Za-z]{{3}}\s+20\d\d)", label),
            ];

            for pattern in &patterns {
                if let Ok(re) = Regex::new(pattern) {
                    if let Some(cap) = re.captures(body) {
                        if let Some(m) = cap.get(1) {
                            let date_str = m.as_str().trim();
                            
                            // Try parsing as ISO format
                            if let Ok(date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                                debug!("Parsed {} date: {} -> {}", label, date_str, date);
                                return Some(date);
                            }
                            
                            // Try parsing as readable format
                            if let Ok(date) = NaiveDate::parse_from_str(date_str, "%d %b %Y") {
                                debug!("Parsed {} date: {} -> {}", label, date_str, date);
                                return Some(date);
                            }
                        }
                    }
                }
            }
        }

        None
    }
}

#[async_trait]
impl CongressSource for CapitolTradesSource {
    async fn fetch_all(&self, max_pages: usize) -> Result<Vec<NormalizedTrade>> {
        info!("🏛️ CapitolTrades: scraping up to {max_pages} pages");
        let mut out = Vec::new();
        let mut pages_scraped = 0usize;
        let mut seen_urls = HashSet::new(); // Deduplication

        for page in 1..=max_pages {
            let html = self.fetch_list_page(page).await?;
            let links = self.extract_detail_links(&html);
            info!("📄 page {page}: found {} trade links", links.len());
            if links.is_empty() {
                if page == 1 {
                    warn!("No trade links found on first page. Selector or anti-bot issue.");
                }
                break;
            }

            for rel in links {
                // Skip if we've already processed this URL (deduplication)
                if !seen_urls.insert(rel.clone()) {
                    debug!("Skipping duplicate URL: {}", rel);
                    continue;
                }

                match self.parse_detail(&rel).await {
                    Ok(trade) => {
                        // Additional deduplication based on content
                        let trade_key = format!("{}|{}|{}|{}", 
                            trade.representative, 
                            trade.ticker.as_deref().unwrap_or(""), 
                            trade.traded_date.map(|d| d.to_string()).unwrap_or_default(),
                            trade.amount.as_deref().unwrap_or("")
                        );
                        
                        if !seen_urls.contains(&trade_key) {
                            seen_urls.insert(trade_key);
                            out.push(trade);
                        } else {
                            debug!("Skipping duplicate trade: {}", trade_key);
                        }
                    }
                    Err(e) => warn!("detail parse failed for {}: {}", rel, e),
                }
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
            pages_scraped += 1;
            tokio::time::sleep(Duration::from_millis(300)).await;
        }

        info!("🎯 CapitolTrades: parsed {} unique trades over {} pages", out.len(), pages_scraped);
        Ok(out)
    }
}

/* ----------------------- SenateGithubSource ------------------------ */

pub struct SenateGithubSource {
    client: reqwest::Client,
}

impl SenateGithubSource {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (compatible; VyrlQuantBot/1.0)")
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap();
        Self { client }
    }
}

#[derive(Debug, Deserialize)]
struct SenateTxn {
    #[serde(rename = "transaction_date")]
    transaction_date: Option<String>,
    #[serde(rename = "owner")]
    owner: Option<String>,
    #[serde(rename = "ticker")]
    ticker: Option<String>,
    #[serde(rename = "type")]
    r#type: Option<String>,
    #[serde(rename = "amount")]
    amount: Option<String>,
    #[serde(rename = "disclosure_date")]
    disclosure_date: Option<String>,
    #[serde(rename = "senator")]
    senator: Option<String>,
}

#[async_trait]
impl CongressSource for SenateGithubSource {
    async fn fetch_all(&self, _max_pages: usize) -> Result<Vec<NormalizedTrade>> {
        // Use the GitHub mirror of the S3 aggregate to avoid 403
        let url = "https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/master/aggregate/all_transactions.json";
        let res = self.client.get(url).send().await?;
        if !res.status().is_success() {
            return Err(anyhow!("Senate GitHub status {}", res.status()));
        }
        let items: Vec<SenateTxn> = res.json().await?;
        let parse_date = |s: &Option<String>| -> Option<NaiveDate> {
            if let Some(v) = s {
                if let Ok(d) = NaiveDate::parse_from_str(v, "%m/%d/%Y") {
                    return Some(d);
                }
                if let Ok(d) = NaiveDate::parse_from_str(v, "%Y-%m-%d") {
                    return Some(d);
                }
            }
            None
        };

        let out: Vec<NormalizedTrade> = items.into_iter().map(|t| {
            NormalizedTrade {
                representative: t.senator.unwrap_or_else(|| "Unknown".to_string()),
                ticker: t.ticker.filter(|x| !x.trim().is_empty()).map(|x| x.replace("--", "")),
                transaction_type: t.r#type.map(|x| x.to_lowercase()),
                amount: t.amount,
                owner: t.owner,
                traded_date: parse_date(&t.transaction_date),
                disclosed_date: parse_date(&t.disclosure_date),
                source: "senate_github",
                detail_url: None,
            }
        }).collect();

        info!("🧰 Senate GitHub fallback: loaded {} trades", out.len());
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_detail_links() {
        let source = CapitolTradesSource::new();
        let mock_html = r#"
            <a href="/trades/20003789699">Trade Details</a>
            <a href="/trades/20003789700">Another Trade</a>
            <a href="/politicians/nancy-pelosi">Nancy Pelosi</a>
            <a href="/trades/20003789699">Duplicate Trade</a>
        "#;
        
        let links = source.extract_detail_links(mock_html);
        assert_eq!(links.len(), 2); // Should deduplicate
        assert!(links.contains(&"/trades/20003789699".to_string()));
        assert!(links.contains(&"/trades/20003789700".to_string()));
    }

    #[test]
    fn test_amount_extraction() {
        let source = CapitolTradesSource::new();
        
        // Test the problematic format from your data
        let body1 = "Some text with amount 6648-93318 in it";
        assert_eq!(source.extract_amount(body1), Some("6648-93318".to_string()));
        
        // Test standard format
        let body2 = "Amount: $1,001 - $15,000";
        assert_eq!(source.extract_amount(body2), Some("$1,001 - $15,000".to_string()));
        
        // Test K format
        let body3 = "Size: 15K–50K";
        assert_eq!(source.extract_amount(body3), Some("15K–50K".to_string()));
    }

    #[test]
    fn test_ticker_extraction() {
        let source = CapitolTradesSource::new();
        
        // Test exchange format (should work)
        let body1 = "Some content with NVDA:US ticker";
        assert_eq!(source.extract_ticker(body1), Some("NVDA".to_string()));
        
        // Test private company (should return None)
        let body2 = "Investment in FIGMA INC -REDH private company";
        assert_eq!(source.extract_ticker(body2), None);
        
        // Test standalone ticker without context (should return None - conservative approach)
        let body3 = "Some text with GTM somewhere in it";
        assert_eq!(source.extract_ticker(body3), None);
        
        // Test ticker field (should work)
        let body4 = "ticker>AAPL< other content";
        assert_eq!(source.extract_ticker(body4), Some("AAPL".to_string()));
    }

    #[test]
    fn test_private_company_detection() {
        let source = CapitolTradesSource::new();
        
        assert!(source.is_private_company("FIGMA INC -REDH"));
        assert!(source.is_private_company("COMPANY LLC -ABCD"));
        assert!(source.is_private_company("privately held company"));
        assert!(!source.is_private_company("APPLE INC with ticker AAPL:US"));
    }
}