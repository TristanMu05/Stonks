import type { TickerColumn, TickerColumnKey } from "./types";
import {
  formatCompact,
  formatCurrency,
  formatInteger,
  formatNumber,
  formatPercent,
  formatRange,
  formatTimestamp,
} from "./format";

export const tickerColumns: TickerColumn[] = [
  {
    key: "symbol",
    label: "Symbol",
    description: "Ticker symbol and company name",
    group: "Identity",
    align: "left",
    widthClass: "min-w-[180px]",
    render: (ticker) => (
      <div className="flex flex-col">
        <span className="text-sm font-semibold">{ticker.symbol}</span>
        <span className="text-xs text-slate-400">{ticker.name}</span>
      </div>
    ),
  },
  {
    key: "exchange",
    label: "Exchange",
    description: "Primary listing exchange",
    group: "Identity",
    align: "left",
    render: (ticker) => ticker.exchange,
  },
  {
    key: "sector",
    label: "Sector",
    description: "GICS sector",
    group: "Identity",
    align: "left",
    render: (ticker) => ticker.sector,
  },
  {
    key: "industry",
    label: "Industry",
    description: "Industry classification",
    group: "Identity",
    align: "left",
    widthClass: "min-w-[200px]",
    render: (ticker) => ticker.industry,
  },
  {
    key: "last",
    label: "Last",
    description: "Last traded price",
    group: "Price",
    align: "right",
    render: (ticker) => formatCurrency(ticker.last),
  },
  {
    key: "mark",
    label: "Mark",
    description: "Midpoint between bid/ask",
    group: "Price",
    align: "right",
    render: (ticker) => formatCurrency(ticker.mark),
  },
  {
    key: "change",
    label: "Net Chg",
    description: "Change from previous close",
    group: "Price",
    align: "right",
    render: (ticker) => (
      <span className={ticker.change >= 0 ? "text-emerald-400" : "text-rose-400"}>
        {formatCurrency(ticker.change)}
      </span>
    ),
  },
  {
    key: "changePercent",
    label: "Chg %",
    description: "Percent change from previous close",
    group: "Price",
    align: "right",
    render: (ticker) => (
      <span
        className={
          ticker.changePercent >= 0 ? "text-emerald-400" : "text-rose-400"
        }
      >
        {formatPercent(ticker.changePercent)}
      </span>
    ),
  },
  {
    key: "bid",
    label: "Bid",
    description: "Current bid price",
    group: "Order Book",
    align: "right",
    render: (ticker) => formatCurrency(ticker.bid),
  },
  {
    key: "ask",
    label: "Ask",
    description: "Current ask price",
    group: "Order Book",
    align: "right",
    render: (ticker) => formatCurrency(ticker.ask),
  },
  {
    key: "bidSize",
    label: "Bid Size",
    description: "Bid size (shares)",
    group: "Order Book",
    align: "right",
    render: (ticker) => formatInteger(ticker.bidSize),
  },
  {
    key: "askSize",
    label: "Ask Size",
    description: "Ask size (shares)",
    group: "Order Book",
    align: "right",
    render: (ticker) => formatInteger(ticker.askSize),
  },
  {
    key: "spread",
    label: "Spread",
    description: "Bid/ask spread",
    group: "Order Book",
    align: "right",
    render: (ticker) => formatCurrency(ticker.ask - ticker.bid),
  },
  {
    key: "spreadPercent",
    label: "Spread %",
    description: "Bid/ask spread as percent of mark",
    group: "Order Book",
    align: "right",
    render: (ticker) => {
      const spread = ticker.ask - ticker.bid;
      const percent = ticker.mark === 0 ? 0 : (spread / ticker.mark) * 100;
      return formatPercent(percent);
    },
  },
  {
    key: "volume",
    label: "Volume",
    description: "Current session volume",
    group: "Liquidity",
    align: "right",
    render: (ticker) => formatCompact(ticker.volume),
  },
  {
    key: "avgVolume",
    label: "Avg Vol",
    description: "30-day average volume",
    group: "Liquidity",
    align: "right",
    render: (ticker) => formatCompact(ticker.avgVolume),
  },
  {
    key: "relativeVolume",
    label: "Rel Vol",
    description: "Volume vs average volume",
    group: "Liquidity",
    align: "right",
    render: (ticker) => {
      const relative =
        ticker.avgVolume === 0 ? 0 : ticker.volume / ticker.avgVolume;
      return `${formatNumber(relative)}x`;
    },
  },
  {
    key: "open",
    label: "Open",
    description: "Session open price",
    group: "Price",
    align: "right",
    render: (ticker) => formatCurrency(ticker.open),
  },
  {
    key: "high",
    label: "High",
    description: "Session high price",
    group: "Price",
    align: "right",
    render: (ticker) => formatCurrency(ticker.high),
  },
  {
    key: "low",
    label: "Low",
    description: "Session low price",
    group: "Price",
    align: "right",
    render: (ticker) => formatCurrency(ticker.low),
  },
  {
    key: "prevClose",
    label: "Prev Close",
    description: "Previous close price",
    group: "Price",
    align: "right",
    render: (ticker) => formatCurrency(ticker.prevClose),
  },
  {
    key: "vwap",
    label: "VWAP",
    description: "Volume weighted average price",
    group: "Price",
    align: "right",
    render: (ticker) => formatCurrency(ticker.vwap),
  },
  {
    key: "dayRange",
    label: "Day Range",
    description: "Intraday low to high range",
    group: "Price",
    align: "right",
    widthClass: "min-w-[170px]",
    render: (ticker) => formatRange(ticker.dayRange[0], ticker.dayRange[1]),
  },
  {
    key: "week52Range",
    label: "52W Range",
    description: "52-week low to high range",
    group: "Price",
    align: "right",
    widthClass: "min-w-[170px]",
    render: (ticker) => formatRange(ticker.week52Range[0], ticker.week52Range[1]),
  },
  {
    key: "marketCap",
    label: "Mkt Cap",
    description: "Market capitalization",
    group: "Fundamentals",
    align: "right",
    render: (ticker) =>
      ticker.marketCap ? formatCompact(ticker.marketCap) : "--",
  },
  {
    key: "float",
    label: "Float",
    description: "Public float (shares)",
    group: "Fundamentals",
    align: "right",
    render: (ticker) => (ticker.float ? formatCompact(ticker.float) : "--"),
  },
  {
    key: "beta",
    label: "Beta",
    description: "Beta vs market",
    group: "Fundamentals",
    align: "right",
    render: (ticker) => formatNumber(ticker.beta),
  },
  {
    key: "pe",
    label: "P/E",
    description: "Price to earnings ratio",
    group: "Fundamentals",
    align: "right",
    render: (ticker) => (ticker.pe ? formatNumber(ticker.pe) : "--"),
  },
  {
    key: "eps",
    label: "EPS",
    description: "Earnings per share (TTM)",
    group: "Fundamentals",
    align: "right",
    render: (ticker) => (ticker.eps ? formatNumber(ticker.eps) : "--"),
  },
  {
    key: "dividend",
    label: "Div",
    description: "Annual dividend per share",
    group: "Fundamentals",
    align: "right",
    render: (ticker) => (ticker.dividend ? formatCurrency(ticker.dividend) : "--"),
  },
  {
    key: "dividendYield",
    label: "Div Yield",
    description: "Dividend yield",
    group: "Fundamentals",
    align: "right",
    render: (ticker) =>
      ticker.dividendYield ? formatPercent(ticker.dividendYield) : "--",
  },
  {
    key: "earningsDate",
    label: "Earnings",
    description: "Next earnings date",
    group: "Corporate",
    align: "left",
    render: (ticker) => ticker.earningsDate,
  },
  {
    key: "iv",
    label: "IV",
    description: "Implied volatility",
    group: "Volatility",
    align: "right",
    render: (ticker) => formatPercent(ticker.iv),
  },
  {
    key: "ivPercentile",
    label: "IV %ile",
    description: "Implied volatility percentile",
    group: "Volatility",
    align: "right",
    render: (ticker) => formatPercent(ticker.ivPercentile),
  },
  {
    key: "shortInterest",
    label: "Short Int",
    description: "Short interest (shares)",
    group: "Sentiment",
    align: "right",
    render: (ticker) => formatCompact(ticker.shortInterest),
  },
  {
    key: "shortFloat",
    label: "Short Float",
    description: "Short float percentage",
    group: "Sentiment",
    align: "right",
    render: (ticker) => formatPercent(ticker.shortFloat),
  },
  {
    key: "timestamp",
    label: "Last Update",
    description: "Quote timestamp",
    group: "Metadata",
    align: "right",
    render: (ticker) => formatTimestamp(ticker.timestamp),
  },
];

export const defaultVisibleColumns: TickerColumnKey[] = tickerColumns.map(
  (column) => column.key,
);
