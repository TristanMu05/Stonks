const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
});

const integerFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

const compactFormatter = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 2,
});

const percentFormatter = new Intl.NumberFormat("en-US", {
  style: "percent",
  maximumFractionDigits: 2,
});

export const formatCurrency = (value?: number) => {
  if (value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return currencyFormatter.format(value);
};

export const formatNumber = (value?: number) => {
  if (value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return numberFormatter.format(value);
};

export const formatInteger = (value?: number) => {
  if (value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return integerFormatter.format(value);
};

export const formatCompact = (value?: number) => {
  if (value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return compactFormatter.format(value);
};

export const formatPercent = (value?: number) => {
  if (value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return percentFormatter.format(value / 100);
};

export const formatRange = (low?: number, high?: number) => {
  if (low === undefined || high === undefined) {
    return "--";
  }
  return `${formatCurrency(low)} - ${formatCurrency(high)}`;
};

export const formatTimestamp = (value?: string) => {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};
