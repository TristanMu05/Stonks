"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { defaultVisibleColumns, tickerColumns } from "../lib/tickerColumns";
import type { TickerColumnKey } from "../lib/types";

const STORAGE_KEY = "stonks.ticker-columns.v1";

const allColumnKeys = tickerColumns.map((column) => column.key);

const sanitizeColumns = (value: TickerColumnKey[]) =>
  value.filter((column) => allColumnKeys.includes(column));

export const useTickerSettings = () => {
  const [visibleColumns, setVisibleColumns] =
    useState<TickerColumnKey[]>(defaultVisibleColumns);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      return;
    }
    try {
      const parsed = JSON.parse(stored) as TickerColumnKey[];
      const sanitized = sanitizeColumns(parsed);
      if (sanitized.length > 0) {
        setVisibleColumns(sanitized);
      }
    } catch {
      setVisibleColumns(defaultVisibleColumns);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(visibleColumns));
  }, [visibleColumns]);

  const updateColumn = useCallback((key: TickerColumnKey, enabled: boolean) => {
    setVisibleColumns((current) => {
      if (enabled && !current.includes(key)) {
        return [...current, key];
      }
      if (!enabled) {
        return current.filter((column) => column !== key);
      }
      return current;
    });
  }, []);

  const selectAll = useCallback(() => {
    setVisibleColumns(allColumnKeys);
  }, []);

  const clearAll = useCallback(() => {
    setVisibleColumns([]);
  }, []);

  const resetDefaults = useCallback(() => {
    setVisibleColumns(defaultVisibleColumns);
  }, []);

  const visibleCount = visibleColumns.length;

  const columnGroups = useMemo(() => {
    const groups = new Map<string, TickerColumnKey[]>();
    tickerColumns.forEach((column) => {
      if (!groups.has(column.group)) {
        groups.set(column.group, []);
      }
      groups.get(column.group)?.push(column.key);
    });
    return groups;
  }, []);

  return {
    visibleColumns,
    visibleCount,
    updateColumn,
    selectAll,
    clearAll,
    resetDefaults,
    columnGroups,
  };
};
