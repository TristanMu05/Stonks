const DEFAULT_DATA_COLLECTOR_URL = "http://localhost:8090";

export const getDataCollectorBaseUrl = () =>
  process.env.NEXT_PUBLIC_DATA_COLLECTOR_URL ?? DEFAULT_DATA_COLLECTOR_URL;

export const buildDataCollectorUrl = (
  path: string,
  params?: Record<string, string | number | undefined>,
) => {
  const base = getDataCollectorBaseUrl();
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  const normalizedPath = path.replace(/^\//, "");
  const url = new URL(normalizedPath, normalizedBase);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        url.searchParams.set(key, String(value));
      }
    });
  }
  return url;
};
