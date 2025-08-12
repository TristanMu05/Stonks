import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Stonks Live Tickers",
  description: "Real-time market data from Finnhub",
  icons: { icon: "/icon.svg" },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="bg-neutral-950 text-neutral-100 antialiased">{children}</body>
    </html>
  );
}


