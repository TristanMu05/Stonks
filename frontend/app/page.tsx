import LiveTickers from "./components/LiveTickers";

export default function Page() {
  return (
    <main className="min-h-dvh p-4 md:p-8">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl md:text-3xl font-bold">Stonks Live Tickers</h1>
        <span className="text-sm text-neutral-400">SSE from data_collector</span>
      </header>
      <LiveTickers />
    </main>
  );
}


