import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity, BrainCircuit, KeyRound, LayoutDashboard, Monitor, Moon, Network, PlusCircle, Search, Settings, Sun,
} from "lucide-react";
import { ApiError, api, getToken, setToken } from "./lib/api";
import { useApp, type Route } from "./lib/app";
import { ScopePicker } from "./components/ScopePicker";
import { FactDrawer } from "./components/FactDrawer";
import { Seg } from "./components/ui";
import { Overview } from "./pages/Overview";
import { MemoryPage } from "./pages/Memory";
import { GraphPage } from "./pages/Graph";
import { AddPage, IngestPage } from "./pages/Ingest";
import { AskPage, RecallPage } from "./pages/Ask";
import { ActivityPage } from "./pages/Activity";
import { SettingsPage } from "./pages/Settings";

const NAV: { page: Route["page"]; label: string; icon: JSX.Element }[] = [
  { page: "overview", label: "Overview", icon: <LayoutDashboard /> },
  { page: "memory", label: "Memory", icon: <BrainCircuit /> },
  { page: "graph", label: "Relationships", icon: <Network /> },
  { page: "add", label: "Add memory", icon: <PlusCircle /> },
  { page: "ask", label: "Ask", icon: <Search /> },
  { page: "activity", label: "Activity", icon: <Activity /> },
  { page: "settings", label: "Settings", icon: <Settings /> },
];

const Logo = () => (
  <svg viewBox="0 0 84 84" fill="none" aria-hidden="true">
    <rect width="84" height="84" rx="20" fill="url(#dg)" />
    <circle cx="19" cy="17" r="5.5" stroke="white" />
    <circle cx="63" cy="20" r="5.5" stroke="white" />
    <circle cx="23" cy="62" r="5.5" stroke="white" />
    <circle cx="62" cy="63" r="8.5" stroke="white" />
    <circle cx="38" cy="35" r="8.5" stroke="white" strokeWidth="5" />
    <line x1="30.67" y1="27.38" x2="22.67" y2="20.38" stroke="white" />
    <line x1="45.7" y1="28.6" x2="57.7" y2="19.6" stroke="white" />
    <line x1="45.39" y1="42.69" x2="56.39" y2="56.69" stroke="white" />
    <line x1="26.52" y1="57.85" x2="31.52" y2="41.85" stroke="white" />
    <defs>
      <linearGradient id="dg" x1="75.5" y1="5.5" x2="3" y2="80" gradientUnits="userSpaceOnUse">
        <stop stopColor="#2527D1" />
        <stop offset="1" stopColor="#6F06CC" />
      </linearGradient>
    </defs>
  </svg>
);

function TokenGate() {
  const [value, setValue] = useState(getToken());
  const qc = useQueryClient();
  return (
    <div className="page" style={{ maxWidth: 520, margin: "10vh auto" }}>
      <div className="card card-body stack">
        <div className="row"><KeyRound /><b style={{ fontSize: 16 }}>This memx API needs a token</b></div>
        <p className="muted" style={{ margin: 0 }}>
          Enter the value of <code>MEMX_API_TOKEN</code> set on the server. It is kept in this browser only.
        </p>
        <form className="row" onSubmit={(e) => { e.preventDefault(); setToken(value.trim()); qc.invalidateQueries(); }}>
          <input id="token" className="input" style={{ flex: 1 }} type="password" value={value} onChange={(e) => setValue(e.target.value)} placeholder="API token" autoFocus />
          <button className="btn primary" type="submit">Connect</button>
        </form>
      </div>
    </div>
  );
}

function Page() {
  const { route } = useApp();
  switch (route.page) {
    case "memory": return <MemoryPage />;
    case "graph": return <GraphPage focus={route.focus} />;
    case "add": return <AddPage />;
    case "ask": return <AskPage />;
    case "activity": return <ActivityPage tab={route.tab} />;
    case "ingest": return <IngestPage id={route.id} />;
    case "recall": return <RecallPage id={route.id} />;
    case "settings": return <SettingsPage />;
    default: return <Overview />;
  }
}

export default function App() {
  const { route, go, theme, setTheme } = useApp();
  const ready = useQuery({ queryKey: ["ready"], queryFn: api.ready, refetchInterval: 15_000, retry: false });
  const probe = useQuery({ queryKey: ["config"], queryFn: api.config, retry: false });
  const unauthorized = probe.error instanceof ApiError && probe.error.status === 401;
  const current = route.page === "ingest" ? "activity" : route.page === "recall" ? "activity" : route.page;

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <Logo />
          <div><b>memx</b><span>DataGOL memory console</span></div>
        </div>
        <nav className="nav" aria-label="Main">
          <div className="nav-label">Workspace</div>
          {NAV.map((n) => (
            <button key={n.page} aria-current={current === n.page ? "page" : undefined} onClick={() => go({ page: n.page } as Route)}>
              {n.icon}
              {n.label}
            </button>
          ))}
        </nav>
        <div className="rail-foot">
          <div className="health">
            <span className={`dot ${ready.data ? "ok" : ready.isError ? "bad" : ""}`} />
            {ready.data ? "Database connected" : ready.isError ? "API unreachable" : "Checking…"}
          </div>
          {ready.data && (
            <div className="faint">
              Jobs: {Object.entries(ready.data.jobs).map(([k, v]) => `${v} ${k}`).join(" · ") || "none"}
            </div>
          )}
          {probe.data && <div className="faint">Judge {probe.data.models.judge} · {probe.data.models.extractor.split("/").pop()}</div>}
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <ScopePicker />
          <div className="spacer" />
          <Seg
            label="Theme"
            value={theme}
            onChange={setTheme}
            options={[
              { value: "light", label: "", icon: <Sun aria-label="Light" /> },
              { value: "system", label: "", icon: <Monitor aria-label="System" /> },
              { value: "dark", label: "", icon: <Moon aria-label="Dark" /> },
            ]}
          />
        </header>
        {unauthorized ? <TokenGate /> : <Page />}
      </div>
      <FactDrawer />
    </div>
  );
}
