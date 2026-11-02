import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type Route =
  | { page: "overview" }
  | { page: "memory" }
  | { page: "graph"; focus?: string }
  | { page: "add" }
  | { page: "ask" }
  | { page: "activity"; tab?: string }
  | { page: "ingest"; id: string }
  | { page: "recall"; id: string }
  | { page: "settings" };

export function parseHash(hash: string): Route {
  const [path, query] = hash.replace(/^#\/?/, "").split("?");
  const [page, id] = path.split("/");
  const params = new URLSearchParams(query ?? "");
  switch (page) {
    case "memory": return { page: "memory" };
    case "graph": return { page: "graph", focus: params.get("focus") ?? undefined };
    case "add": return { page: "add" };
    case "ask": return { page: "ask" };
    case "activity": return { page: "activity", tab: params.get("tab") ?? undefined };
    case "ingests": return id ? { page: "ingest", id } : { page: "activity", tab: "ingests" };
    case "recalls": return id ? { page: "recall", id } : { page: "activity", tab: "recalls" };
    case "settings": return { page: "settings" };
    default: return { page: "overview" };
  }
}

export function routeHash(r: Route): string {
  switch (r.page) {
    case "graph": return `#/graph${r.focus ? `?focus=${r.focus}` : ""}`;
    case "activity": return `#/activity${r.tab ? `?tab=${r.tab}` : ""}`;
    case "ingest": return `#/ingests/${r.id}`;
    case "recall": return `#/recalls/${r.id}`;
    default: return `#/${r.page}`;
  }
}

const read = (k: string, fallback: string) => {
  try { return localStorage.getItem(k) ?? fallback; } catch { return fallback; }
};
const write = (k: string, v: string) => {
  try { localStorage.setItem(k, v); } catch { /* storage unavailable */ }
};

export type Theme = "system" | "light" | "dark";

interface AppState {
  route: Route;
  go: (r: Route) => void;
  scope: string;
  setScope: (s: string) => void;
  factId: string | null;
  openFact: (id: string | null) => void;
  theme: Theme;
  setTheme: (t: Theme) => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  const [scope, setScopeState] = useState(() => read("memx.scope", "memory"));
  const [factId, openFact] = useState<string | null>(null);
  const [theme, setThemeState] = useState<Theme>(() => read("memx.theme", "system") as Theme);

  // A link may carry ?scope=… so it opens on the right memory.
  useEffect(() => {
    const onHash = () => {
      setRoute(parseHash(window.location.hash));
      const linked = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("scope");
      if (linked) { setScopeState(linked); write("memx.scope", linked); }
    };
    onHash();
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      const dark = theme === "dark" || (theme === "system" && mq.matches);
      document.documentElement.dataset.theme = dark ? "dark" : "light";
    };
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [theme]);

  const go = useCallback((r: Route) => {
    const h = routeHash(r);
    if (window.location.hash !== h) window.location.hash = h;
    else setRoute(r);
    window.scrollTo({ top: 0 });
  }, []);
  const setScope = useCallback((s: string) => { setScopeState(s); write("memx.scope", s); }, []);
  const setTheme = useCallback((t: Theme) => { setThemeState(t); write("memx.theme", t); }, []);

  const value = useMemo(
    () => ({ route, go, scope, setScope, factId, openFact, theme, setTheme }),
    [route, go, scope, setScope, factId, theme, setTheme],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside AppProvider");
  return v;
}
