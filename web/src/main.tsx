import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "@fontsource/inter-tight/400.css";
import "@fontsource/inter-tight/500.css";
import "@fontsource/inter-tight/600.css";
import "@fontsource/inter-tight/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "@xyflow/react/dist/style.css";
import "./styles.css";
import App from "./App";
import { AppProvider } from "./lib/app";
import { TipProvider, ToastProvider } from "./components/ui";

const client = new QueryClient({
  defaultOptions: { queries: { staleTime: 5_000, refetchOnWindowFocus: false, retry: 1 } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <AppProvider>
        <ToastProvider>
          <TipProvider>
            <App />
          </TipProvider>
        </ToastProvider>
      </AppProvider>
    </QueryClientProvider>
  </StrictMode>,
);
