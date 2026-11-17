import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@xyflow/react/dist/style.css";
import "./styles.css";
import App from "./App";
import { AppProvider } from "./lib/app";
import { ErrorBoundary, TipProvider, ToastProvider } from "./components/ui";

const client = new QueryClient({
  defaultOptions: { queries: { staleTime: 5_000, refetchOnWindowFocus: false, retry: 1 } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <AppProvider>
        <ToastProvider>
          <TipProvider>
            <ErrorBoundary>
              <App />
            </ErrorBoundary>
          </TipProvider>
        </ToastProvider>
      </AppProvider>
    </QueryClientProvider>
  </StrictMode>,
);
