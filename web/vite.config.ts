import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `pnpm dev` proxies API calls to the memx API on :8000 (set MEMX_API to point it at another one).
// Build: output goes to memx_api/static, which the API serves at /.
export default defineConfig(({ mode }) => {
  const target = loadEnv(mode, ".", "MEMX_").MEMX_API ?? "http://localhost:8000";
  return {
    plugins: [react()],
    server: { port: 5173, proxy: { "/v1": target, "/readyz": target } },
    build: { outDir: "../memx_api/static", emptyOutDir: true },
  };
});
