import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `pnpm dev` proxies API calls to the memx API on :8000.
// Build: output goes to memx_api/static, which the API serves at /.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/v1": "http://localhost:8000", "/readyz": "http://localhost:8000" } },
  build: { outDir: "../memx_api/static", emptyOutDir: true, chunkSizeWarningLimit: 1200 },
});
