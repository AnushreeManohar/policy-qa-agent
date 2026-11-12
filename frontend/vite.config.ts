import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the API runs on :8000; Vite proxies /api so the browser sees one origin (no CORS needed).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": process.env.VITE_API_PROXY ?? "http://localhost:8000" },
  },
  build: { outDir: "dist", sourcemap: false },
});
