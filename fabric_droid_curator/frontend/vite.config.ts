import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget =
  (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.CURATOR_API_URL ||
  "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    watch: {
      usePolling: true,
      interval: 500
    },
    proxy: {
      "/api": apiTarget
    }
  },
  preview: {
    port: 5173,
    proxy: {
      "/api": apiTarget
    }
  }
});
