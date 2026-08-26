import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build output goes to app/frontend/build, which the Flask backend serves.
// In dev, `npm run dev` starts Vite and proxies /api to the Flask server.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "build",
    emptyOutDir: true,
  },
  server: {
    port: 5199,
    proxy: {
      "/api": "http://127.0.0.1:8777",
    },
  },
});
