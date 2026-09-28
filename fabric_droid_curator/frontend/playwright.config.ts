import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  use: {
    baseURL: process.env.CURATOR_UI_URL || "http://127.0.0.1:5173",
    headless: true,
    trace: "retain-on-failure",
    launchOptions: {
      executablePath: process.env.CHROME_BIN || "/usr/bin/google-chrome"
    }
  }
});
