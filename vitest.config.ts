import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    coverage: {
      provider: "v8",
      // Report on library code only: scripts/smoke.ts is a manual harness,
      // types.ts is type-only (erased at runtime).
      include: ["src/**"],
      exclude: ["src/types.ts"],
    },
  },
});
