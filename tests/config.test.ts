import { describe, it, expect } from "vitest";
import { validateConfig, stripJsonComments, loadConfigFile, ConfigError } from "../src/config.js";

describe("stripJsonComments", () => {
  it("removes line and block comments, preserves strings", () => {
    const src = `{
      // full line comment
      "a": "url with // and /* inside" /* block
        comment */
    }`;
    const parsed = JSON.parse(stripJsonComments(`{ // c\n "s": "a//b/*c", /* b */ "n": 2 }`));
    expect(parsed).toEqual({ s: "a//b/*c", n: 2 });
    expect(JSON.parse(stripJsonComments(src))).toEqual({ a: "url with // and /* inside" });
  });

  it("handles unterminated block comments and a trailing backslash without throwing", () => {
    // Unterminated block comment: everything to EOF is dropped (the resulting
    // JSON may then be invalid — that is JSON.parse's job, not the stripper's).
    expect(() => stripJsonComments(`{ "a": 1, /* oops`)).not.toThrow();
    // Escaped quote must not unbalance the string state: the `//` after it is
    // inside the string and must be preserved.
    expect(JSON.parse(stripJsonComments(`{ "a": "x\\" // not a comment" }`))).toEqual({
      a: 'x" // not a comment',
    });
    // Trailing backslash at EOF (malformed input) must not throw.
    expect(() => stripJsonComments(`{ "a": "x\\`).trimEnd()).not.toThrow();
  });
});

describe("validateConfig", () => {
  const base = {
    server: { host: "127.0.0.1", port: 9999, apiToken: "t" },
    defaults: { requestTimeoutMs: 1000 },
    models: {
      "m1": { provider: "ollama", targetBaseUrl: "http://127.0.0.1:11434", targetModel: "llama3" },
    },
  };

  it("fills defaults with built-in values", () => {
    const cfg = validateConfig(base);
    expect(cfg.server.port).toBe(9999);
    expect(cfg.server.apiToken).toBe("t");
    expect(cfg.defaults.requestTimeoutMs).toBe(1000);
    expect(cfg.defaults.healthCheckIntervalMs).toBe(500);
    expect(cfg.defaults.vramThresholdPct).toBe(0.15);
  });

  it("applies model-level overrides", () => {
    const cfg = validateConfig({
      ...base,
      models: {
        m: { provider: "vllm", targetBaseUrl: "http://127.0.0.1:8000", coolDownMs: 7, healthEndpoint: "/health", metrics: { endpoint: "/mx", vramUsedMiB: "a.b" } },
      },
    });
    expect(cfg.models.m!.coolDownMs).toBe(7);
    expect(cfg.models.m!.metrics?.vramUsedMiB).toBe("a.b");
  });

  it("rejects missing provider / bad baseUrl / no models", () => {
    expect(() =>
      validateConfig({ models: { m: { targetBaseUrl: "http://x" } } }),
    ).toThrow(ConfigError);
    expect(() =>
      validateConfig({ models: { m: { provider: "p", targetBaseUrl: "ftp://x" } } }),
    ).toThrow(/http/);
    expect(() => validateConfig({ server: {}, defaults: {} })).toThrow(/models/);
  });

  it("rejects out-of-range vram threshold", () => {
    expect(() => validateConfig({ ...base, defaults: { vramThresholdPct: 1.5 } })).toThrow(/vramThresholdPct/);
  });

  it("rejects zero/negative/non-numeric duration fields", () => {
    expect(() => validateConfig({ ...base, defaults: { requestTimeoutMs: 0 } })).toThrow(/requestTimeoutMs/);
    expect(() => validateConfig({ ...base, defaults: { healthCheckIntervalMs: -5 } })).toThrow(/healthCheckIntervalMs/);
    expect(() => validateConfig({ ...base, defaults: { vramCheckIntervalMs: "500" } })).toThrow(/vramCheckIntervalMs/);
    expect(() => validateConfig({ ...base, defaults: { coolDownMs: -1 } })).toThrow(/coolDownMs/);
    // zero cool-down is legitimate (skip the buffer)
    expect(validateConfig({ ...base, defaults: { coolDownMs: 0 } }).defaults.coolDownMs).toBe(0);
  });

  it("requires relative paths for healthEndpoint and metrics.endpoint", () => {
    const models = {
      m: { provider: "p", targetBaseUrl: "http://x", healthEndpoint: "health" },
    };
    expect(() => validateConfig({ ...base, models })).toThrow(/healthEndpoint/);
    const m2 = { m: { provider: "p", targetBaseUrl: "http://x", metrics: { endpoint: "metrics" } } };
    expect(() => validateConfig({ ...base, models: m2 })).toThrow(/metrics\.endpoint/);
  });

  it("fills vramBackend with 'auto' and accepts explicit backends", () => {
    expect(validateConfig(base).defaults.vramBackend).toBe("auto");
    expect(validateConfig({ ...base, defaults: { vramBackend: "none" } }).defaults.vramBackend).toBe("none");
    expect(validateConfig({ ...base, defaults: { vramBackend: "macos" } }).defaults.vramBackend).toBe("macos");
    expect(() => validateConfig({ ...base, defaults: { vramBackend: "vramd" } })).toThrow(/vramBackend/);
  });

  it("parses the model-level managed flag", () => {
    const cfg = validateConfig({
      ...base,
      models: { m: { provider: "p", targetBaseUrl: "http://x", managed: false } },
    });
    expect(cfg.models.m!.managed).toBe(false);
    expect(validateConfig(base).models.m1!.managed).toBeUndefined();
  });

  it("expands ${ENV} in targetApiKey and targetHeaders", () => {
    process.env.INFERMUX_TEST_KEY = "s3cret";
    const cfg = validateConfig({
      ...base,
      models: {
        m: {
          provider: "p",
          targetBaseUrl: "http://x",
          targetApiKey: "${INFERMUX_TEST_KEY}",
          targetHeaders: { "x-api-key": "prefix-${INFERMUX_TEST_KEY}", "x-static": "plain" },
        },
      },
    });
    expect(cfg.models.m!.targetApiKey).toBe("s3cret");
    expect(cfg.models.m!.targetHeaders!["x-api-key"]).toBe("prefix-s3cret");
    expect(cfg.models.m!.targetHeaders!["x-static"]).toBe("plain");
    delete process.env.INFERMUX_TEST_KEY;
  });

  it("fails to load when a referenced env var is unset", () => {
    delete process.env.INFERMUX_UNSET_KEY_XYZ;
    expect(() =>
      validateConfig({
        ...base,
        models: { m: { provider: "p", targetBaseUrl: "http://x", targetApiKey: "${INFERMUX_UNSET_KEY_XYZ}" } },
      }),
    ).toThrow(/INFERMUX_UNSET_KEY_XYZ/);
  });

  it("rejects non-string targetHeaders values", () => {
    expect(() =>
      validateConfig({
        ...base,
        models: { m: { provider: "p", targetBaseUrl: "http://x", targetHeaders: { "x": 5 } } as never },
      }),
    ).toThrow(/targetHeaders/);
  });
});

describe("loadConfigFile", () => {
  it("loads the example config", () => {
    const cfg = loadConfigFile(new URL("../config.example.json", import.meta.url).pathname);
    expect(cfg.models["my-custom-llama"]!.provider).toBe("ollama");
    expect(cfg.models["my-vllm-model"]!.metrics?.endpoint).toBe("/metrics/vram");
  });
});
