import { describe, it, expect } from "vitest";
import { readFileSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import Ajv2020 from "ajv/dist/2020.js";
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
    // ajv reports the path in slash notation (…/metrics/endpoint)
    expect(() => validateConfig({ ...base, models: m2 })).toThrow(/metrics\/endpoint/);
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

  it("rejects a model that sets both command and commands", () => {
    expect(() =>
      validateConfig({
        models: { m: { provider: "p", targetBaseUrl: "http://x", command: "a", commands: { start: "s", stop: "t" } } },
      }),
    ).toThrow(/mutually exclusive/);
  });

  it("parses command and commands lifecycle modes", () => {
    const cfg = validateConfig({
      ...base,
      models: {
        a: { provider: "p", targetBaseUrl: "http://x", command: "ollama serve" },
        b: { provider: "q", targetBaseUrl: "http://y", commands: { start: "nohup serve &", stop: "pkill -f serve" } },
      },
    });
    expect(cfg.models.a!.command).toBe("ollama serve");
    expect(cfg.models.a!.commands).toBeUndefined();
    expect(cfg.models.b!.commands).toEqual({ start: "nohup serve &", stop: "pkill -f serve" });
    expect(cfg.models.b!.command).toBeUndefined();
  });

  it("rejects non-string targetHeaders values", () => {
    expect(() =>
      validateConfig({
        ...base,
        models: { m: { provider: "p", targetBaseUrl: "http://x", targetHeaders: { "x": 5 } } as never },
      }),
    ).toThrow(/targetHeaders/);
  });

  it("rejects mixed lifecycle modes across aliases of the same provider", () => {
    expect(() =>
      validateConfig({
        ...base,
        models: {
          a: { provider: "p", targetBaseUrl: "http://x", command: "serve" },
          b: { provider: "p", targetBaseUrl: "http://x", commands: { start: "s", stop: "t" } },
        },
      }),
    ).toThrow(/different lifecycles/);
    // command vs no lifecycle
    expect(() =>
      validateConfig({
        ...base,
        models: {
          a: { provider: "p", targetBaseUrl: "http://x", command: "serve" },
          b: { provider: "p", targetBaseUrl: "http://x" },
        },
      }),
    ).toThrow(/different lifecycles/);
    // same mode is fine (incl. both without a lifecycle)
    expect(() =>
      validateConfig({
        ...base,
        models: {
          a: { provider: "p", targetBaseUrl: "http://x", command: "serve" },
          b: { provider: "p", targetBaseUrl: "http://x", command: "serve" },
        },
      }),
    ).not.toThrow();
    expect(() =>
      validateConfig({
        ...base,
        models: {
          a: { provider: "p", targetBaseUrl: "http://x" },
          b: { provider: "p", targetBaseUrl: "http://x" },
        },
      }),
    ).not.toThrow();
    // managed:false counts as "no local lifecycle" even with commands declared
    expect(() =>
      validateConfig({
        ...base,
        models: {
          a: { provider: "p", targetBaseUrl: "http://x", managed: false, commands: { start: "s", stop: "t" } },
          b: { provider: "p", targetBaseUrl: "http://x" },
        },
      }),
    ).not.toThrow();
  });
});

describe("${ENV} expansion across the whole config", () => {
  const base = {
    models: { m: { provider: "p", targetBaseUrl: "http://x" } },
  };

  it("expands references in any string value, not just api key/headers", () => {
    process.env.IMX_HOST = "10.0.0.5";
    process.env.IMX_PORT = "9123";
    const cfg = validateConfig({
      ...base,
      server: { host: "${IMX_HOST}", port: 1 },
      models: {
        m: { provider: "p", targetBaseUrl: "http://127.0.0.1:${IMX_PORT}", command: "ollama serve ${IMX_HOST}" },
      },
    });
    expect(cfg.server.host).toBe("10.0.0.5");
    // nested inside a larger string: plain text substitution
    expect(cfg.models.m!.targetBaseUrl).toBe("http://127.0.0.1:9123");
    expect(cfg.models.m!.command).toBe("ollama serve 10.0.0.5");
    delete process.env.IMX_HOST;
    delete process.env.IMX_PORT;
  });

  it("expands multiple references in one string and leaves plain strings untouched", () => {
    process.env.IMX_A = "a";
    process.env.IMX_B = "b";
    const cfg = validateConfig({
      ...base,
      models: { m: { provider: "p", targetBaseUrl: "http://x", commands: { start: "run ${IMX_A} and ${IMX_B}", stop: "stop" } } },
    });
    expect(cfg.models.m!.commands?.start).toBe("run a and b");
    // no references -> unchanged
    expect(cfg.models.m!.targetBaseUrl).toBe("http://x");
    delete process.env.IMX_A;
    delete process.env.IMX_B;
  });

  it("expands object keys (model aliases)", () => {
    process.env.IMX_ALIAS = "my-alias";
    const cfg = validateConfig({
      models: { "${IMX_ALIAS}": { provider: "p", targetBaseUrl: "http://x" } },
    });
    expect(cfg.models["my-alias"]!.provider).toBe("p");
    delete process.env.IMX_ALIAS;
  });

  it("casts ${VAR:int} to a real number that satisfies the schema", () => {
    process.env.IMX_PORT = "9123";
    const cfg = validateConfig({
      ...base,
      server: { port: "${IMX_PORT:int}" },
    });
    expect(cfg.server.port).toBe(9123);
    delete process.env.IMX_PORT;
  });

  it("casts ${VAR:float} to a number for fractional fields", () => {
    process.env.IMX_PCT = "0.25";
    const cfg = validateConfig({
      ...base,
      defaults: { vramThresholdPct: "${IMX_PCT:float}" },
    });
    expect(cfg.defaults.vramThresholdPct).toBe(0.25);
    delete process.env.IMX_PCT;
  });

  it("rejects non-integer values for :int and non-numeric values for :float", () => {
    process.env.IMX_BAD_INT = "12.5";
    expect(() =>
      validateConfig({ ...base, server: { port: "${IMX_BAD_INT:int}" } }),
    ).toThrow(/not an integer/);
    process.env.IMX_BAD_FLOAT = "abc";
    expect(() =>
      validateConfig({ ...base, defaults: { vramThresholdPct: "${IMX_BAD_FLOAT:float}" } }),
    ).toThrow(/not a number/);
    delete process.env.IMX_BAD_INT;
    delete process.env.IMX_BAD_FLOAT;
  });

  it("rejects an empty env var when a cast is requested", () => {
    process.env.IMX_EMPTY = "";
    expect(() =>
      validateConfig({ ...base, server: { port: "${IMX_EMPTY:int}" } }),
    ).toThrow(/empty/);
    delete process.env.IMX_EMPTY;
  });

  it("rejects unknown cast specifiers", () => {
    process.env.IMX_X = "1";
    expect(() =>
      validateConfig({ ...base, server: { port: "${IMX_X:bool}" } }),
    ).toThrow(/unknown type cast/);
    delete process.env.IMX_X;
  });

  it("rejects a cast nested inside a larger string", () => {
    process.env.IMX_MS = "500";
    expect(() =>
      validateConfig({ ...base, defaults: { requestTimeoutMs: "${IMX_MS:int}x" as never } }),
    ).toThrow(/only allowed when the whole value is a single variable reference/);
    delete process.env.IMX_MS;
  });

  it("fails when a referenced env var is unset, wherever it appears", () => {
    delete process.env.IMX_UNSET_ANYWHERE;
    expect(() =>
      validateConfig({ ...base, defaults: { requestTimeoutMs: "${IMX_UNSET_ANYWHERE:int}" } }),
    ).toThrow(/IMX_UNSET_ANYWHERE/);
  });

  it("fails on an unclosed ${ reference (typo detection)", () => {
    expect(() =>
      validateConfig({ ...base, models: { m: { provider: "p", targetBaseUrl: "http://x", command: "ollama serve ${IMX_HOST" } } }),
    ).toThrow(/unclosed/);
    // also when the whole value is a malformed reference
    expect(() =>
      validateConfig({ ...base, server: { port: "${IMX_PORT" as never } }),
    ).toThrow(/unclosed/);
  });

  it("leaves closed-but-non-reference forms (shell syntax) untouched", () => {
    const cfg = validateConfig({ ...base, models: { m: { provider: "p", targetBaseUrl: "http://x", command: "run ${SHELL_VAR:-default}" } } });
    expect(cfg.models.m!.command).toBe("run ${SHELL_VAR:-default}");
  });

  it("rejects keys that expand to the same name (collision)", () => {
    process.env.IMX_ALIAS = "m";
    try {
      expect(() =>
        validateConfig({ models: { m: { provider: "p", targetBaseUrl: "http://x" }, "${IMX_ALIAS}": { provider: "p", targetBaseUrl: "http://y" } } }),
      ).toThrow(/collides/);
    } finally {
      delete process.env.IMX_ALIAS;
    }
  });

  it("does not re-expand ${...} sequences inside env var values", () => {
    process.env.IMX_OUTER = "run ${IMX_INNER}";
    try {
      const cfg = validateConfig({ ...base, models: { m: { provider: "p", targetBaseUrl: "http://x", command: "${IMX_OUTER}" } } });
      expect(cfg.models.m!.command).toBe("run ${IMX_INNER}");
    } finally {
      delete process.env.IMX_OUTER;
    }
  });
});

describe("config.schema.json (direct ajv — anti-drift)", () => {
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  const schema = JSON.parse(readFileSync(new URL("../config.schema.json", import.meta.url), "utf8"));
  const validate = ajv.compile(schema);

  it("accepts the shipped example config", () => {
    const example = JSON.parse(
      stripJsonComments(readFileSync(new URL("../config.example.json", import.meta.url), "utf8")),
    );
    expect(validate(example)).toBe(true);
  });

  it("rejects unknown top-level keys", () => {
    expect(validate({ models: { m: { provider: "p", targetBaseUrl: "http://x" } }, bogus: 1 })).toBe(false);
    expect(validate.errors?.[0]?.keyword).toBe("additionalProperties");
  });

  it("rejects a model missing targetBaseUrl", () => {
    expect(validate({ models: { m: { provider: "p" } } })).toBe(false);
    expect(
      validate.errors?.some((e) => e.message === "must have required property 'targetBaseUrl'"),
    ).toBe(true);
  });

  it("rejects a non-URL targetBaseUrl and a missing / on healthEndpoint", () => {
    expect(validate({ models: { m: { provider: "p", targetBaseUrl: "ftp://x" } } })).toBe(false);
    expect(
      validate({ models: { m: { provider: "p", targetBaseUrl: "http://x", healthEndpoint: "h" } } }),
    ).toBe(false);
  });

  it("rejects a model that sets both command and commands", () => {
    expect(
      validate({
        models: { m: { provider: "p", targetBaseUrl: "http://x", command: "a", commands: { start: "s", stop: "t" } } },
      }),
    ).toBe(false);
  });

  it("rejects commands without a stop (and accepts command-only and commands-only models)", () => {
    expect(
      validate({ models: { m: { provider: "p", targetBaseUrl: "http://x", commands: { start: "s" } } } }),
    ).toBe(false);
    expect(validate({ models: { m: { provider: "p", targetBaseUrl: "http://x", command: "a" } } })).toBe(true);
    expect(
      validate({ models: { m: { provider: "p", targetBaseUrl: "http://x", commands: { start: "s", stop: "t" } } } }),
    ).toBe(true);
  });
});

describe("loadConfigFile", () => {
  it("loads the example config", () => {
    const cfg = loadConfigFile(new URL("../config.example.json", import.meta.url).pathname);
    expect(cfg.models["my-custom-llama"]!.provider).toBe("ollama");
    expect(cfg.models["my-vllm-model"]!.metrics?.endpoint).toBe("/metrics/vram");
  });

  it("expands env vars (incl. unquoted YAML casts) in a .yaml file", () => {
    const dir = mkdtempSync(tmpdir() + "/infermux-env-");
    const file = dir + "/cfg.yaml";
    writeFileSync(
      file,
      [
        "server:",
        "  port: ${IMX_YAML_PORT:int}",
        "models:",
        "  m:",
        "    provider: p",
        "    targetBaseUrl: http://127.0.0.1:${IMX_YAML_PORT}",
      ].join("\n"),
    );
    process.env.IMX_YAML_PORT = "7777";
    try {
      const cfg = loadConfigFile(file);
      expect(cfg.server.port).toBe(7777);
      expect(cfg.models.m!.targetBaseUrl).toBe("http://127.0.0.1:7777");
    } finally {
      delete process.env.IMX_YAML_PORT;
    }
  });
});
