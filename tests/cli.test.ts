import { describe, expect, it } from "vitest";
import { parseArgs } from "../src/cli.js";

// Only the non-terminating paths are exercised here: --help and unknown
// flags call process.exit() and would kill the test runner.
describe("parseArgs", () => {
  it("parses long flags", () => {
    expect(parseArgs(["--config", "a.json", "--log-level", "debug"])).toEqual({
      configPath: "a.json",
      logLevel: "debug",
    });
  });

  it("parses the -c short flag", () => {
    expect(parseArgs(["-c", "b.yaml"])).toEqual({ configPath: "b.yaml" });
  });

  it("returns an empty result with no arguments", () => {
    expect(parseArgs([])).toEqual({});
  });
});
