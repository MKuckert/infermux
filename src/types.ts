import type { DefaultsConfig, ModelConfig, ServerConfig } from "./config.js";

export type { ModelConfig, ServerConfig, DefaultsConfig, InfermuxConfig } from "./config.js";

/** Defaults with every field guaranteed present (filled in at load time). */
export type RequiredDefaults = Required<DefaultsConfig>;

/** Fully resolved runtime configuration. */
export interface RuntimeConfig {
  server: ServerConfig;
  defaults: RequiredDefaults;
  models: Record<string, ModelConfig>;
}

/* ------------------------------------------------------------------ */
/* Minimal OpenAI API request/response types (enough for proxying)     */
/* ------------------------------------------------------------------ */

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool" | (string & {});
  content: string | null;
  name?: string;
  tool_calls?: unknown[];
  tool_call_id?: string;
  [key: string]: unknown;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  max_completion_tokens?: number;
  n?: number;
  stop?: string | string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  seed?: number;
  user?: string;
  response_format?: unknown;
  tools?: unknown[];
  tool_choice?: unknown;
  logprobs?: boolean | null;
  top_logprobs?: number | null;
  logit_bias?: Record<string, number> | null;
  [key: string]: unknown;
}

export interface CompletionRequest {
  model: string;
  prompt: string | string[] | number[] | number[][];
  suffix?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  n?: number;
  stream?: boolean;
  logprobs?: number | null;
  echo?: boolean;
  stop?: string | string[] | null;
  presence_penalty?: number;
  frequency_penalty?: number;
  user?: string;
  [key: string]: unknown;
}
