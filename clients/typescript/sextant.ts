/**
 * Thin TypeScript client for the Sextant decision engine (System One compatible API).
 *
 * Zero dependencies; uses the global `fetch` (Node 18+, Deno, Bun, browsers). The
 * request / response interfaces mirror docs/openapi.yaml.
 *
 *   import { SextantClient, SextantError, choice, score, noul } from "./sextant";
 *
 *   const client = new SextantClient("http://127.0.0.1:8080");
 *   const reply = await client.systemone("Help! My payouts have been failing for 3 days.", {
 *     department: choice("Which team should handle this?", {
 *       billing: "Payments, invoicing, refunds, payouts",
 *       technical: "Bugs, outages, integrations",
 *       sales: "Pricing, upgrades, new accounts",
 *     }),
 *     is_urgent: noul("Does this convey urgency?"),
 *     frustration: score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
 *   });
 *   reply.answers.department.choice;   // "billing"      (typed as ChoiceAnswer)
 *   reply.answers.is_urgent.noul;      // P(true), 0..1  (typed as NoulAnswer)
 *   reply.answers.frustration.score;   // 0..2           (typed as ScoreAnswer)
 */

// ----------------------------------------------------------------------------- wire types

/** string | object | array | null */
export type Entry = string | Record<string, unknown> | unknown[] | null;

/** The evidence: a string, an object or an array. */
export type State = string | Record<string, unknown> | unknown[];

export interface NoulQuestion {
  type: "noul";
  instructions?: Entry;
  criteria?: { true?: Entry; false?: Entry };
}

export interface ChoiceQuestion {
  type: "choice";
  instructions?: Entry;
  /** option key -> description (2..255 options) */
  criteria: Record<string, Entry>;
}

export interface ScoreQuestion {
  type: "score";
  instructions?: Entry;
  /** ordered level descriptions (2..10 levels), index 0 = lowest */
  criteria: Entry[];
}

export type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion;

export interface SystemOneRequest<Q extends Record<string, Question> = Record<string, Question>> {
  /** `sextant-1` (aliases: `sextant-latest`, `sextant`) */
  model: string;
  state: State;
  /** caller-chosen ids -> questions; ids never influence inference */
  questions: Q;
  /** attach an `explain` block to every answer (never alters inference) */
  explain?: boolean;
}

export interface Evidence {
  path?: string;
  text?: string;
  score?: number;
}

/** Optional inspection block (only when explain=true). */
export interface Explanation {
  family?: string;
  /** `symbolic:<resolver>` or `semantic` */
  path?: string;
  top_evidence?: Evidence[];
  raw_scores?: Record<string, number>;
  features?: Record<string, Record<string, number>>;
  calibration?: Record<string, unknown>;
  notes?: string[];
}

export interface NoulAnswer {
  type: "noul";
  /** calibrated P(true), 0..1 */
  noul: number;
  explain?: Explanation;
}

export interface ChoiceAnswer {
  type: "choice";
  /** argmax option (lexicographic tie-break) */
  choice: string;
  /** exactly the supplied keys; sums to 1 */
  probabilities: Record<string, number>;
  confidence: number;
  explain?: Explanation;
}

export interface ScoreAnswer {
  type: "score";
  /** Σ level_index × P(level) */
  score: number;
  legend: Record<string, string>;
  /** keyed by level index as a string: "0".."K-1" */
  probabilities: Record<string, number>;
  confidence: number;
  explain?: Explanation;
}

export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

/** The answer type that a given question type produces. */
export type AnswerFor<Q extends Question> = Q extends ChoiceQuestion
  ? ChoiceAnswer
  : Q extends ScoreQuestion
    ? ScoreAnswer
    : Q extends NoulQuestion
      ? NoulAnswer
      : Answer;

export interface Usage {
  input_tokens?: number;
  output_tokens?: number;
}

export interface SystemOneResponse<Q extends Record<string, Question> = Record<string, Question>> {
  model: string;
  answers: { [K in keyof Q]: AnswerFor<Q[K]> };
  usage: Usage;
}

export interface ErrorBody {
  status?: number;
  code?: string;
  message?: string;
  field?: string;
}

export interface ErrorResponse {
  error: ErrorBody;
}

export interface ModelCard {
  name?: string;
  id?: string;
  object?: string;
  description?: string;
  release_date?: string;
  aliases?: string[];
  max_choice_options?: number;
  max_score_levels?: number;
  neural?: boolean;
}

export interface ModelList {
  object?: string;
  models: ModelCard[];
}

export interface Health {
  status: string;
  version: string;
  model: string;
  uptime_seconds: number;
  neural: boolean;
  network_required: boolean;
}

// ----------------------------------------------------------------------------- helpers

/** A Choice question: pick one of 2..255 options (key -> description). */
export function choice(instructions: Entry, criteria: Record<string, Entry>): ChoiceQuestion {
  return { type: "choice", instructions, criteria };
}

/** A Score question: place the state on 2..10 ordered levels (index 0 = lowest). */
export function score(instructions: Entry, criteria: Entry[]): ScoreQuestion {
  return { type: "score", instructions, criteria };
}

/** A Noul question: calibrated P(true). `criteria` may describe the true / false sides. */
export function noul(instructions: Entry, criteria?: { true?: Entry; false?: Entry }): NoulQuestion {
  return criteria ? { type: "noul", instructions, criteria } : { type: "noul", instructions };
}

// ----------------------------------------------------------------------------- client

export const DEFAULT_MODEL = "sextant-1";
export const DEFAULT_BASE_URL = "http://127.0.0.1:8080";

/** A failed request: `status` is the HTTP status (0 when no response arrived) and
 * `code` / `field` / `body` mirror the server's `{"error": {...}}` body. */
export class SextantError extends Error {
  readonly status: number;
  readonly code: string;
  readonly field?: string;
  readonly body?: unknown;

  constructor(status: number, code: string, message: string, field?: string, body?: unknown) {
    super(status ? `HTTP ${status} ${code}: ${message}${field ? ` (field: ${field})` : ""}` : `${code}: ${message}`);
    this.name = "SextantError";
    this.status = status;
    this.code = code;
    this.field = field;
    this.body = body;
  }

  static fromResponse(status: number, text: string): SextantError {
    let body: unknown = text;
    try {
      body = text ? JSON.parse(text) : undefined;
    } catch {
      /* not JSON: keep the raw text */
    }
    const err = (body as ErrorResponse | undefined)?.error;
    if (err && typeof err === "object") {
      return new SextantError(status, err.code ?? "error", err.message ?? "request failed", err.field, body);
    }
    return new SextantError(status, "http_error", text.slice(0, 500) || "request failed", undefined, body);
  }
}

export interface SextantClientOptions {
  /** per-request timeout in milliseconds (default 30000; 0 disables) */
  timeoutMs?: number;
  /** extra headers sent with every request */
  headers?: Record<string, string>;
  /** custom fetch implementation (defaults to the global one) */
  fetch?: typeof fetch;
}

export interface SystemOneOptions {
  /** default `sextant-1` */
  model?: string;
  /** attach explanations to every answer */
  explain?: boolean;
  signal?: AbortSignal;
}

export class SextantClient {
  readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly headers: Record<string, string>;
  private readonly fetchImpl: typeof fetch;

  constructor(baseUrl: string = DEFAULT_BASE_URL, options: SextantClientOptions = {}) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.headers = { Accept: "application/json", ...(options.headers ?? {}) };
    const impl = options.fetch ?? globalThis.fetch;
    if (typeof impl !== "function") {
      throw new Error("no fetch implementation available; pass one via options.fetch");
    }
    this.fetchImpl = impl;
  }

  /** GET /health */
  health(): Promise<Health> {
    return this.request<Health>("GET", "/health");
  }

  /** GET /v1/models */
  models(): Promise<ModelList> {
    return this.request<ModelList>("GET", "/v1/models");
  }

  /**
   * POST /v1/systemone. Evaluate typed questions against a state. The result's
   * `answers` is typed per question id (choice -> ChoiceAnswer, ...).
   */
  systemone<Q extends Record<string, Question>>(
    state: State,
    questions: Q,
    options: SystemOneOptions = {},
  ): Promise<SystemOneResponse<Q>> {
    const body: SystemOneRequest<Q> = { model: options.model ?? DEFAULT_MODEL, state, questions };
    if (options.explain) body.explain = true;
    return this.request<SystemOneResponse<Q>>("POST", "/v1/systemone", body, options.signal);
  }

  private async request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
    const headers: Record<string, string> = { ...this.headers };
    let payload: string | undefined;
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      payload = JSON.stringify(body);
    }
    const controller = this.timeoutMs > 0 && typeof AbortController === "function" ? new AbortController() : undefined;
    const timer = controller ? setTimeout(() => controller.abort(), this.timeoutMs) : undefined;
    if (controller && signal) {
      signal.addEventListener("abort", () => controller.abort(), { once: true });
    }
    let response: Response;
    try {
      response = await this.fetchImpl(this.baseUrl + path, {
        method,
        headers,
        body: payload,
        signal: controller?.signal ?? signal,
      });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      throw new SextantError(0, controller?.signal.aborted ? "timeout" : "connection_error", `${this.baseUrl}${path}: ${message}`);
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
    const text = await response.text();
    if (!response.ok) {
      throw SextantError.fromResponse(response.status, text);
    }
    try {
      return JSON.parse(text) as T;
    } catch (e) {
      throw new SextantError(0, "malformed_response", `non-JSON body from ${path}: ${(e as Error).message}`, undefined, text.slice(0, 500));
    }
  }
}
