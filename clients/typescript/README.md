# Sextant TypeScript client

`sextant.ts` is a single-file, dependency-free client for the Sextant HTTP API. It uses
the global `fetch` (Node 18+, Deno, Bun, browsers) and ships typed request / response
interfaces that mirror `docs/openapi.yaml`. Copy the file into your project.

```ts
import { SextantClient, SextantError, choice, score, noul } from "./sextant";

const client = new SextantClient("http://127.0.0.1:8080", { timeoutMs: 30_000 });

const reply = await client.systemone(
  "Help! My payouts have been failing for 3 days.",
  {
    department: choice("Which team should handle this?", {
      billing: "Payments, invoicing, refunds, payouts",
      technical: "Bugs, outages, integrations",
      sales: "Pricing, upgrades, new accounts",
    }),
    is_urgent: noul("Does this convey urgency?"),
    frustration: score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
  },
  { model: "sextant-1", explain: false },
);

reply.answers.department.choice;          // "billing"  – typed as ChoiceAnswer
reply.answers.department.probabilities;   // { billing: 0.94, technical: 0.03, sales: 0.03 }
reply.answers.is_urgent.noul;             // 0.38      – typed as NoulAnswer, P(true)
reply.answers.frustration.score;          // 1.38      – typed as ScoreAnswer, Σ index × P(level)
reply.usage;                              // { input_tokens: 56, output_tokens: 12 }
```

The `answers` map is typed per question id: `choice(...)` questions yield `ChoiceAnswer`,
`score(...)` yields `ScoreAnswer` and `noul(...)` yields `NoulAnswer`, so the fields above
type-check without casts.

## Errors

Any non-2xx response rejects with `SextantError`; `status`, `code`, `field` and `body`
mirror the server's `{"error": {...}}` body:

```ts
try {
  await client.systemone("hi", { q: choice("?", { only: "one option" }) });
} catch (e) {
  if (e instanceof SextantError) console.log(e.status, e.code, e.field); // 422 invalid_request questions.q.criteria
}
```

Network failures reject with `status === 0` and `code === "connection_error"`; a timeout
(`timeoutMs`, default 30 s) rejects with `code === "timeout"`.

## Other endpoints

```ts
await client.health();   // { status: "ok", version: "0.1.0", model: "sextant-1", neural: false, network_required: false, ... }
await client.models();   // { object: "list", models: [{ name: "sextant-1", aliases: [...], ... }] }
```

Type-check the file on its own with `tsc --noEmit --strict --target es2020 --lib es2020,dom sextant.ts`.
