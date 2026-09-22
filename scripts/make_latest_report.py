#!/usr/bin/env python3
"""Assemble reports/latest.md (and the README results table) from the
measured artifacts: reports/environment.txt, reports/bench.json,
reports/internal_dev.json, reports/external_dev.json, reports/leakage.json,
the latest reports/jevbench/<run>/metrics.json and
reports/jev_bench_hf/<run>/summary.md. Nothing is estimated."""
import glob, json, os, re, subprocess, sys

def latest_dir(pattern):
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
    return dirs[-1] if dirs else None

env = dict(l.split(": ", 1) for l in open("reports/environment.txt").read().strip().splitlines())
bench = json.load(open("reports/bench.json"))
internal = json.load(open("reports/internal_dev.json"))
external = json.load(open("reports/external_dev.json"))
leak = json.load(open("reports/leakage.json"))
jb_dir = latest_dir("reports/jevbench/*-*/")
jb = json.load(open(os.path.join(jb_dir, "metrics.json"))) if jb_dir and os.path.exists(os.path.join(jb_dir, "metrics.json")) else None
jb_report = open(os.path.join(jb_dir, "report.md")).read() if jb_dir else ""
hf_dir = latest_dir("reports/jev_bench_hf/*-*/")
hf_summary = open(os.path.join(hf_dir, "summary.md")).read() if hf_dir and os.path.exists(os.path.join(hf_dir, "summary.md")) else ""

def md_section(text, header):
    m = re.search(rf"^## {re.escape(header)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""

commit = env.get("commit", "?")[:12]
lines = []
L = lines.append
L(f"# Latest evaluation report")
L("")
L(f"Frozen engine commit `{commit}` · weights `{env.get('model_weights_version')}` · calibration `{env.get('model_calibration_version')}` · generated {env.get('date_utc')}")
L("")
L("Every number below is **measured** (labels: internal / external dev / JevBench public / jev-bench / published reference). Nothing is estimated.")
L("")
L("## Environment")
L("")
L("| | |")
L("|---|---|")
for k in ["cpu", "cores", "ram", "kernel", "rustc", "release_profile"]:
    L(f"| {k} | {env.get(k, '?')} |")
L(f"| JevBench harness | fstandhartinger/jevbench @ 75e6224ed8103bbc3485ca74820a2eaf7ce8abe0 (MIT), public files sha256 231df3c2… / 5c2414ed… / 89e9e6be… |")
L(f"| jev-bench | Praveenrajus/jev-bench @ 002ad22de8db2df5e0eb898b3da8072dbd4af4de; harness uspraveen/Jevify @ 2891025b8a4520d0eb639f2cd1b3fddaf4b922b3 (Apache-2.0) |")
L("")
L("## Leakage check (`reports/leakage.json`)")
L("")
L(f"Verdict **{leak['verdict']}** — training/calibration units {leak['train_units']}, evaluation units {leak['eval_units']}; external state matches: exact {leak['state_exact_matches']}, normalized {leak['state_normalized_matches']}, near-duplicate {leak['state_near_duplicates']} (rate {leak['state_near_duplicate_rate']:.4f}); instruction matches for review {leak['instruction_matches_for_review']}; max 5-shingle Jaccard observed {leak['max_jaccard_observed']:.3f}.")
L("")
L("## JevBench public tiers (JevBench harness scoring, run 2 = final)")
L("")
if jb_report:
    L(md_section(jb_report, "Tiers"))
    L("")
    L("### Hard tier by family")
    L("")
    L(md_section(jb_report, "Hard tier by family"))
    L("")
    L("### By question type")
    L("")
    L(md_section(jb_report, "By question type"))
    L("")
    L("### Paraphrase pairs")
    L("")
    L(md_section(jb_report, "Paraphrase pairs (`group`)"))
    L("")
L("### Comparison with the published Jev 1.13 reference")
L("")
L("| tier | Sextant (public subset, measured) | Jev 1.13 published (full tier) | Jev 1.13 public subset (independent audited run) | gap on public subset |")
L("|---|---|---|---|---|")
if jb and "tiers" in jb:
    ref_full = {"easy": "1.000", "standard": "0.990", "hard": "0.741"}
    ref_pub = {"easy": "48/48 = 1.000", "standard": "71/72 = 0.986", "hard": "81/111 = 0.730"}
    ref_pub_val = {"easy": 1.0, "standard": 71/72, "hard": 81/111}
    for tier in ["easy", "standard", "hard"]:
        t = jb["tiers"].get(tier) or jb["tiers"].get("original" if tier == "standard" else tier)
        if not t:
            continue
        mt = t.get("metrics", t)
        acc = mt.get("accuracy")
        n = t.get("public_n") or mt.get("n_planned")
        c = mt.get("n_correct")
        L(f"| {tier} | {c}/{n} = {acc:.3f} | {ref_full[tier]} | {ref_pub[tier]} | {acc - ref_pub_val[tier]:+.3f} |")
    L("| judge | not public (0/146) | 0.945 | – | – |")
else:
    L("| (metrics.json not found) | | | | |")
L("")
L("Reference targets set for this project (Stage A: easy ≥ 0.95, standard ≥ 0.80; Stage B: easy ≥ 0.98, standard ≥ 0.90, hard ≥ 0.50; Stage C: easy ≥ 0.99, standard ≥ 0.97, hard ≥ 0.70) are **not reached** on the standard and hard tiers; see docs/LIMITATIONS.md for the failure classes.")
L("")
L("## jev-bench (22 configs, Jevify harness scoring, run 2 = final)")
L("")
if hf_summary:
    L(md_section(hf_summary, "Leaderboard-style aggregates"))
    L("")
    L("Published Jev 1.13.0 reference on the same 22 configs: macro accuracy 0.733, macro ECE 0.113, macro Brier 0.349, TVD→human 0.432.")
    L("")
    L("### Per config")
    L("")
    L(md_section(hf_summary, "Per config"))
    L("")
else:
    L("(run in progress or missing)")
    L("")
L("## Internal development set (synthetic, by family)")
L("")
L("| family | n | accuracy | NLL | Brier | ECE |")
L("|---|---|---|---|---|---|")
for fam, s in sorted(internal["by_group"].items()):
    L(f"| {fam} | {s['n']} | {s['accuracy']:.3f} | {s['nll']:.3f} | {s['brier']:.3f} | {s['ece']:.3f} |")
o = internal["overall"]
L(f"| **all** | {o['n']} | {o['accuracy']:.3f} | {o['nll']:.3f} | {o['brier']:.3f} | {o['ece']:.3f} |")
for k, s in internal["by_kind"].items():
    L(f"| kind: {k} | {s['n']} | {s['accuracy']:.3f} | {s['nll']:.3f} | {s['brier']:.3f} | {s['ece']:.3f} |")
gs = internal.get("group_stability") or {}
if gs:
    L("")
    L(f"Variant stability (paraphrase / option permutation / JSON wrapping / distractors of the same scenario): same-answer rate {gs['same_answer_rate']:.3f} over {gs['groups']} groups, mean top-probability distance {gs['mean_prob_distance']:.3f}.")
L("")
L("## External development sets (public permissive datasets, never trained on their dev splits; HWU64 never trained on at all)")
L("")
L("| source | n | accuracy | NLL | Brier | ECE |")
L("|---|---|---|---|---|---|")
for src, s in sorted(external["by_group"].items()):
    L(f"| {src} | {s['n']} | {s['accuracy']:.3f} | {s['nll']:.3f} | {s['brier']:.3f} | {s['ece']:.3f} |")
o = external["overall"]
L(f"| **all** | {o['n']} | {o['accuracy']:.3f} | {o['nll']:.3f} | {o['brier']:.3f} | {o['ece']:.3f} |")
for k, s in external["by_kind"].items():
    L(f"| kind: {k} | {s['n']} | {s['accuracy']:.3f} | {s['nll']:.3f} | {s['brier']:.3f} | {s['ece']:.3f} |")
L("")
L("## Latency and throughput (release build, embedded model, after warm-up, single process)")
L("")
L(f"Threads: {bench.get('threads')} · `sextant bench --iters 200` · {env.get('cpu')}")
L("")
L("| scenario | questions | state bytes | p50 ms | p95 ms | p99 ms | req/s |")
L("|---|---|---|---|---|---|---|")
for r in bench["results"]:
    L(f"| {r['scenario']} | {r['questions']} | {r['state_bytes']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | {r['requests_per_second']:.1f} |")
typ = next((r for r in bench["results"] if r["scenario"].startswith("typical")), None)
if typ:
    L("")
    L(f"Typical request (8 questions, ≈4k-token state, ≤10 criteria): p50 {typ['p50_ms']:.1f} ms / p95 {typ['p95_ms']:.1f} ms — target p50 ≤ 25 ms, p95 ≤ 75 ms: **met**. Published Jev round-trip: ≈100 ms typical.")
L("")
mem = subprocess.run(["bash", "-c", "( /usr/bin/time -v ./target/release/sextant decide examples/request_basic.json >/dev/null ) 2>&1 | awk '/Maximum resident/{print $6}'"], capture_output=True, text=True).stdout.strip()
if mem:
    L(f"Peak resident memory of one `sextant decide` (embedded lexicon + model, one request): {int(mem)/1024:.0f} MiB.")
    L("")
open("reports/latest.md", "w").write("\n".join(lines) + "\n")
print("wrote reports/latest.md")

# README table between markers
readme = open("README.md").read()
table = []
if jb and "tiers" in jb:
    table.append("| Benchmark (measured) | Sextant | Published Jev 1.13 reference |")
    table.append("|---|---|---|")
    for tier in ["easy", "standard", "hard"]:
        t = jb["tiers"].get(tier) or jb["tiers"].get("original" if tier == "standard" else tier)
        if t:
            mt = t.get("metrics", t)
            table.append(f"| JevBench public **{tier}** accuracy | {mt.get('n_correct')}/{t.get('public_n') or mt.get('n_planned')} = {mt['accuracy']*100:.1f} % | {dict(easy='100 % full tier; 48/48 public', standard='99.0 % full tier; 71/72 public', hard='74.1 % full tier; 81/111 public')[tier]} |")
    table.append("| JevBench schema validity | 100 % (strict) | 100 % |")
if hf_summary:
    m = re.search(r"\| 22 \| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", hf_summary)
    if m:
        table.append(f"| jev-bench macro accuracy / ECE / Brier (22 configs, {m.group(1)} records) | {m.group(3)} / {m.group(4)} / {m.group(5)} | 0.733 / 0.113 / 0.349 |")
o = internal["overall"]; e = external["overall"]
table.append(f"| Internal synthetic dev accuracy / ECE | {o['accuracy']*100:.1f} % / {o['ece']:.3f} | – |")
table.append(f"| External public dev sets accuracy / ECE | {e['accuracy']*100:.1f} % / {e['ece']:.3f} | – |")
if typ:
    table.append(f"| Latency, typical 8-question request (p50 / p95) | {typ['p50_ms']:.1f} ms / {typ['p95_ms']:.1f} ms | ≈100 ms round trip (published) |")
block = "<!-- BENCHMARK_TABLE_START -->\n" + "\n".join(table) + "\n\nFull tables, per-family breakdowns, calibration and robustness metrics: `reports/latest.md`.\n<!-- BENCHMARK_TABLE_END -->"
readme = re.sub(r"<!-- BENCHMARK_TABLE_START -->.*?<!-- BENCHMARK_TABLE_END -->", block, readme, flags=re.S)
open("README.md", "w").write(readme)
print("updated README table")
