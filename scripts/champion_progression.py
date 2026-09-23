#!/usr/bin/env python3
"""Write the champion progression and the experiment ledger as markdown.

Reads experiments/index.jsonl and experiments/CURRENT_CHAMPION.json and
writes reports/champions.md. Every figure comes from a recorded run; a
metric that was never measured is left blank rather than filled in.

    python3 scripts/champion_progression.py [--out reports/champions.md]
"""
import argparse
import datetime
import json
from pathlib import Path

INDEX = Path("experiments/index.jsonl")
CHAMP = Path("experiments/CURRENT_CHAMPION.json")


def load():
    if not INDEX.exists():
        raise SystemExit(f"{INDEX} missing; nothing to report")
    return [json.loads(l) for l in INDEX.read_text().splitlines() if l.strip()]


def cell(v, digits=4):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def metric(rec, *names):
    m = rec.get("dev_metrics") or {}
    for n in names:
        if n in m:
            return m[n]
        comp = m.get("components") or {}
        if n in comp:
            return comp[n]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/champions.md")
    a = ap.parse_args()
    recs = load()
    champ = json.loads(CHAMP.read_text()) if CHAMP.exists() else {}

    out = []
    out.append("# Champion progression\n")
    out.append(
        "Generated from `experiments/index.jsonl`. Every figure is a recorded "
        "measurement; a metric a run never measured is left blank rather than "
        "filled in.\n"
    )
    out.append(f"Generated {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n")

    # A champion is a run that produced a model artifact and a dev score.
    # Retrieval adoptions and milestones are promoted too, but they are not
    # champions and putting them in the same table only makes it unreadable.
    accepted = [r for r in recs if r.get("decision") in ("PROMOTE", "BASELINE")]
    promoted = [r for r in accepted if metric(r, "dev_score") is not None]
    others = [r for r in accepted if metric(r, "dev_score") is None]
    out.append("## Champions, in order\n")
    out.append("| id | dev score | overall | easy | standard | hard | ECE | p50 ms | architecture |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in promoted:
        out.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["experiment_id"],
                cell(metric(r, "dev_score")),
                cell(metric(r, "overall_accuracy")),
                cell(metric(r, "easy")),
                cell(metric(r, "standard")),
                cell(metric(r, "hard")),
                cell(metric(r, "ece")),
                cell(metric(r, "p50_ms"), 1),
                (r.get("architecture") or "")[:60],
            )
        )

    if champ:
        out.append(f"\nCurrent champion: **{champ.get('experiment_id')}**, dev score "
                   f"{cell(champ.get('dev_score'))}, promoted {champ.get('promoted_at')}.\n")

    if others:
        out.append("\n## Adopted without becoming a champion\n")
        out.append(
            "Changes to retrieval or measurement, and benchmark milestones. "
            "They carry no dev score of their own because they change what a "
            "model is given or how it is judged, not the model.\n"
        )
        out.append("| id | what it changed |")
        out.append("|---|---|")
        for r in others:
            reason = (r.get("reason") or "").split(". ")[0]
            out.append(f"| {r['experiment_id']} | {reason[:150]} |")

    out.append("\n## Every experiment, including the ones that failed\n")
    out.append(
        "A rejected or aborted run is kept because the reason it failed is the "
        "result. Three learned rankers were rejected before the answer turned "
        "out to be contiguity, and two measurements were aborted because they "
        "could not have failed.\n"
    )
    out.append("| id | decision | what it establishes |")
    out.append("|---|---|---|")
    for r in recs:
        reason = (r.get("reason") or "").split(". ")[0]
        out.append(f"| {r['experiment_id']} | {r.get('decision','')} | {reason[:150]} |")

    counts = {}
    for r in recs:
        counts[r.get("decision", "?")] = counts.get(r.get("decision", "?"), 0) + 1
    out.append("\n" + ", ".join(f"{v} {k.lower()}" for k, v in sorted(counts.items())) + ".\n")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(out) + "\n")
    print(f"wrote {a.out}: {len(promoted)} champions, {len(recs)} experiments")


if __name__ == "__main__":
    main()
