#!/usr/bin/env python3
"""Champion / challenger bookkeeping.

  record   append an experiment to experiments/index.jsonl
  promote  make an experiment the champion (writes CURRENT_CHAMPION.json)
  show     print the champion and the last experiments
  compare  decide PROMOTE / REJECT from two dev-score files
"""
import argparse, datetime, json, os, platform, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "experiments/index.jsonl"
CHAMP = ROOT / "experiments/CURRENT_CHAMPION.json"
# A challenger must beat the champion's dev score by this margin to promote.
MARGIN = 0.005


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def hardware():
    try:
        cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        cpu = platform.processor()
    try:
        mem = round(int([l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal")][0]) / 1024 / 1024)
    except Exception:
        mem = None
    return {"cpu": cpu, "cores": os.cpu_count(), "ram_gb": mem, "gpu": None, "platform": platform.platform()}


def load_champion():
    return json.loads(CHAMP.read_text()) if CHAMP.exists() else None


def cmd_record(a):
    rec = {
        "experiment_id": a.id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "parent_champion": (load_champion() or {}).get("experiment_id"),
        "hypothesis": a.hypothesis,
        "architecture": a.architecture,
        "dataset_version": a.dataset,
        "model_init": a.model_init,
        "hyperparameters": json.loads(a.hparams) if a.hparams else {},
        "seed": a.seed,
        "training_time_s": a.training_time,
        "hardware": hardware(),
        "git_commit": git_commit(),
        "parameter_count": a.params,
        "artifact_bytes": a.artifact_bytes,
        "dev_metrics": json.load(open(a.dev_metrics)) if a.dev_metrics else {},
        "calibration_metrics": json.load(open(a.calibration_metrics)) if a.calibration_metrics else {},
        "latency_ms": json.loads(a.latency) if a.latency else {},
        "memory_mb": a.memory_mb,
        "decision": a.decision,
        "reason": a.reason,
        "artifacts": a.artifacts.split(",") if a.artifacts else [],
    }
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with INDEX.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"recorded {a.id}: {a.decision}")


def cmd_promote(a):
    recs = [json.loads(l) for l in INDEX.read_text().splitlines() if l.strip()] if INDEX.exists() else []
    rec = next((r for r in reversed(recs) if r["experiment_id"] == a.id), None)
    if rec is None:
        sys.exit(f"experiment {a.id} not found in {INDEX}")
    champ = {
        "experiment_id": a.id,
        "promoted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "architecture": rec["architecture"],
        "model_dir": a.model_dir,
        "neural_dir": a.neural_dir,
        "parameter_count": rec.get("parameter_count"),
        "dev_metrics": rec.get("dev_metrics", {}),
        "dev_score": (rec.get("dev_metrics") or {}).get("dev_score"),
        "latency_ms": rec.get("latency_ms", {}),
        "previous_champion": (load_champion() or {}).get("experiment_id"),
        "note": a.note or "",
    }
    CHAMP.write_text(json.dumps(champ, indent=2) + "\n")
    print(f"champion is now {a.id} (dev_score {champ['dev_score']})")
    if a.neural_dir:
        print(
            "\nThe new champion is neural. These files still claim the engine has no\n"
            "neural network and must be corrected in the same commit (see the\n"
            "promotion checklist in docs/NEURAL.md):\n"
            "  README.md\n"
            "  crates/core/src/lib.rs\n"
            "  crates/cli/src/main.rs\n"
            "  docs/LIMITATIONS.md\n"
            "  docs/BENCHMARKS.md\n"
            "  scripts/prepare_data.py\n"
        )


def cmd_show(a):
    champ = load_champion()
    print("CURRENT_CHAMPION:", json.dumps(champ, indent=2) if champ else "none")
    if INDEX.exists():
        recs = [json.loads(l) for l in INDEX.read_text().splitlines() if l.strip()]
        print(f"\n{len(recs)} experiments:")
        for r in recs[-a.tail :]:
            d = (r.get("dev_metrics") or {}).get("dev_score")
            print(f"  {r['experiment_id']:<10} {r['decision']:<8} dev_score={d if d is None else round(d,4)} {r['architecture'][:48]} :: {r['reason'][:70]}")


def cmd_compare(a):
    ch = json.load(open(a.champion))
    cl = json.load(open(a.challenger))
    gain = cl["dev_score"] - ch["dev_score"]
    ece_ok = (cl.get("ece") or 0) <= (ch.get("ece") or 0) + 0.05
    schema_ok = (cl.get("schema_validity") or 1.0) >= 0.999
    decision = "PROMOTE" if (gain > MARGIN and ece_ok and schema_ok) else "REJECT"
    reasons = []
    reasons.append(f"dev_score {ch['dev_score']:.4f} -> {cl['dev_score']:.4f} ({gain:+.4f})")
    if not ece_ok:
        reasons.append(f"calibration collapse: ECE {ch.get('ece')} -> {cl.get('ece')}")
    if not schema_ok:
        reasons.append(f"schema validity {cl.get('schema_validity')}")
    for k in ("standard", "hard", "easy"):
        a_, b_ = ch["components"].get(k), cl["components"].get(k)
        if a_ is not None and b_ is not None:
            reasons.append(f"{k} {a_:.3f} -> {b_:.3f}")
    out = {"decision": decision, "gain": gain, "reason": "; ".join(reasons)}
    print(json.dumps(out, indent=2))
    return 0 if decision == "PROMOTE" else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--id", required=True)
    r.add_argument("--hypothesis", default="")
    r.add_argument("--architecture", default="")
    r.add_argument("--dataset", default="")
    r.add_argument("--model-init", default="")
    r.add_argument("--hparams", default="")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--training-time", type=float, default=0.0)
    r.add_argument("--params", type=int, default=0)
    r.add_argument("--artifact-bytes", type=int, default=0)
    r.add_argument("--dev-metrics")
    r.add_argument("--calibration-metrics")
    r.add_argument("--latency", default="")
    r.add_argument("--memory-mb", type=float, default=0.0)
    r.add_argument("--decision", required=True, choices=["PROMOTE", "REJECT", "ABORT", "BASELINE"])
    r.add_argument("--reason", default="")
    r.add_argument("--artifacts", default="")
    r.set_defaults(func=cmd_record)
    p = sub.add_parser("promote")
    p.add_argument("--id", required=True)
    p.add_argument("--model-dir", default="model")
    p.add_argument("--neural-dir", default=None)
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_promote)
    s = sub.add_parser("show")
    s.add_argument("--tail", type=int, default=20)
    s.set_defaults(func=cmd_show)
    c = sub.add_parser("compare")
    c.add_argument("--champion", required=True)
    c.add_argument("--challenger", required=True)
    c.set_defaults(func=cmd_compare)
    a = ap.parse_args()
    rc = a.func(a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
