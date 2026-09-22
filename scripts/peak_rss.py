#!/usr/bin/env python3
"""Run a command and report its peak resident set size in megabytes.

    python3 scripts/peak_rss.py --rss-out FILE -- cmd arg...

The memory term of the dev score is a real cost of shipping a model, so
it is measured rather than assumed. `/usr/bin/time` is not present in
every image, but `getrusage(RUSAGE_CHILDREN)` is always available.

The command's exit status is passed through, so this is transparent to a
caller that checks it. If the peak cannot be read the file is left
empty, which the caller should treat as "not measured" rather than
substituting a guess.
"""
import resource
import subprocess
import sys


def main():
    argv = sys.argv[1:]
    out = None
    if argv and argv[0] == "--rss-out":
        out, argv = argv[1], argv[2:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise SystemExit(__doc__)

    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    rc = subprocess.call(argv)
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    # ru_maxrss is the high-water mark across ALL children reaped so far,
    # so it only rises; taking the max with the prior value keeps it
    # honest when an earlier child was larger, and the caller is told the
    # figure covers this process tree.
    peak_kb = max(after, before)
    if out and peak_kb > 0:
        with open(out, "w") as f:
            f.write("%.1f\n" % (peak_kb / 1024.0))
    return rc


if __name__ == "__main__":
    sys.exit(main())
