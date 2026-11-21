"""Fail-closed isolation for tests that put STUB commands in front of real ones (REV 4.2, Fix 4).

A test of sync_code.sh runs a script whose real commands (ssh, scp, tar) would act on the CLUSTER. Two independent protections, BOTH mandatory:
  1. the stubs are VERIFIED to be what the script will run: inside the very bash that runs the script, `command -v <tool>` must be `<stub dir>/<tool>` for every stubbed tool (and the tools the script needs
     must still resolve to something else). If not, the run is refused (pytest.fail, not skip) before the script starts;
  2. the host argument is a name that can never resolve (`.invalid`, RFC 2606), so even a mistake in 1 cannot reach a real machine. The production alias is refused outright.
PATH is extended INSIDE bash (`PATH="<stub dir>:$PATH"`) so the Windows PATH is never joined with ':' (that splits `C:\\...` at the drive letter and corrupts the rest of the path).
"""
import os
import subprocess

import pytest

TEST_HOST = "sync-test.invalid"
PRODUCTION_ALIASES = ("origami-fyp",)
SENTINEL = "STUB-RESOLUTION-FAILED"
NEEDED = "git sha256sum cut tr"

WRAPPER = r'''
d=$(cygpath -u "$STUB_DIR" 2>/dev/null || printf '%s' "$STUB_DIR")
PATH="$d:$PATH"; export PATH
for t in $STUB_TOOLS; do
  p=$(command -v "$t") || { echo "STUB-RESOLUTION-FAILED: $t not found" >&2; exit 97; }
  [ "$p" = "$d/$t" ] || { echo "STUB-RESOLUTION-FAILED: $t resolves to $p, not the stub $d/$t" >&2; exit 97; }
done
for t in $NEEDED_TOOLS; do
  p=$(command -v "$t") || { echo "STUB-RESOLUTION-FAILED: needed tool $t not found" >&2; exit 97; }
  case "$p" in "$d"/*) echo "STUB-RESOLUTION-FAILED: needed tool $t resolves into the stub directory" >&2; exit 97;; esac
done
exec "$@"
'''


def stub_script(tool, log):
    """A stub that records its own identity and does nothing else."""
    return f'#!/bin/sh\necho "stub:{tool} $*" >> "{log.as_posix()}"\ncat > /dev/null 2>&1 < /dev/null\nexit 0\n'


def refuse_production_host(argv):
    for item in argv:
        if str(item) in PRODUCTION_ALIASES:
            pytest.fail(f"refusing to run a test against the production host alias {item!r}")


def run_stubbed(bash, fakes, argv, *, tools=("ssh", "scp", "tar"), cwd=None, extra_env=None, timeout=300):
    """Run `argv` (a list: the script and its arguments) in `bash` with `fakes` first on PATH, after verifying that every tool in `tools` resolves to the stub. A resolution failure FAILS the test."""
    refuse_production_host(argv)
    if any(str(a).endswith("sync_code.sh") for a in argv) and not (len(argv) >= 2 and str(argv[-1]).endswith(".invalid")):
        pytest.fail("a sync_code.sh test must pass an explicit, never-resolving host (…invalid) as its last argument")
    env = {**os.environ, "STUB_DIR": str(fakes), "STUB_TOOLS": " ".join(tools), "NEEDED_TOOLS": NEEDED, **(extra_env or {})}
    done = subprocess.run([bash, "-c", WRAPPER, "stub-wrapper", *map(str, argv)], cwd=cwd, capture_output=True, text=True, env=env, timeout=timeout)
    if SENTINEL in done.stderr or done.returncode == 97 and "STUB-RESOLUTION" in done.stderr:
        pytest.fail(f"stub resolution failed, the script was NOT run: {done.stderr.strip()}")
    return done


def assert_only_stubs_were_called(log_text):
    """Every recorded call is a stub's own line and none names the production host."""
    for line in log_text.splitlines():
        assert line.startswith("stub:"), f"a non-stub call was logged: {line!r}"
        assert not any(alias in line for alias in PRODUCTION_ALIASES), f"the production alias reached a stub: {line!r}"
