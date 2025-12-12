Run a minimal set of smoke tests.

Goal:
- quickly detect catastrophic breakage

Suggested coverage:
- simulation initializes
- one integration step runs
- no NaNs or shape mismatches
- core connectivity objects construct successfully

Command:
- pytest -q -m smoke

Rules:
- do NOT modify code
- do NOT modify tests

Output:
- pass/fail summary
- first failure if any
