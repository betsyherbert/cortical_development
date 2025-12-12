Clean up the currently open Python file in an integration-safe way.

Scope:
- primary focus on this file
- you may read and modify other files if needed to preserve compatibility

Allowed changes:
- simplify logic
- improve naming
- refactor internal structure
- change function or class interfaces IF:
  - all call sites are updated
  - behavior is preserved

Constraints:
- scientific behavior must not change
- diffs must remain readable
- no architectural redesigns

Process:
1. Identify any interface changes required.
2. Update all dependent code.
3. Run relevant tests if available.

Output:
1. Summary of changes
2. List of files modified
3. Confirmation that compatibility was preserved
