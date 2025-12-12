Update tests to reflect an intentional change in model behavior.

Precondition:
- briefly state (1–2 sentences) what changed scientifically

Scope:
- only modify tests that are now invalid
- do not touch production code

Rules:
- preserve test strictness where possible
- do NOT weaken assertions without justification
- numerical tolerances must be motivated

Tasks:
- update or rewrite affected tests
- ensure tests reflect the new intended behavior

Output:
- list of modified test files
- explanation of how the test contract changed
