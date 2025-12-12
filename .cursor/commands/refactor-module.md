Refactor the current Python file (module).

Definition:
- a module = this single .py file
- do not assume permission to change other files

Steps:
1. Summarize what this file currently does.
2. Identify structural problems:
   - mixed responsibilities
   - excessive length
   - duplication
   - unclear data flow
3. Propose a refactoring plan limited to this file.

Constraints:
- do not make code changes yet
- do not split files unless explicitly proposed
- do not change scientific behavior
- do not introduce new architectural patterns

Assumptions:
- this is research code
- flexibility and clarity matter more than abstraction

Output:
- a clear, step-by-step refactoring plan
- explanation of why each step improves the code
