Analyze the current test failures.

Tasks:
- for each failing test:
  1. explain what the test is asserting
  2. explain why it might be failing
  3. classify the failure as:
     - [bug in code]
     - [intentional behavioral change]
     - [brittle or outdated test]

Constraints:
- do NOT modify code
- do NOT modify tests
- do NOT suggest loosening tolerances

Output:
- a structured list of failures
- classification + reasoning for each
