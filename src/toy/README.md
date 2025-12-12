# Toy Model

A radically simplified cortical circuit model for understanding core dynamics.

## Purpose

This toy model exists to:
1. Build intuition about the model's behavior from first principles
2. Test hypotheses with minimal computational overhead
3. Serve as a pedagogical tool for explaining the main model

## Boundaries

The toy model:
- Lives in `src/toy/` with its own tests in `tests/test_toy_*.py`
- Does NOT share code with `src/model/` (except possibly config constants)
- Should be runnable independently
- Has its own presets and parameter definitions

## Status

**Scaffold only** - no implementation yet.

When ready to implement, start with:
1. Single population (E only)
2. Add one inhibitory population
3. Add spatial structure
4. Add layers
5. Compare to full model at each step

## Usage

```python
# Future usage (not implemented yet)
from src.toy import ToyCircuit

circuit = ToyCircuit()
circuit.update()
```

