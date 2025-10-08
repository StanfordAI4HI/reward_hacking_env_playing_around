# Code Organization Principles

_Modern software development practices for maintainable, scalable code_

## Core Design Principles

- [[Locality of Behavior]] - Behavior should be obvious from looking at the code unit itself
- [[Separation of Concerns]] - Divide programs into distinct sections addressing separate concerns
- [[Don't Repeat Yourself]] - Single, unambiguous representation of knowledge/logic
- [[Clean Code]] - Readable, maintainable, efficient; clarity over cleverness

## Fundamental Rules

- **Don't live with broken windows** - Fix problems immediately before they compound
- **YAGNI** (You Aren't Gonna Need It) - Only add functionality when necessary
- **KISS** (Keep It Simple, Stupid) - Avoid unnecessary complexity
- **Fail Fast** - Detect and report problems as early as possible
- **Principle of Least Surprise** - Code should behave as users expect

## [[SOLID Principles]]

Five core guidelines for maintainable software:

- Single Responsibility Principle
- Open/Closed Principle
- Liskov Substitution Principle
- Interface Segregation Principle
- Dependency Inversion Principle

## Code Quality & Style Guidelines

- **Meaningful names** - Variables, functions, classes should express intent
- **Small, focused functions** - Each function should do one thing well
- **Consistent formatting** - Follow established style guides (PEP 8 for Python)
- **Minimal comments** - Code should be self-documenting
- **Readability over conciseness** - Optimize for human understanding

## Python-Specific Guidelines

### Modern Python Practices

- **Type hints** - Use for function signatures and complex variables (PEP 484)
- **Dataclasses** - Structured data with automatic `__init__`, `__repr__`, `__eq__`
- **Pydantic** - Data validation and settings management with type enforcement
- **f-strings** - Modern string formatting over `.format()` or `%` (Python 3.6+)
- **Pathlib** - Use over `os.path` for file operations (Python 3.4+)
- **Context managers** - Use `with` statements for resource management

### Code Style & Structure

- **List/dict comprehensions** - Prefer over loops when readable
- **Exception handling** - Specific exceptions, avoid bare `except:`
- **Async/await** - Use for I/O-bound operations; handle exceptions properly
- **Immutable defaults** - Avoid mutable default arguments in functions
- **Underscore conventions** - `_private`, `__dunder__`, `_unused` variables

## Balancing Competing Principles

### Pragmatic Trade-offs

- **DRY vs Locality** - Sometimes duplication aids understanding
- **Abstraction vs Simplicity** - Don't over-engineer small projects
- **Separation vs Cohesion** - Balance modularization with code locality
- **Clean Code vs Delivery** - Perfect code doesn't ship; working code does
- **Rules vs Context** - Principles are heuristics, not absolute laws

### Clean Code Criticism & Reality

**Valid pushback against rigid clean code:**

- **Over-fragmentation** - Tiny functions can be harder to follow than cohesive blocks
- **Comment phobia** - "Self-documenting code" often isn't, especially for business logic
- **Context blindness** - Dogmatic rule-following ignores team/domain constraints
- **Cargo cult programming** - Following rules without understanding purpose

**Real-world considerations:**

- **Team cognitive load** - What's readable varies by experience level
- **Legacy constraints** - Clean code principles assume greenfield projects
- **Performance trade-offs** - Sometimes "dirty" code is more efficient
- **Delivery pressure** - Shipping functional code beats perfect code

**Nuanced approach:** Use clean code as starting point, optimize for your context

## Testing & Quality Assurance

### Testing Practices

- **Test-Driven Development (TDD)** - Write tests before implementation
- **Pytest fixtures** - Reusable test setup and teardown
- **Mocking** - Isolate units under test from dependencies
- **Parametrized tests** - Test multiple scenarios with same logic
- **Property-based testing** - Generate test cases automatically
- **Test coverage** - Aim for meaningful coverage, not just percentage

### Key Metrics to consider

- **Cyclomatic Complexity** - Number of independent paths through code (target: <10)
- **Code Coverage** - Percentage of code executed by tests
- **Maintainability Index** - Composite score based on complexity, size, and coupling
- **Technical Debt Ratio** - Cost to fix code quality issues vs. development cost
- **Code Duplication** - Percentage of duplicated code blocks

### Maintenance Practices

- **Prototype to Learn** - Build to understand, then rebuild properly
- **Put Abstractions in Code, Details in Metadata** - Configuration should be external
- **Refactor regularly** - Address both technical debt and [[Narrative Debt]]
- **Continuous Integration** - Automated testing and quality checks
- **Code reviews** - Peer feedback before merging changes

### Test-Driven Development

- Write tests+mock first → Make mock real to pass the tests.
  - ALWAYS run tests after changes
- Debugging
  - Workflow: Reproduce → Isolate → Understand → Fix → Prevent
  - Always ask yourself: "why did this bug occur" and "how to prevent similar issues"
  - For logic bugs: trace execution path step-by-step
- You can spawn sub-agents to do specialized work with dedicated context, for example writing tests, fixing bugs, or analyzing results.
  - Headless mode: `claude -p "prompt"` for CI automation and scripting

## Simplification Philosophy

- Prioritize radical simplification over incremental improvements:
  1. Delete over refactor
  2. Consolidate redundant code
  3. Remove non-critical validation/error handling
  4. Question every abstraction layer
  5. Prefer direct implementations
- Proactively identify opportunities for simplification during code review. Look
  for excessive abstractions, duplicate logic, and unused features. **Always**
  confirm with me before making any major changes.

## Domain-Specific Guidelines for This Repository

### Reward Function Design

- **Keep reward functions pure** - No side effects, only compute from observations
- **Single responsibility** - Each reward class should compute one coherent reward signal
- **Clear semantics** - Reward function names should describe what they optimize for
- **Document assumptions** - Note expected observation ranges, edge cases, domain knowledge

### When Adding New Reward Functions

- Follow the `RewardFunction` interface: `calculate_reward(prev_obs, action, obs) -> float`
- Use type hints with domain-specific observation types (`GlucoseObservation`, `PandemicObservation`, `TrafficObservation`)
- Place in appropriate domain file: `reward_functions/<domain>_gt_rew_fns.py`
- Register in `rl_utils/env_setups.py` using descriptive names
- Consider whether to use existing ground truth/proxy categories or add new reward types

### RL Training Code

- **Configuration externalization** - Use config files (`utils/*_config.py`) rather than hardcoding parameters
- **Checkpointing discipline** - Save checkpoints at regular intervals, document checkpoint structure
- **Logging clarity** - Distinguish between `modified_reward`, `original_reward`, `true_reward`, `proxy_reward`
- **Reproducibility** - Always set seeds, document environment versions
