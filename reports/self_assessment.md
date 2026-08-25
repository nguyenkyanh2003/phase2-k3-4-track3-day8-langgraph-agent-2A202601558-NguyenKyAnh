# Rubric Self-Assessment

This assessment maps repository evidence to `docs/RUBRIC.md`. It is not a substitute for the
official hidden-scenario grade.

| Category | Score | Evidence |
|---|---:|---|
| Architecture and state schema | 15/15 | Typed state, overwrite controls, append reducers, normalized audit events |
| Graph construction and wiring | 15/15 | Eleven registered nodes, four conditional routers, common `finalize → END` |
| LLM integration | 15/15 | Structured classifier, grounded answer generation, structured LLM-as-judge |
| Graph behavior | 20/20 | 7/7 sample routes pass; bounded retry, approval, and dead-letter paths observed |
| Persistence and recovery | 10/10 | SQLite WAL, unique thread IDs, reopen/history recovery test, optional Postgres |
| Metrics and tests | 15/15 | Valid metrics at 100% sample success; 39 tests pass; Ruff and mypy pass |
| Report and demo | 10/10 | Generated architecture, metrics, failure analysis, recovery, Mermaid, improvements |
| **Total** | **100/100** | Multiple bonus extensions are included |

## Automated evidence

- `outputs/metrics.json`: 7 scenarios, 100% success, 3 retries, 2 approval events,
  `resume_success=true`.
- Full test suite: 39 passed using the configured Gemini provider for smoke tests.
- Static quality: `ruff check src tests` and `mypy src` pass.
- Recovery: automated SQLite test closes the first connection, reopens the database, and
  verifies the final state and state-history count.

## Grading caveat

The repository satisfies every visible rubric item, so the rubric-based projection is
100/100. The official score may be lower if private scenarios expose an intent edge case,
provider quota prevents required live calls, or the instructor applies criteria not present
in the published rubric.
