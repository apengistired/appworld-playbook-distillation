# Examples

Use this directory for public, sanitized examples. A sample named
`playbook_top20.txt` can use the same format as the output of
`scripts/playbook_topn.py`.

```text
## VERIFICATION_CHECKLIST
[ver-example] helpful=1 harmful=0 :: In Gmail, verify the changed label with a read-only query before submitting.
```

For a failure and repair comparison, include the intended behavior, a short
synthetic code or observation excerpt, and the corrected next action. For
example, an agent that sees exactly ten search results can check for another
page before treating the results as exhaustive.

Use fabricated task inputs and API responses. Keep AppWorld reference answers,
raw trajectories, credentials, and model checkpoints out of public examples.
