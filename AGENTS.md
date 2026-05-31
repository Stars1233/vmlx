# vMLX Python Engine Guard Worktree

This worktree is Python engine/panel work only.

Before any non-trivial work:

1. Read `.agents/STATUS.md`, `.agents/LOG.md`, and `.agents/MAIL.md`.
2. Coordinate in `.agents/MAIL.md` before live model/server probes.
3. Do not touch `/Users/eric/vmlx`, `/Users/eric/vmlx/swift`, or old Swift
   handoff material unless Eric explicitly asks for that exact path.

Current release-hardening boundary:

- DSV4 long-output/code/file-generation quality is not release-cleared unless
  the current objective proof digest says the row passed.
- Do not claim Python source, bundle, installed app, or release status is current
  without checking the corresponding artifact and `.agents` notes.
- No prompt rewrite, identifier substitution, postprocessing, or hidden
  monkeypatch is an acceptable production fix for DSV4 exact-code quality.

If you are here because `/Users/eric/vmlx/AGENTS.md` routed you out of the
deprecated wrapper, stay in this Python worktree until `.agents` explicitly says
to reconcile back to `/Users/eric/mlx/vllm-mlx`.
