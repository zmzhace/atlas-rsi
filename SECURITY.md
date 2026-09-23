# Security policy

## Threat model

Atlas-RSI treats candidate code and model output as untrusted. The proposer receives only public development-task descriptions and declarative incumbent state. Holdout and anchor descriptions are withheld. Private evaluator files are copied into the ephemeral workspace only after the solver exits.

## Runner guidance

- Use `DockerRunner` or a stronger sandbox for generated code.
- Pin images by digest and keep secrets outside the mounted workspace.
- The built-in Docker profile disables networking, drops Linux capabilities, enables `no-new-privileges`, limits memory/CPU/PIDs, and uses a read-only container root.
- The task workspace is writable because the solver must produce an answer. Treat all output as hostile until evaluators finish.
- `LocalProcessRunner` is for trusted commands only; it is not a security boundary.

## Remaining risks

Docker is not a perfect boundary against kernel vulnerabilities, resource side channels, or a compromised Docker daemon. Output-based prompt injection can also affect poorly designed external reviewers. For higher-risk experiments use a disposable VM or microVM, independent evaluators, pinned dependencies, outbound-deny network policy, and operator review before publishing artifacts.

Do not place credentials, proprietary holdouts, or host sockets in public fixtures. Report vulnerabilities privately through GitHub's security advisory workflow rather than a public issue.
