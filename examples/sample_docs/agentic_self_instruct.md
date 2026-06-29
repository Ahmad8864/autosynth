# Agentic Self-Instruct: notes

Agentic Self-Instruct constructs training examples through a multi-agent loop: a challenger generates a candidate, a quality verifier audits it for leakage and formatting, a weak solver and a strong solver attempt it, and a judge scores both against an auto-generated rubric. A candidate is accepted only when the strong solver consistently outperforms the weak one by a wide margin and quality checks pass.

A reflective step summarizes why prior rounds failed (too easy, strong failed, quality rejected) and feeds targeted bullets back to the challenger, which then attempts a different reasoning angle. The process repeats until accepted or a round budget is exhausted.

A secondary, meta-optimization loop evolves the orchestrator's instructions themselves: failure trajectories are mined, a code-editing agent applies a mutation to the harness, and the mutation is kept only if validation improves.
