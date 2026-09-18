# EU AI Act Compliance Assistant

> **Status:** Phase 0 (scaffolding) — not yet functional. See [`RAG_PORTFOLIO_SPEC.md`](RAG_PORTFOLIO_SPEC.md) and [`PROGRESS.md`](PROGRESS.md) for current state.

## What this is

A retrieval-augmented generation (RAG) system that answers questions about obligations under the EU AI Act (Regulation (EU) 2024/1689) — e.g. "which obligations apply to my AI system, and when." It retrieves from a corpus spanning the core Regulation, the Digital Omnibus amendment, supporting Commission guidance and Codes of Practice, and the GDPR, and answers with citations back to the source text.

## Who it's for

Built as a portfolio project demonstrating senior/lead-level AI engineering judgment: evaluation discipline (every quality claim backed by a golden-set score, not a vibe), production awareness (tracing, CI-gated eval, deployed and reachable, not just a local demo), and documented trade-off decisions (see [`DECISIONS.md`](DECISIONS.md)). It's aimed at engineers and hiring managers evaluating RAG-system design skill, not at providing actual legal advice.

## Why this domain

The EU AI Act corpus is current and specific to European employers: GPAI transparency obligations became enforceable with real fines on 2 August 2026, and the high-risk-system timeline was just restructured by the "Digital Omnibus" amendment (in force since 27 July 2026). The corpus spans multiple distinct, cross-referencing documents rather than one regulation that fits in a single context window, and it includes a genuine versioning challenge: the Omnibus amendment changed dates written into the original Regulation, which is a real, checkable test of whether the system surfaces the current obligation or a stale one — not a synthetic benchmark.

## Architecture, results, and how to run

To be filled in as the project reaches Phase 5 (see the phase plan in [`RAG_PORTFOLIO_SPEC.md`](RAG_PORTFOLIO_SPEC.md)): architecture diagram, results table (baseline vs. each improvement), and known limitations / failure modes will land here once there's a working, evaluated system to describe.
