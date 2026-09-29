# Backend Development Guidelines

> Best practices for backend development in this project.

---

## Overview

This directory contains guidelines for backend development. Fill in each file with your project's specific conventions.

---

## Guidelines Index

| Guide | Description | Status |
|-------|-------------|--------|
| [Training Foundation](./training-foundation.md) | P0 device, tensor, optimizer, checkpoint and doctor contracts | Implemented |
| [Data Pipeline](./data-pipeline.md) | P1 CSV, preprocessing, scaler, cache, windows and input contracts | Implemented |
| [Training Loop](./training-loop.md) | P1 baseline RNG, masked training, inference bundles and acceptance | Implemented |
| [Directory Structure](./directory-structure.md) | Module organization and file layout | To fill |
| [Database Guidelines](./database-guidelines.md) | ORM patterns, queries, migrations | To fill |
| [Error Handling](./error-handling.md) | Error types, handling strategies | To fill |
| [Quality Guidelines](./quality-guidelines.md) | Code standards, forbidden patterns | To fill |
| [Logging Guidelines](./logging-guidelines.md) | Structured logging, log levels | To fill |

---

## Pre-Development Checklist

For Rust core, nn, train, or CLI work, read [Training Foundation](./training-foundation.md) and the task PRD/design/implementation plan. Data/configuration/input work also requires [Data Pipeline](./data-pipeline.md). Baseline/training/evaluation work also requires [Training Loop](./training-loop.md). Other listed documents remain templates until their own scope is implemented.

## Quality Check

Run `cargo fmt --all --check`, CPU and CUDA `cargo clippy --workspace --all-targets -- -D warnings` (add `--features cuda` before `--` for CUDA), and both workspace test variants. Follow docs/environment.md for native Windows CUDA setup. Run the private-data guard before staging.

## How to Fill These Guidelines

For each guideline file:

1. Document your project's **actual conventions** (not ideals)
2. Include **code examples** from your codebase
3. List **forbidden patterns** and why
4. Add **common mistakes** your team has made

The goal is to help AI assistants and new team members understand how YOUR project works.

---

**Language**: All documentation should be written in **English**.
