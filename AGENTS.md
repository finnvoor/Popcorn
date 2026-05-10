# AGENTS.md

## Build, Test, and Format

This project uses [mise](https://mise.jdx.dev/) for task running. Use the following commands:

- `mise build` — build the project
- `mise test` — run the test suite
- `mise format` — format the code (run this after making code changes)

Always use these `mise` tasks instead of invoking the underlying tools (e.g. `swift build`, `swift test`) directly. Run `mise format` after any code changes.
