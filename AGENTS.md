<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Documentation

The README.md contains a general overview of the project.
Openspec files in the `openspec` directory contain specifications and guidelines for various components of the project.
General documentation lives in the `docs` directory.
No other documentation files should be created outside of these locations.

Reproducability, easy of validation and verification, and clear documentation are as important to the project as the successful training of the model.
Where possible, infrastructure as code, automated tests, and clear documentation should be provided to ensure that results can be reproduced and verified by third parties.
