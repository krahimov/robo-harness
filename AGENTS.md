# Working on Robot Harness Studio

- Keep the product centered on demonstration-conditioned inference with fixed model weights.
- The replay baseline is not an ICL model. Preserve this distinction in code, tests, and user-facing copy.
- The supplied Cartesian gripper uses an idealized grasp. Do not label it a commercial arm or claim sim-to-real validity.
- Keep task requirements separate from judge rubrics. New rubric versions must not rewrite previous evaluations or experiment snapshots.
- Do not place endpoint credentials in frontend code, database records, artifacts, or logs.
- Backend tests run with `cd backend && uv run pytest -q`; frontend verification uses `cd web && npm run build`.
- Keep generated recordings in `.data/`, not in Git. API changes must preserve import/export and review/promotion behavior.
- Model-specific camera, context, and action transforms belong in adapters.
