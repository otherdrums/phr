# Contributing to PHR

## Contributor License Agreement (CLA)

By submitting a pull request, you affirm that:

1. You are the sole author of the contribution, or you have the right to
   submit it under the project license (GNU AGPLv3).

2. You grant the project maintainers a perpetual, worldwide, non-exclusive
   license to use, reproduce, and distribute your contribution under the
   AGPLv3, and to relicense it under any future version of the AGPL published
   by the Free Software Foundation.

3. You understand that your contribution will be made available under the
   same AGPLv3 license as the rest of the project.

4. If your employer has rights to intellectual property you create, you
   have received permission to make this contribution on behalf of that
   employer.

All contributions are subject to these terms. Submission of a pull request
constitutes acceptance of this agreement.

## Development Setup

```bash
git clone https://github.com/otherdrums/phr.git
cd phr
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Code Style

- Follow existing conventions in the codebase (no comments unless necessary,
  mimic surrounding code style)
- Run `python -m py_compile` on any modified `.py` files before committing
- Keep the `phr/` package flat — no nested subdirectories beyond `tests/`

## Pull Request Process

1. Open an issue describing the bug or feature first
2. Create a branch from `master`
3. Keep changes focused — one concern per PR
4. Include a clear description of what changed and why
5. Verify `phr` and `tests` directories import successfully after your changes
