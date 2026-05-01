# tests/

Pytest suite for the project. The default invocation runs every test
that does not require a GPU.

## Layout

- `conftest.py` - registers the `gpu` marker and skips GPU tests when
  CUDA is unavailable.
- `test_utils.py` - exercises the helpers under `src/fmllm/utils/`.

## Running

Local (no GPU):

```
uv run pytest -m "not gpu" -v
```

Remote (with all four H100s visible):

```
uv run pytest -v
```

## Markers

- `gpu` - the test requires a CUDA device. Local runs skip these
  automatically with a pointer to `docs/remote-setup.md`.
