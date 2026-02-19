# Empty files audit

This audit reviews empty Python files and applies one of two actions:
- **Delete** when there are no in-repo references.
- **Keep with minimal stable implementation** when the file is part of package compatibility.

## Reference audit method

Commands used:

```bash
rg -n "^(from|import) " -g'*.py'
rg -n "app\\.utils|token_counter|prompt_builder|logger|(^|\\s)import config|from config|test_memory|test_tutor_chatbot" -g'*.py'
```

## Decisions

### Deleted (no references found)

- `config.py`
- `app/utils/__init__.py`
- `app/utils/logger.py`
- `app/utils/prompt_builder.py`
- `app/utils/token_counter.py`
- `scripts/reset_databases.py`
- `tests/test_memory.py`
- `tests/test_tutor_chatbot.py`
- `tests/__init__.py`

### Kept with minimal stable implementation

- `app/__init__.py`
- `app/conversation/__init__.py`
- `app/memory/__init__.py`
- `app/memory/processors/__init__.py`

These package marker modules now include descriptive docstrings to make the compatibility intent explicit.
