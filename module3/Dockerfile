FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY module3/ ./module3/

RUN pip install --no-cache-dir -e ".[test]"

# Default: run the full test suite
CMD ["python", "-m", "pytest", "module3/tests/", "-v", "--asyncio-mode=auto"]
