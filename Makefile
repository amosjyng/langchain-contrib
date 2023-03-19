.PHONY: format lint test tests docs clean release

all: format lint test docs

format:
	poetry run autoflake .
	poetry run black .
	poetry run isort .

lint:
	poetry run mypy . --exclude docs/
	poetry run flake8 . --extend-exclude docs/
	poetry run black . --check
	poetry run isort . --check

test: tests
tests:
	poetry run pytest -v

docs:
	poetry run sphinx-apidoc -f -o docs/modules/ langchain_contrib

clean:
# https://stackoverflow.com/a/41386937/257583
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

release:
	test -z "$$(git status --porcelain)"
	git checkout main
	git pull
	poetry version patch
	git checkout -b "release/v$$(poetry version -s)"
	git commit -am "Releasing version v$$(poetry version -s)"
	git tag -a -m "Releasing version v$$(poetry version -s)" "v$$(poetry version -s)"
	poetry publish --build --username $$PYPI_USERNAME --password $$PYPI_PASSWORD
# git push at the very end to get Github PR link
	git push --set-upstream origin "release/v$$(poetry version -s)"
# --follow-tags seems to suppress Github message output
	git push --follow-tags