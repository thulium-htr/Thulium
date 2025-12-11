# Contributing to Thulium

Thank you for your interest in contributing to **Thulium**! We value contributions from the community to make this the best Open Source handwriting intelligence library.

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

## How to Contribute

### Reporting Issues

1.  Check the existing issues to avoid duplicates.
2.  Open a new issue with a clear title and description.
3.  Include steps to reproduce, expected behavior, and environment details.

### Pull Requests

1.  Fork the repository and clone it locally.
2.  Create a fresh branch: `git checkout -b feature/my-new-feature`.
3.  Make your changes.
4.  Run tests locally: `pytest`.
5.  Commit your changes following the [Conventional Commits](https://www.conventionalcommits.org/) convention.
6.  Push to your fork and submit a Pull Request.

## Development Setup

### Prerequisites

*   Python 3.10 or higher
*   Git

### Installation

```bash
git clone https://github.com/olaflaitinen/Thulium.git
cd Thulium
pip install -e .[dev]
```

## Style Guide

We enforce a strict coding style to maintain quality.

*   **Formatting**: We use `black`.
*   **Linting**: We use `ruff`.
*   **Type Hinting**: All public APIs must have type hints.

Run the linters before committing:

```bash
black thulium tests
ruff check thulium tests
```
