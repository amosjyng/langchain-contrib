# langchain-contrib

This is an **unofficial** collection of utilities that are too experimental for [langchain proper](https://github.com/hwchase17/langchain), but are nonetheless generic enough to potentially be useful for multiple projects. Currently consists of code dumped from [ZAMM](https://github.com/amosjyng/zamm), but is of course open to contributions with lax procedures.

## Quickstart

```bash
pip install langchain-contrib
```

To add interop with [`vcr-langchain`](https://github.com/amosjyng/vcr-langchain), simply install it as well:

```bash
pip install vcr-langchain
```

## Historical features

This is a list of langchain-contrib features that have ended up making their own way into langchain proper, independently of this library:

- `ChoiceChain` has become langchain's `MultiRouteChain`
