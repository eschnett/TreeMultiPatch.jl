# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Contents**

- [Read first, and where things go](#read-first-and-where-things-go)
- [What this is](#what-this-is)
- [Commands](#commands)
- [Rules and sharp edges](#rules-and-sharp-edges)
- [Repository facts](#repository-facts)

## Read first, and where things go

Read `CODE.md` before changing anything non-trivial: it says why things
are the way they are. The files split as in TreeAMR (its `CLAUDE.md`):
`README.md` for users, `CODE.md` the current design in the present tense,
`PLAN.md` the milestones, `HISTORY.md` who decided what and when, and the
records of finished milestones. Every fact lives in one file. Keep the
tables of contents current. The conventions are TreeAMR's: 4-space
indent, about 90 columns, prose-first docstrings, `ArgumentError`s that
say why, testsets named as claims, each opening with a comment naming
the failure it guards.

## What this is

The multi-patch layouts of TreeAMR's forests of patches (TreeAMR's M20):
the annulus and the two balls, their glue, maps, frames and atlases. No
physics.

## Commands

The suite, about 35 s:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

One file or several, after the helpers:

```bash
julia --project=test test/runtests.jl atlas_tests
```

The kernel test on Metal, from a scratch environment that develops this
checkout and adds Metal, KernelAbstractions and the test dependencies
(`CODE.md`, "Tests"):

```bash
TREEMULTIPATCH_TEST_BACKEND=metal julia --project=<env> test/runtests.jl device_tests
```

## Rules and sharp edges

- **TreeAMR's exports and public `chart_*` only**, `[compat] TreeAMR =
  "0.2"`. Needing an internal means it belongs in TreeAMR, exported.
- **Everything at a point runs in a kernel**: `isbits` layouts, no
  allocation, concrete inferred types. No `Base.setindex` at a run-time
  index, no variable both captured by a closure and reassigned; the
  suite checks the types ("Code structure" in `CODE.md`).
- **Every map promotes** to the evaluation type and forms its constants
  there, and calls `sincos_local`/`atan_local`, not `sin`/`cos`/`sincos`
  (`CODE.md`, "Number types").
- **Maps hold past the box**: a change to a map is tested half a root
  past every box.
- **Every radial coordinate runs outward**; a glue change is tested by
  "A glued face is one surface under both patches' maps".
- **The scheme stays in the application** (`CODE.md`, "What stays in the
  applications").

## Repository facts

- Remote: `github.com/eschnett/TreeMultiPatch.jl`, branch `main`.
  Unregistered; registering is Erik's (M3 in `PLAN.md`).
- CI (`.github/workflows/CI.yml`) tests on Julia 1.13 in the two tiers
  of every Tree* repository: a pull request runs the Linux cells only,
  serial and at four threads, without coverage; macOS and coverage
  (Codecov, through the `CODECOV_TOKEN` secret) run on `main`, weekly
  and on dispatch. So run the suite locally before opening a pull
  request. TagBot (`TagBot.yml`) tags each registered version and needs
  a write deploy key behind the `DOCUMENTER_KEY` secret, as in TreeAMR.
- `Manifest.toml` files are gitignored; `test/Project.toml` locates the
  package through `[sources]`.
- Julia 1.13, the floor of every Tree* package.
