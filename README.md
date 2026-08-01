# 6x6 Sudoku: Backtracking vs Neural Net Solver

A lightweight, dependency-free 6x6 Sudoku (2x3 boxes) app with two
solvers: constraint-propagation + backtracking (always correct), and a
small neural net trained from scratch on 20,000 generated puzzles
(usually isn't). The gap between those two is the actual point of this
project.

**Live demo**: https://prathiba-dr.github.io/sudoku-6x6-solver/

## Why two solvers

Same pairing as the [tic-tac-toe project](https://github.com/prathiba-dr/tic-tac-toe-ai)
this is a sibling to - search vs. a learned model - but Sudoku makes the
tradeoff sharper, because getting *one* cell right isn't the task. All 36
cells have to be simultaneously consistent, and that's exactly where a
per-cell classifier struggles.

## The honest result

The neural net (252-dim one-hot input, one 128-unit ReLU hidden layer, 36
independent 6-way softmax heads, trained via hand-rolled backprop + Adam)
evaluated on 1,000 fresh held-out puzzles:

| Metric | Result |
|---|---|
| Overall per-cell accuracy | 77.7% |
| Accuracy on clue cells (already given in the input) | 99.96% |
| Accuracy on hidden cells (actually inferred) | 63.6% |
| Puzzles solved **exactly** right | **0.0%** |
| Puzzles where the raw output is even a **valid** grid (no rule violations) | **0.0%** |
| Backtracking solver: puzzles solved exactly right | 100% |

77.7% sounds like a working solver. It isn't one - it's mostly measuring
that the model learned to copy the clues it's handed (99.96%, which is
trivial: those values are literally in the input). On the cells it
actually has to infer, it's right about 6 times in 10 - and because a
correct grid needs 22 hidden cells right *simultaneously*, not
independently, `0.636^22` is astronomically small. Zero exact solves
across 1,000 puzzles is the expected outcome of that math, not a bug.

The neural net's raw output isn't even internally consistent (no
duplicate in any row/column/box) on a single one of those 1,000 puzzles.
Each cell is predicted independently from a shared hidden layer, with
nothing forcing agreement between cells - there's no structural
constraint-satisfaction step, which is exactly what the backtracking
solver *is*.

Full numbers: [`reports/evaluation_results.json`](reports/evaluation_results.json).

## Does the NN help as a search hint, at least?

If a fully-correct standalone output is unrealistic, a more modest
question: can the NN's per-cell *guess* still speed up backtracking, by
suggesting which value to try first at each cell? This is a real,
documented technique (learned heuristics guiding CSP search) - tested it
by having backtracking try the NN's predicted digit before falling back
to normal ordering, and counting how many search nodes it takes.

Result: no measurable difference (0% node reduction on 14-clue puzzles,
slightly *negative* at 6-10 clues). The reason is straightforward once you
look at the baseline: plain constraint propagation already solves ~14-clue
6x6 puzzles in about 1 search node on average - there's essentially no
backtracking happening in the first place for a hint to speed up. This
technique's real value shows up on search spaces large enough to need
real backtracking (classic 9x9 Sudoku, bigger CSPs); 6x6 is too small to
demonstrate it, which is itself worth knowing before reaching for it on a
problem that doesn't need it.

## Structure

```
docs/                      # the actual site (served via GitHub Pages)
  index.html
  style.css
  script.js                 # board rendering, both solvers, manual play
  game-logic.js               # rules (rows/cols/2x3 boxes) - shared everywhere
  backtracking-solver.js        # constraint propagation + MRV backtracking
  generate-puzzle.js             # valid grid generation + unique-solution carving
  neural-net.js                   # forward pass only (inference) - no framework
  nn_weights.json                  # trained weights (~1.2 MB)
training/
  train_nn.js                # hand-rolled backprop + Adam (run: node training/train_nn.js)
  evaluate.js                  # the numbers above, on a fresh held-out set
reports/
  nn_training_results.json
  evaluation_results.json
```

No npm dependencies for the site itself. The training script only needs
Node - `game-logic.js`, `backtracking-solver.js`, `generate-puzzle.js`,
and `neural-net.js` are all written to work as both browser `<script>`
tags and Node `require()`s, so training and the live page share one
implementation of the rules, the solver, the generator, and the network -
not four things that could quietly disagree with each other.

## Run it locally

```
cd docs && python3 -m http.server 8000
# open http://localhost:8000

# retrain the neural net from scratch (~1-2 min)
node training/train_nn.js

# re-evaluate against a fresh held-out set
node training/evaluate.js
```

## Deployment

Served via GitHub Pages directly from `docs/` on `master` - no build
step, what's committed is what's live.
