/*
 * Evaluates the trained NN and the backtracking solver on a FRESH
 * held-out set (different seed than both training and the training
 * script's own internal test set) across several metrics - overall
 * per-cell accuracy alone is misleadingly reassuring for this task, so
 * this breaks it down further.
 */
const fs = require("fs");
const path = require("path");
const GL = require("../docs/game-logic.js");
const { forward, encodeGrid, N_CELLS, N_CLASSES } = require("../docs/neural-net.js");
const { solve } = require("../docs/backtracking-solver.js");
const { generateDataset } = require("../docs/generate-puzzle.js");

const SEARCH_BENCHMARK_SAMPLES = 300;

const TEST_SAMPLES = 1000;
const TARGET_CLUES = 14;
const SEED = 99999; // distinct from training (1234) and training's internal test set (5678)

function loadNet() {
  const raw = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "docs", "nn_weights.json")));
  return {
    hiddenDim: raw.hiddenDim,
    W1: Float32Array.from(raw.W1), b1: Float32Array.from(raw.b1),
    W2: Float32Array.from(raw.W2), b2: Float32Array.from(raw.b2),
  };
}

function nnPredict(net, puzzle) {
  const input = encodeGrid(puzzle);
  const { probs } = forward(net, input);
  const grid = new Array(N_CELLS);
  for (let cell = 0; cell < N_CELLS; cell++) {
    let best = 0, bestP = -1;
    for (let k = 0; k < N_CLASSES; k++) {
      const p = probs[cell * N_CLASSES + k];
      if (p > bestP) { bestP = p; best = k; }
    }
    grid[cell] = best + 1;
  }
  return grid;
}

function main() {
  const net = loadNet();
  const testSet = generateDataset(TEST_SAMPLES, SEED, TARGET_CLUES);

  let clueCellsCorrect = 0, clueCellsTotal = 0;
  let hiddenCellsCorrect = 0, hiddenCellsTotal = 0;
  let nnExactGrids = 0;
  let nnValidGrids = 0;
  let backtrackingCorrect = 0;

  for (const { puzzle, solved } of testSet) {
    const nnGrid = nnPredict(net, puzzle);
    let allCorrect = true;
    for (let i = 0; i < N_CELLS; i++) {
      const isClue = puzzle[i] !== 0;
      const correct = nnGrid[i] === solved[i];
      if (isClue) { clueCellsTotal++; if (correct) clueCellsCorrect++; }
      else { hiddenCellsTotal++; if (correct) hiddenCellsCorrect++; }
      if (!correct) allCorrect = false;
    }
    if (allCorrect) nnExactGrids++;
    if (GL.isValidGrid(nnGrid) && GL.isComplete(nnGrid)) nnValidGrids++;

    const backtrackingSolution = solve(puzzle);
    if (backtrackingSolution && JSON.stringify(backtrackingSolution) === JSON.stringify(solved)) {
      backtrackingCorrect++;
    }
  }

  // Does using the NN's predicted digit as a search hint (try it first,
  // if it's a legal candidate at that cell) reduce backtracking effort,
  // even though the NN's raw output alone is essentially never a fully
  // valid grid on its own?
  let plainNodesTotal = 0, guidedNodesTotal = 0;
  const benchmarkSet = testSet.slice(0, SEARCH_BENCHMARK_SAMPLES);
  for (const { puzzle } of benchmarkSet) {
    const nnGrid = nnPredict(net, puzzle);

    const plainCounter = { nodes: 0 };
    solve(puzzle, { counter: plainCounter });
    plainNodesTotal += plainCounter.nodes;

    const guidedCounter = { nodes: 0 };
    solve(puzzle, {
      counter: guidedCounter,
      orderCandidates: (index, candidates) => {
        const hint = nnGrid[index];
        if (!candidates.includes(hint)) return candidates;
        return [hint, ...candidates.filter((v) => v !== hint)];
      },
    });
    guidedNodesTotal += guidedCounter.nodes;
  }

  const results = {
    n_test_puzzles: TEST_SAMPLES,
    target_clues: TARGET_CLUES,
    neural_net: {
      clue_cell_accuracy: round(clueCellsCorrect / clueCellsTotal),
      hidden_cell_accuracy: round(hiddenCellsCorrect / hiddenCellsTotal),
      overall_cell_accuracy: round((clueCellsCorrect + hiddenCellsCorrect) / (clueCellsTotal + hiddenCellsTotal)),
      exact_grid_accuracy: round(nnExactGrids / TEST_SAMPLES),
      valid_and_complete_grid_rate: round(nnValidGrids / TEST_SAMPLES),
    },
    backtracking_solver: {
      exact_grid_accuracy: round(backtrackingCorrect / TEST_SAMPLES),
    },
    nn_guided_search_benchmark: {
      n_puzzles: SEARCH_BENCHMARK_SAMPLES,
      avg_nodes_plain_mrv: round(plainNodesTotal / SEARCH_BENCHMARK_SAMPLES),
      avg_nodes_nn_guided: round(guidedNodesTotal / SEARCH_BENCHMARK_SAMPLES),
      node_reduction_pct: round(1 - guidedNodesTotal / plainNodesTotal),
    },
  };

  function round(x) { return Math.round(x * 10000) / 10000; }

  console.log(JSON.stringify(results, null, 2));
  fs.writeFileSync(
    path.join(__dirname, "..", "reports", "evaluation_results.json"),
    JSON.stringify(results, null, 2)
  );
}

main();
