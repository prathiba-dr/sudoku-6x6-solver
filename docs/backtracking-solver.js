// Constraint-propagation + backtracking solver. This is the ground truth
// used everywhere else in this project: generating puzzles, checking
// puzzle uniqueness, producing training labels for the neural net, and
// grading the neural net's guesses.
(function (root) {
  const GL = typeof module !== "undefined" && module.exports
    ? require("./game-logic.js")
    : window.GameLogic;

  // repeatedly fill any cell that has exactly one candidate left - doesn't
  // require search, and shrinks the problem before backtracking has to guess
  function propagate(grid) {
    grid = grid.slice();
    let progress = true;
    while (progress) {
      progress = false;
      for (let i = 0; i < grid.length; i++) {
        if (grid[i] !== 0) continue;
        const cands = GL.candidates(grid, i);
        if (cands.length === 0) return null; // contradiction - unsolvable from here
        if (cands.length === 1) {
          grid[i] = cands[0];
          progress = true;
        }
      }
    }
    return grid;
  }

  function findMRVCell(grid) {
    // most-constrained-cell heuristic: pick the empty cell with fewest
    // candidates - fails fast on wrong branches instead of wandering
    let best = -1;
    let bestCands = null;
    for (let i = 0; i < grid.length; i++) {
      if (grid[i] !== 0) continue;
      const cands = GL.candidates(grid, i);
      if (bestCands === null || cands.length < bestCands.length) {
        best = i;
        bestCands = cands;
        if (cands.length <= 1) break;
      }
    }
    return { index: best, candidates: bestCands };
  }

  // returns a solved grid, or null if unsolvable
  function solve(grid) {
    const propagated = propagate(grid);
    if (propagated === null) return null;
    if (GL.isComplete(propagated)) return propagated;

    const { index, candidates } = findMRVCell(propagated);
    for (const val of candidates) {
      const next = propagated.slice();
      next[index] = val;
      const result = solve(next);
      if (result !== null) return result;
    }
    return null;
  }

  // counts solutions up to `limit` (used to verify a puzzle has a unique
  // solution when generating puzzles - stops early once limit is hit)
  function countSolutions(grid, limit = 2) {
    const propagated = propagate(grid);
    if (propagated === null) return 0;
    if (GL.isComplete(propagated)) return 1;

    const { index, candidates } = findMRVCell(propagated);
    let count = 0;
    for (const val of candidates) {
      const next = propagated.slice();
      next[index] = val;
      count += countSolutions(next, limit - count);
      if (count >= limit) break;
    }
    return count;
  }

  const BacktrackingSolver = { solve, propagate, countSolutions };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = BacktrackingSolver;
  } else {
    root.BacktrackingSolver = BacktrackingSolver;
  }
})(typeof window !== "undefined" ? window : globalThis);
