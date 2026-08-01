// Shared 6x6 Sudoku rules - used by the browser page, the puzzle
// generator, the backtracking solver, and the NN training script, so
// there's exactly one definition of "what's a valid 6x6 Sudoku" instead
// of several that could quietly disagree.
//
// Board: 36 cells, flat array, 0 = empty, 1-6 = filled. Boxes are 2 rows x
// 3 cols (3 box-rows x 2 box-cols = 6 boxes), the standard 6x6 layout.
(function (root) {
  const SIZE = 6;
  const BOX_H = 2;
  const BOX_W = 3;

  function rowOf(i) { return Math.floor(i / SIZE); }
  function colOf(i) { return i % SIZE; }
  function boxOf(i) {
    const r = rowOf(i), c = colOf(i);
    return Math.floor(r / BOX_H) * (SIZE / BOX_W) + Math.floor(c / BOX_W);
  }

  function cellsInRow(r) { return Array.from({ length: SIZE }, (_, c) => r * SIZE + c); }
  function cellsInCol(c) { return Array.from({ length: SIZE }, (_, r) => r * SIZE + c); }
  function cellsInBox(b) {
    const boxRow = Math.floor(b / (SIZE / BOX_W));
    const boxCol = b % (SIZE / BOX_W);
    const cells = [];
    for (let dr = 0; dr < BOX_H; dr++) {
      for (let dc = 0; dc < BOX_W; dc++) {
        const r = boxRow * BOX_H + dr;
        const c = boxCol * BOX_W + dc;
        cells.push(r * SIZE + c);
      }
    }
    return cells;
  }

  function peers(i) {
    const set = new Set([...cellsInRow(rowOf(i)), ...cellsInCol(colOf(i)), ...cellsInBox(boxOf(i))]);
    set.delete(i);
    return [...set];
  }

  // is `val` allowed at cell `i`, given the rest of the grid (ignoring i's own current value)
  function isValidPlacement(grid, i, val) {
    for (const p of peers(i)) {
      if (grid[p] === val) return false;
    }
    return true;
  }

  function isComplete(grid) {
    return grid.every((v) => v !== 0);
  }

  // full validity check: every filled cell obeys row/col/box constraints
  function isValidGrid(grid) {
    for (let i = 0; i < grid.length; i++) {
      if (grid[i] === 0) continue;
      const val = grid[i];
      for (const p of peers(i)) {
        if (p > i && grid[p] === val) return false;
      }
    }
    return true;
  }

  function candidates(grid, i) {
    if (grid[i] !== 0) return [];
    const used = new Set(peers(i).map((p) => grid[p]).filter((v) => v !== 0));
    const out = [];
    for (let v = 1; v <= SIZE; v++) if (!used.has(v)) out.push(v);
    return out;
  }

  const GameLogic = {
    SIZE, BOX_H, BOX_W,
    rowOf, colOf, boxOf, cellsInRow, cellsInCol, cellsInBox, peers,
    isValidPlacement, isComplete, isValidGrid, candidates,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = GameLogic;
  } else {
    root.GameLogic = GameLogic;
  }
})(typeof window !== "undefined" ? window : globalThis);
