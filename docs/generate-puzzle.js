// Generates valid, solved 6x6 grids (via a base Latin-square pattern
// transformed by the standard Sudoku-preserving shuffles: digit
// relabeling, row swaps within a band, column swaps within a stack, and
// whole band/stack swaps), then carves puzzles out of them by removing
// cells while checking the puzzle still has a UNIQUE solution.
//
// Dual-mode (Node + browser) like the other docs/*.js files, so the
// training script's dataset generation and the browser's "New Puzzle"
// button use the exact same generator - not two implementations that
// could quietly diverge.
(function (root) {
  const GL = typeof module !== "undefined" && module.exports
    ? require("./game-logic.js")
    : window.GameLogic;
  const { countSolutions } = typeof module !== "undefined" && module.exports
    ? require("./backtracking-solver.js")
    : window.BacktrackingSolver;

  const { SIZE, BOX_H, BOX_W } = GL;
  const N_BANDS = SIZE / BOX_H; // 3 bands of 2 rows
  const N_STACKS = SIZE / BOX_W; // 2 stacks of 3 cols

  function shuffle(arr, rand) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function baseGrid() {
    const grid = new Array(SIZE * SIZE);
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        grid[r * SIZE + c] = ((r * BOX_W + Math.floor(r / BOX_H) + c) % SIZE) + 1;
      }
    }
    return grid;
  }

  function randomSolvedGrid(rand) {
    const grid = baseGrid();

    // permute rows within each band (keeps box-row membership intact)
    const rowOrder = [];
    for (let band = 0; band < N_BANDS; band++) {
      const rowsInBand = shuffle(Array.from({ length: BOX_H }, (_, i) => band * BOX_H + i), rand);
      rowOrder.push(...rowsInBand);
    }
    // permute columns within each stack
    const colOrder = [];
    for (let stack = 0; stack < N_STACKS; stack++) {
      const colsInStack = shuffle(Array.from({ length: BOX_W }, (_, i) => stack * BOX_W + i), rand);
      colOrder.push(...colsInStack);
    }
    // permute bands and stacks themselves
    const bandOrder = shuffle(Array.from({ length: N_BANDS }, (_, i) => i), rand);
    const stackOrder = shuffle(Array.from({ length: N_STACKS }, (_, i) => i), rand);

    const finalRowOrder = [];
    for (const band of bandOrder) {
      for (let i = 0; i < BOX_H; i++) finalRowOrder.push(rowOrder[band * BOX_H + i]);
    }
    const finalColOrder = [];
    for (const stack of stackOrder) {
      for (let i = 0; i < BOX_W; i++) finalColOrder.push(colOrder[stack * BOX_W + i]);
    }

    const shuffled = new Array(SIZE * SIZE);
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        shuffled[r * SIZE + c] = grid[finalRowOrder[r] * SIZE + finalColOrder[c]];
      }
    }

    // relabel digits 1-6 randomly
    const digitMap = shuffle([1, 2, 3, 4, 5, 6], rand);
    return shuffled.map((v) => digitMap[v - 1]);
  }

  // removes cells one at a time (random order), keeping a removal only if
  // the puzzle still has exactly one solution - guarantees every generated
  // puzzle is uniquely solvable, not just "a solver can find a solution"
  function carvePuzzle(solved, rand, targetClues) {
    const puzzle = solved.slice();
    const order = shuffle(Array.from({ length: SIZE * SIZE }, (_, i) => i), rand);
    let clues = SIZE * SIZE;

    for (const i of order) {
      if (clues <= targetClues) break;
      const backup = puzzle[i];
      puzzle[i] = 0;
      if (countSolutions(puzzle, 2) === 1) {
        clues--;
      } else {
        puzzle[i] = backup;
      }
    }
    return puzzle;
  }

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function generateDataset(n, seed, targetClues = 14) {
    const rand = mulberry32(seed);
    const samples = [];
    for (let i = 0; i < n; i++) {
      const solved = randomSolvedGrid(rand);
      const puzzle = carvePuzzle(solved, rand, targetClues);
      samples.push({ puzzle, solved });
    }
    return samples;
  }

  function generatePuzzle(targetClues = 14) {
    const rand = mulberry32((Date.now() * Math.random()) | 0);
    const solved = randomSolvedGrid(rand);
    const puzzle = carvePuzzle(solved, rand, targetClues);
    return { puzzle, solved };
  }

  const PuzzleGenerator = { randomSolvedGrid, carvePuzzle, generateDataset, generatePuzzle, mulberry32 };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = PuzzleGenerator;
  } else {
    root.PuzzleGenerator = PuzzleGenerator;
  }
})(typeof window !== "undefined" ? window : globalThis);
