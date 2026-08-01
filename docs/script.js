(function () {
  const GL = window.GameLogic;
  const { solve } = window.BacktrackingSolver;
  const { generatePuzzle } = window.PuzzleGenerator;
  const { forward, encodeGrid, N_CELLS, N_CLASSES } = window.NeuralNet;

  const TARGET_CLUES = 14; // matches what the shipped nn_weights.json was trained on

  const boardEl = document.getElementById("board");
  const statusEl = document.getElementById("status");
  const newPuzzleBtn = document.getElementById("new-puzzle-btn");
  const solveBacktrackBtn = document.getElementById("solve-backtrack-btn");
  const solveNNBtn = document.getElementById("solve-nn-btn");
  const checkBtn = document.getElementById("check-btn");
  const clearBtn = document.getElementById("clear-btn");

  let puzzle = new Array(N_CELLS).fill(0);
  let solved = new Array(N_CELLS).fill(0);
  let net = null;

  const solveNNBtnDefaultText = solveNNBtn.textContent;
  solveNNBtn.disabled = true;
  solveNNBtn.textContent = "Solve (Neural Net) - loading model...";
  fetch("nn_weights.json")
    .then((r) => r.json())
    .then((raw) => {
      net = {
        hiddenDim: raw.hiddenDim,
        W1: Float32Array.from(raw.W1), b1: Float32Array.from(raw.b1),
        W2: Float32Array.from(raw.W2), b2: Float32Array.from(raw.b2),
      };
      solveNNBtn.disabled = false;
      solveNNBtn.textContent = solveNNBtnDefaultText;
    })
    .catch((err) => {
      solveNNBtn.textContent = "Solve (Neural Net) - failed to load";
      console.error(err);
    });

  function nnPredict(grid) {
    const input = encodeGrid(grid);
    const { probs } = forward(net, input);
    const out = new Array(N_CELLS);
    for (let cell = 0; cell < N_CELLS; cell++) {
      let best = 0, bestP = -1;
      for (let k = 0; k < N_CLASSES; k++) {
        const p = probs[cell * N_CLASSES + k];
        if (p > bestP) { bestP = p; best = k; }
      }
      out[cell] = best + 1;
    }
    return out;
  }

  function buildBoardDOM() {
    boardEl.innerHTML = "";
    for (let i = 0; i < N_CELLS; i++) {
      const r = GL.rowOf(i), c = GL.colOf(i);
      const cell = document.createElement("div");
      cell.className = "cell";
      if ((c + 1) % GL.BOX_W === 0) cell.classList.add("box-right");
      if ((r + 1) % GL.BOX_H === 0) cell.classList.add("box-bottom");
      cell.dataset.index = i;
      boardEl.appendChild(cell);
    }
  }

  function renderPuzzle() {
    const cells = boardEl.querySelectorAll(".cell");
    cells.forEach((cell, i) => {
      cell.classList.remove("clue", "nn-correct", "nn-wrong");
      cell.innerHTML = "";
      if (puzzle[i] !== 0) {
        cell.classList.add("clue");
        cell.textContent = puzzle[i];
      } else {
        const input = document.createElement("input");
        input.type = "text";
        input.maxLength = 1;
        input.inputMode = "numeric";
        input.addEventListener("input", () => {
          input.value = input.value.replace(/[^1-6]/g, "").slice(0, 1);
        });
        cell.appendChild(input);
      }
    });
  }

  function readUserGrid() {
    const cells = boardEl.querySelectorAll(".cell");
    const grid = new Array(N_CELLS);
    cells.forEach((cell, i) => {
      if (puzzle[i] !== 0) { grid[i] = puzzle[i]; return; }
      const input = cell.querySelector("input");
      if (input) {
        grid[i] = input.value ? parseInt(input.value, 10) : 0;
      } else {
        // cell was auto-filled by a solver button (no <input> left, just text)
        const text = cell.textContent.trim();
        grid[i] = text ? parseInt(text, 10) : 0;
      }
    });
    return grid;
  }

  function fillNonClueCells(grid, { colorCorrectness } = {}) {
    const cells = boardEl.querySelectorAll(".cell");
    cells.forEach((cell, i) => {
      if (puzzle[i] !== 0) return;
      cell.innerHTML = "";
      cell.textContent = grid[i] || "";
      cell.classList.remove("nn-correct", "nn-wrong");
      if (colorCorrectness) {
        cell.classList.add(grid[i] === solved[i] ? "nn-correct" : "nn-wrong");
      }
    });
  }

  function newPuzzle() {
    statusEl.textContent = "Generating puzzle...";
    setTimeout(() => {
      const gen = generatePuzzle(TARGET_CLUES);
      puzzle = gen.puzzle;
      solved = gen.solved;
      renderPuzzle();
      statusEl.textContent = `New puzzle: ${TARGET_CLUES} clues, unique solution guaranteed.`;
    }, 10);
  }

  function solveWithBacktracking() {
    const result = solve(puzzle);
    if (!result) { statusEl.textContent = "No solution found (unexpected - please report)."; return; }
    fillNonClueCells(result);
    statusEl.textContent = "Solved via constraint propagation + backtracking - always exactly correct.";
  }

  function solveWithNN() {
    if (!net) return;
    const guess = nnPredict(puzzle);
    fillNonClueCells(guess, { colorCorrectness: true });

    let hiddenCorrect = 0, hiddenTotal = 0;
    for (let i = 0; i < N_CELLS; i++) {
      if (puzzle[i] !== 0) continue;
      hiddenTotal++;
      if (guess[i] === solved[i]) hiddenCorrect++;
    }
    const valid = GL.isValidGrid(guess) && GL.isComplete(guess);
    const exact = JSON.stringify(guess) === JSON.stringify(solved);
    statusEl.textContent = `Neural net guess: ${hiddenCorrect}/${hiddenTotal} hidden cells correct. `
      + `Valid grid? ${valid ? "Yes" : "No"}. Exactly correct? ${exact ? "Yes" : "No"}.`;
  }

  function checkAnswer() {
    const grid = readUserGrid();
    if (grid.some((v) => v === 0)) {
      statusEl.textContent = "Fill in every cell first.";
      return;
    }
    const valid = GL.isValidGrid(grid) && GL.isComplete(grid);
    const exact = JSON.stringify(grid) === JSON.stringify(solved);
    statusEl.textContent = exact
      ? "Correct! That's the unique solution."
      : valid
        ? "Valid grid, but not the solution to this puzzle - it satisfies the rules without being right."
        : "Not a valid grid - there's a row/column/box conflict somewhere.";
  }

  function clearEntries() {
    // re-render from scratch rather than just clearing <input> values,
    // since a solver button may have replaced inputs with plain text -
    // this resets back to the original blank, editable puzzle either way
    renderPuzzle();
    statusEl.textContent = "Cleared - back to the original puzzle.";
  }

  newPuzzleBtn.addEventListener("click", newPuzzle);
  solveBacktrackBtn.addEventListener("click", solveWithBacktracking);
  solveNNBtn.addEventListener("click", solveWithNN);
  checkBtn.addEventListener("click", checkAnswer);
  clearBtn.addEventListener("click", clearEntries);

  buildBoardDOM();
  newPuzzle();
})();
