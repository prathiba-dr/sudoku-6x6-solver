// A small hand-rolled MLP (no ML framework) that predicts a full 6x6
// solution in one forward pass, given a partially-filled grid. Forward
// pass only - this file is shared between training (which also needs
// backprop, implemented separately in training/train_nn.js) and the
// browser, which only ever needs inference.
//
// Input: 36 cells x 7-way one-hot (0=empty, 1-6=filled) = 252 dims.
// Hidden: one ReLU layer.
// Output: 36 cells x 6-way softmax (predicted digit 1-6 per cell) = 216 dims.
(function (root) {
  const { SIZE } = typeof module !== "undefined" && module.exports
    ? require("./game-logic.js")
    : window.GameLogic;

  const N_CELLS = SIZE * SIZE; // 36
  const N_CLASSES = SIZE;      // 6
  const INPUT_DIM = N_CELLS * (SIZE + 1); // 252 (7-way one-hot per cell)

  function encodeGrid(grid) {
    const input = new Float32Array(INPUT_DIM);
    for (let i = 0; i < N_CELLS; i++) {
      input[i * (SIZE + 1) + grid[i]] = 1; // grid[i] in [0,6], slot 0 = empty
    }
    return input;
  }

  function matVec(W, x, outDim, inDim) {
    // W is a flat Float32Array of length outDim*inDim, row-major
    const out = new Float32Array(outDim);
    for (let o = 0; o < outDim; o++) {
      let sum = 0;
      const base = o * inDim;
      for (let i = 0; i < inDim; i++) sum += W[base + i] * x[i];
      out[o] = sum;
    }
    return out;
  }

  function relu(x) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
    return out;
  }

  function softmaxInPlace(x, offset, len) {
    let max = -Infinity;
    for (let i = 0; i < len; i++) max = Math.max(max, x[offset + i]);
    let sum = 0;
    for (let i = 0; i < len; i++) {
      x[offset + i] = Math.exp(x[offset + i] - max);
      sum += x[offset + i];
    }
    for (let i = 0; i < len; i++) x[offset + i] /= sum;
  }

  // net = { W1 (hidden x input), b1 (hidden), W2 (output x hidden), b2 (output), hiddenDim }
  function forward(net, input) {
    const z1 = matVec(net.W1, input, net.hiddenDim, INPUT_DIM);
    for (let i = 0; i < net.hiddenDim; i++) z1[i] += net.b1[i];
    const a1 = relu(z1);

    const z2 = matVec(net.W2, a1, N_CELLS * N_CLASSES, net.hiddenDim);
    for (let i = 0; i < z2.length; i++) z2[i] += net.b2[i];

    const probs = z2.slice();
    for (let cell = 0; cell < N_CELLS; cell++) softmaxInPlace(probs, cell * N_CLASSES, N_CLASSES);

    return { z1, a1, z2, probs };
  }

  // returns { grid: predicted digits per cell (1-6), confidence: per-cell max prob }
  function predict(net, inputGrid) {
    const input = encodeGrid(inputGrid);
    const { probs } = forward(net, input);
    const grid = new Array(N_CELLS);
    const confidence = new Array(N_CELLS);
    for (let cell = 0; cell < N_CELLS; cell++) {
      let best = 0;
      let bestP = -1;
      for (let k = 0; k < N_CLASSES; k++) {
        const p = probs[cell * N_CLASSES + k];
        if (p > bestP) { bestP = p; best = k; }
      }
      grid[cell] = best + 1;
      confidence[cell] = bestP;
    }
    return { grid, confidence };
  }

  const NeuralNet = { INPUT_DIM, N_CELLS, N_CLASSES, encodeGrid, matVec, relu, softmaxInPlace, forward, predict };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = NeuralNet;
  } else {
    root.NeuralNet = NeuralNet;
  }
})(typeof window !== "undefined" ? window : globalThis);
