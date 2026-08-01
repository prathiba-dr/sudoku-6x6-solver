/*
 * Trains the MLP defined in docs/neural-net.js via hand-rolled backprop +
 * Adam - no ML framework, matching the "no dependencies" spirit of the
 * rest of this project (and the tic-tac-toe-ai project it's a sibling to).
 *
 * Loss: cross-entropy per cell (36 independent 6-way softmax heads,
 * sharing one hidden layer), summed per sample.
 */
const fs = require("fs");
const path = require("path");
const { forward, encodeGrid, INPUT_DIM, N_CELLS, N_CLASSES } = require("../docs/neural-net.js");
const { generateDataset, mulberry32 } = require("./generate_puzzles.js");

const HIDDEN_DIM = 128;
const OUTPUT_DIM = N_CELLS * N_CLASSES;
const LR = 0.003;
const BATCH_SIZE = 32;
const EPOCHS = 40;
const TRAIN_SAMPLES = 20000;
const TEST_SAMPLES = 2000;
const TARGET_CLUES = 14; // out of 36 cells - a moderately hard puzzle

function randInit(rand, fanIn, fanOut, size) {
  const scale = Math.sqrt(2 / fanIn); // He init, since the hidden layer is ReLU
  return Float32Array.from({ length: size }, () => (rand() * 2 - 1) * scale);
}

function initNetwork(rand) {
  return {
    hiddenDim: HIDDEN_DIM,
    W1: randInit(rand, INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM * INPUT_DIM),
    b1: new Float32Array(HIDDEN_DIM),
    W2: randInit(rand, HIDDEN_DIM, OUTPUT_DIM, OUTPUT_DIM * HIDDEN_DIM),
    b2: new Float32Array(OUTPUT_DIM),
  };
}

function initAdamState(net) {
  const zerosLike = (arr) => new Float32Array(arr.length);
  return {
    mW1: zerosLike(net.W1), vW1: zerosLike(net.W1),
    mb1: zerosLike(net.b1), vb1: zerosLike(net.b1),
    mW2: zerosLike(net.W2), vW2: zerosLike(net.W2),
    mb2: zerosLike(net.b2), vb2: zerosLike(net.b2),
    t: 0,
  };
}

function adamUpdate(param, grad, m, v, t, lr) {
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  for (let i = 0; i < param.length; i++) {
    m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
    const mHat = m[i] / (1 - Math.pow(beta1, t));
    const vHat = v[i] / (1 - Math.pow(beta2, t));
    param[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
  }
}

// one sample's forward + backward pass, accumulating gradients into the given buffers
function backwardSample(net, input, targetGrid, grads) {
  const { a1, z1, probs } = forward(net, input);

  const dz2 = new Float32Array(OUTPUT_DIM);
  let loss = 0;
  for (let cell = 0; cell < N_CELLS; cell++) {
    const trueClass = targetGrid[cell] - 1; // 0-indexed
    for (let k = 0; k < N_CLASSES; k++) {
      const p = probs[cell * N_CLASSES + k];
      const target = k === trueClass ? 1 : 0;
      dz2[cell * N_CLASSES + k] = p - target;
      if (target === 1) loss -= Math.log(Math.max(p, 1e-9));
    }
  }

  // dL/dW2, dL/db2, and da1
  const da1 = new Float32Array(net.hiddenDim);
  for (let o = 0; o < OUTPUT_DIM; o++) {
    const d = dz2[o];
    grads.gb2[o] += d;
    const base = o * net.hiddenDim;
    for (let h = 0; h < net.hiddenDim; h++) {
      grads.gW2[base + h] += d * a1[h];
      da1[h] += d * net.W2[base + h];
    }
  }

  // ReLU backward
  const dz1 = new Float32Array(net.hiddenDim);
  for (let h = 0; h < net.hiddenDim; h++) dz1[h] = z1[h] > 0 ? da1[h] : 0;

  for (let h = 0; h < net.hiddenDim; h++) {
    const d = dz1[h];
    grads.gb1[h] += d;
    const base = h * INPUT_DIM;
    for (let i = 0; i < INPUT_DIM; i++) {
      if (input[i] !== 0) grads.gW1[base + i] += d * input[i]; // input is one-hot/sparse
    }
  }

  return loss;
}

function zeroGrads(net) {
  return {
    gW1: new Float32Array(net.W1.length), gb1: new Float32Array(net.b1.length),
    gW2: new Float32Array(net.W2.length), gb2: new Float32Array(net.b2.length),
  };
}

function evaluate(net, samples) {
  let correctCells = 0, totalCells = 0, exactGrids = 0;
  for (const { puzzle, solved } of samples) {
    const input = encodeGrid(puzzle);
    const { probs } = forward(net, input);
    let allCorrect = true;
    for (let cell = 0; cell < N_CELLS; cell++) {
      let best = 0, bestP = -1;
      for (let k = 0; k < N_CLASSES; k++) {
        const p = probs[cell * N_CLASSES + k];
        if (p > bestP) { bestP = p; best = k; }
      }
      const pred = best + 1;
      totalCells++;
      if (pred === solved[cell]) correctCells++;
      else allCorrect = false;
    }
    if (allCorrect) exactGrids++;
  }
  return {
    cellAccuracy: correctCells / totalCells,
    exactGridAccuracy: exactGrids / samples.length,
  };
}

function main() {
  const rand = mulberry32(1234);
  console.log(`Generating ${TRAIN_SAMPLES} training + ${TEST_SAMPLES} test samples (${TARGET_CLUES} clues each)...`);
  const t0 = Date.now();
  const trainSet = generateDataset(TRAIN_SAMPLES, 1234, TARGET_CLUES);
  const testSet = generateDataset(TEST_SAMPLES, 5678, TARGET_CLUES);
  console.log(`Done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  const net = initNetwork(rand);
  const adam = initAdamState(net);

  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    const order = trainSet.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }

    let epochLoss = 0;
    for (let b = 0; b < order.length; b += BATCH_SIZE) {
      const batchIdx = order.slice(b, b + BATCH_SIZE);
      const grads = zeroGrads(net);
      for (const idx of batchIdx) {
        const { puzzle, solved } = trainSet[idx];
        const input = encodeGrid(puzzle);
        epochLoss += backwardSample(net, input, solved, grads);
      }
      const n = batchIdx.length;
      for (let i = 0; i < grads.gW1.length; i++) grads.gW1[i] /= n;
      for (let i = 0; i < grads.gb1.length; i++) grads.gb1[i] /= n;
      for (let i = 0; i < grads.gW2.length; i++) grads.gW2[i] /= n;
      for (let i = 0; i < grads.gb2.length; i++) grads.gb2[i] /= n;

      adam.t++;
      adamUpdate(net.W1, grads.gW1, adam.mW1, adam.vW1, adam.t, LR);
      adamUpdate(net.b1, grads.gb1, adam.mb1, adam.vb1, adam.t, LR);
      adamUpdate(net.W2, grads.gW2, adam.mW2, adam.vW2, adam.t, LR);
      adamUpdate(net.b2, grads.gb2, adam.mb2, adam.vb2, adam.t, LR);
    }

    if (epoch % 5 === 0 || epoch === 1) {
      const evalResult = evaluate(net, testSet.slice(0, 500));
      console.log(`epoch ${epoch}/${EPOCHS} - loss ${(epochLoss / trainSet.length).toFixed(3)} `
        + `- test cell_acc ${(evalResult.cellAccuracy * 100).toFixed(1)}% `
        + `exact_grid_acc ${(evalResult.exactGridAccuracy * 100).toFixed(1)}%`);
    }
  }

  const finalEval = evaluate(net, testSet);
  console.log("\nFinal evaluation on full held-out test set:");
  console.log(finalEval);

  const outPath = path.join(__dirname, "..", "docs", "nn_weights.json");
  fs.writeFileSync(outPath, JSON.stringify({
    hiddenDim: net.hiddenDim,
    W1: Array.from(net.W1), b1: Array.from(net.b1),
    W2: Array.from(net.W2), b2: Array.from(net.b2),
  }));
  console.log(`Saved weights to ${outPath} (${(fs.statSync(outPath).size / 1024).toFixed(0)} KB)`);

  fs.writeFileSync(
    path.join(__dirname, "..", "reports", "nn_training_results.json"),
    JSON.stringify({ targetClues: TARGET_CLUES, trainSamples: TRAIN_SAMPLES, testSamples: TEST_SAMPLES, ...finalEval }, null, 2)
  );
}

main();
