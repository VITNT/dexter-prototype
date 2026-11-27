// ==========================================
// THE AI FACTORY SCRIPT
// ==========================================

// --- 1. FORWARD FILL (Data Rescue) ---
function sanitizeData(frames) {
  if (!frames || frames.length === 0) return frames;

  const frameLength = frames[0].length;

  for (let i = 1; i < frames.length; i++) {
    if (frameLength === 63) {
      if (frames[i][0] === 0 && frames[i][1] === 0) {
        frames[i] = [...frames[i - 1]];
      }
    } else if (frameLength === 225) {
      if (frames[i][99] === 0 && frames[i][100] === 0) {
        for (let j = 99; j <= 161; j++) {
          frames[i][j] = frames[i - 1][j];
        }
      }
      if (frames[i][162] === 0 && frames[i][163] === 0) {
        for (let j = 162; j <= 224; j++) {
          frames[i][j] = frames[i - 1][j];
        }
      }
    }
  }
  return frames;
}

// --- 2. BODY-CENTRIC NORMALIZATION ---
// NOTE: This exact transform is also applied in recorder.js at record-time.
// Keeping both in sync is critical — changing one requires changing the other.
function normalizeFrames(frames) {
  let normalizedFrames = [];

  for (let frame of frames) {
    let anchorX = frame[0];
    let anchorY = frame[1];

    if (anchorX === 0 && anchorY === 0) {
      anchorX = frame[99];
      anchorY = frame[100];
    }
    if (anchorX === 0 && anchorY === 0) {
      anchorX = frame[162];
      anchorY = frame[163];
    }

    let newFrame = [];
    for (let i = 0; i < frame.length; i += 3) {
      let x = frame[i];
      let y = frame[i + 1];
      let z = frame[i + 2];

      if (x === 0 && y === 0) {
        newFrame.push(0, 0, 0);
      } else {
        newFrame.push(x - anchorX, y - anchorY, z);
      }
    }
    normalizedFrames.push(newFrame);
  }
  return normalizedFrames;
}

// ==========================================
// DATA AUGMENTATION PIPELINE
// ==========================================

// Time warp: interpolate or skip frames to change playback speed
function timeWarp(frames, speedFactor) {
  const originalLen = frames.length;
  const result = [];
  for (let i = 0; i < originalLen; i++) {
    // Sample from the original sequence at a different rate
    const srcIdx = Math.min(Math.floor(i * speedFactor), originalLen - 1);
    const nextIdx = Math.min(srcIdx + 1, originalLen - 1);
    const t = i * speedFactor - Math.floor(i * speedFactor);
    // Linear interpolation between frames
    const interp = frames[srcIdx].map(
      (v, j) => v * (1 - t) + frames[nextIdx][j] * t,
    );
    result.push(interp);
  }
  return result;
}

// Spatial jitter: add small Gaussian noise to all landmarks
function spatialJitter(frames, sigma = 0.005) {
  return frames.map((frame) =>
    frame.map((v, i) => {
      // Don't jitter zero-filled (missing) landmarks
      if (i % 3 !== 2 && v === 0) return v; // skip x/y that are 0
      // Simple Box-Muller Gaussian approximation
      const u1 = Math.random(),
        u2 = Math.random();
      const noise =
        sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      return v + noise;
    }),
  );
}

// Pose landmark left/right pairs (MediaPipe Pose topology). When mirroring holistic
// (225-point) data, these must be SWAPPED after negating x — otherwise a mirrored
// "left elbow" point just ends up as a flipped-x point still labeled "left elbow",
// which is anatomically impossible and just injects noise into training.
const POSE_LR_PAIRS = [
  [1, 4],
  [2, 5],
  [3, 6],
  [7, 8],
  [9, 10],
  [11, 12],
  [13, 14],
  [15, 16],
  [17, 18],
  [19, 20],
  [21, 22],
  [23, 24],
  [25, 26],
  [27, 28],
  [29, 30],
  [31, 32],
];

// Horizontal mirror: flip x-coordinates relative to center (0.5 in raw, 0 in normalized)
function mirrorFrames(frames) {
  const frameLength = frames[0].length;
  return frames.map((frame) => {
    const mirrored = [...frame];

    // Negate x for every point (in normalized space, center is 0)
    for (let i = 0; i < frameLength; i += 3) {
      mirrored[i] = -frame[i];
    }

    if (frameLength === 225) {
      // Holistic layout: pose(0-98) + leftHand(99-161) + rightHand(162-224).
      // A mirrored right-handed signer looks left-handed, so the hand DATA
      // must move into the other hand's slot, not just flip x in place.
      const leftHand = mirrored.slice(99, 162);
      const rightHand = mirrored.slice(162, 225);
      for (let i = 0; i < 63; i++) {
        mirrored[99 + i] = rightHand[i];
        mirrored[162 + i] = leftHand[i];
      }

      // Swap paired left/right pose landmarks (shoulders, elbows, hips, etc.)
      for (const [a, b] of POSE_LR_PAIRS) {
        for (let k = 0; k < 3; k++) {
          const tmp = mirrored[a * 3 + k];
          mirrored[a * 3 + k] = mirrored[b * 3 + k];
          mirrored[b * 3 + k] = tmp;
        }
      }
    }
    // frameLength === 63 (single hand only): x-negation alone is correct —
    // there's only one hand slot and no left/right pose skeleton to swap.

    return mirrored;
  });
}

// Generate all augmented variants of a single sample
function augmentSample(frames, label, labelIndex, numClasses) {
  const augmented = [];

  function makeEntry(f) {
    const oneHot = new Array(numClasses).fill(0);
    oneHot[labelIndex] = 1;
    augmented.push({ feature: f, label: oneHot });
  }

  // Original (already normalized)
  makeEntry(frames);

  // Speed variation: 0.8x (slower) and 1.2x (faster)
  makeEntry(timeWarp(frames, 0.8));
  makeEntry(timeWarp(frames, 1.2));

  // Spatial jitter (two different noise seeds)
  makeEntry(spatialJitter(frames, 0.004));
  makeEntry(spatialJitter(frames, 0.007));

  // Horizontal mirror
  const mirrored = mirrorFrames(frames);
  makeEntry(mirrored);

  // Mirror + jitter
  makeEntry(spatialJitter(mirrored, 0.004));

  return augmented;
}

// ==========================================
// TRAINING PIPELINE
// ==========================================
async function startTraining() {
  const fileInput = document.getElementById("fileInput");
  const statusText = document.getElementById("status");

  // --- Sub-model naming: look for an optional #subModelName text input. ---
  // If present and non-blank, exported files are prefixed with this name and
  // registry-compatible filenames are used. Falls back to default naming otherwise.
  const subModelNameInput = document.getElementById("subModelName");
  const subModelName = subModelNameInput ? subModelNameInput.value.trim() : "";
  const usingSubModel = subModelName.length > 0;

  if (fileInput.files.length === 0) {
    alert("Please upload your JSON files first!");
    return;
  }

  statusText.innerText =
    "Step 1: Reading, sanitizing, normalizing, and augmenting files...";
  // Yield to browser to let UI update before heavy work
  await new Promise((r) => setTimeout(r, 50));

  let rawData = [];
  for (let file of fileInput.files) {
    const text = await file.text();
    rawData.push(JSON.parse(text));
  }

  const uniqueLabels = [...new Set(rawData.map((item) => item.label))].sort();
  console.log("Found classes:", uniqueLabels);

  // --- Per-class raw recording counts (imbalance visibility + class weighting) ---
  const rawCountsByLabel = {};
  uniqueLabels.forEach((l) => (rawCountsByLabel[l] = 0));
  rawData.forEach((item) => rawCountsByLabel[item.label]++);
  console.log("Raw recordings per class:", rawCountsByLabel);

  const counts = Object.values(rawCountsByLabel);
  const maxCount = Math.max(...counts),
    minCount = Math.min(...counts);
  if (maxCount > minCount * 3) {
    const maxLabel = Object.keys(rawCountsByLabel).find(
      (k) => rawCountsByLabel[k] === maxCount,
    );
    const minLabel = Object.keys(rawCountsByLabel).find(
      (k) => rawCountsByLabel[k] === minCount,
    );
    console.warn(
      `⚠️ Class imbalance: "${maxLabel}" has ${maxCount} recordings vs "${minLabel}" with only ${minCount}. Sample weighting below will help, but recording more of the underrepresented classes is the real fix.`,
    );
  }

  statusText.innerText = `Found ${uniqueLabels.length} classes: ${uniqueLabels.join(", ")}. Splitting train/val by raw recording (before augmentation) to avoid leakage...`;
  await new Promise((r) => setTimeout(r, 50));

  // --- Split RAW recordings into train/val FIRST, stratified per class ---
  // Augmenting before splitting would let near-duplicate augmented variants (jitter/
  // mirror/time-warp) of the SAME original recording land on both sides of the split,
  // so validation accuracy quietly measures memorization instead of generalization.
  const rawByLabel = {};
  uniqueLabels.forEach((l) => (rawByLabel[l] = []));
  rawData.forEach((item) => rawByLabel[item.label].push(item));

  const trainRaw = [];
  const valRaw = [];
  for (const label of uniqueLabels) {
    const items = rawByLabel[label];
    // Shuffle within this class only
    for (let i = items.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [items[i], items[j]] = [items[j], items[i]];
    }
    const splitIdx = Math.max(1, Math.floor(items.length * 0.8));
    trainRaw.push(...items.slice(0, splitIdx));
    if (items.length > 1) {
      valRaw.push(...items.slice(splitIdx));
    } else {
      console.warn(
        `Only 1 recording found for class "${label}" — none can be held out for validation. Record at least 2 (ideally more) per class.`,
      );
    }
  }

  statusText.innerText = `Train recordings: ${trainRaw.length}, Val recordings: ${valRaw.length}. Augmenting...`;
  await new Promise((r) => setTimeout(r, 50));

  function buildSamples(items) {
    const samples = [];
    items.forEach((item) => {
      const cleanedFrames = sanitizeData(item.frames);
      const normalizedFrames = normalizeFrames(cleanedFrames);
      const labelIndex = uniqueLabels.indexOf(item.label);
      const augmented = augmentSample(
        normalizedFrames,
        item.label,
        labelIndex,
        uniqueLabels.length,
      );
      samples.push(...augmented);
    });
    return samples;
  }

  const trainSamples = buildSamples(trainRaw);
  const valSamples = buildSamples(valRaw);

  console.log(
    `Augmented — Train: ${trainSamples.length} (from ${trainRaw.length} originals) | Val: ${valSamples.length} (from ${valRaw.length} originals)`,
  );
  statusText.innerText = `Augmented: train=${trainSamples.length}, val=${valSamples.length}. Shuffling...`;
  await new Promise((r) => setTimeout(r, 50));

  // Shuffle each split independently — order only, splits never mix
  function shuffleInPlace(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
  shuffleInPlace(trainSamples);
  shuffleInPlace(valSamples);

  if (valSamples.length === 0) {
    alert(
      "No validation samples available — every class only had 1 raw recording. Record at least 2 samples per class so training can measure real accuracy.",
    );
  }

  console.log(`Train: ${trainSamples.length} | Val: ${valSamples.length}`);

  // --- Class weighting: counter imbalance with inverse-frequency per-sample weights ---
  // Computed from raw recording counts (augmentation multiplies every class by the
  // same factor, so the ratio is preserved) and applied as a per-sample weight tensor.
  const numClasses = uniqueLabels.length;
  const totalRaw = rawData.length;
  const classWeightMap = {};
  uniqueLabels.forEach((label, idx) => {
    const count = rawCountsByLabel[label] || 1;
    classWeightMap[idx] = totalRaw / (numClasses * count);
  });
  console.log("Per-class sample weights (inverse frequency):", classWeightMap);

  const trainSampleWeights = trainSamples.map(
    (s) => classWeightMap[s.label.indexOf(1)],
  );
  const sampleWeightTensor = tf.tensor1d(trainSampleWeights);

  const xsTrain = tf.tensor3d(trainSamples.map((s) => s.feature));
  const ysTrain = tf.tensor2d(trainSamples.map((s) => s.label));
  const xsVal =
    valSamples.length > 0
      ? tf.tensor3d(valSamples.map((s) => s.feature))
      : null;
  const ysVal =
    valSamples.length > 0 ? tf.tensor2d(valSamples.map((s) => s.label)) : null;

  const dataWidth = trainSamples[0].feature[0].length;
  console.log(`Auto-detected data width: ${dataWidth} points per frame.`);

  statusText.innerText = `Step 2: Building Deep LSTM Architecture (${dataWidth}-point mode, ${trainSamples.length + valSamples.length} samples)...`;
  await new Promise((r) => setTimeout(r, 50));

  // --- Deeper stacked LSTM model ---
  // Regularization tightened up (L2 weight decay + higher dropout) since small/
  // single-session datasets are easy for a model this size to just memorize.
  const model = tf.sequential();

  // Layer 1: LSTM returning sequences so layer 2 can process them
  model.add(
    tf.layers.lstm({
      units: 96,
      returnSequences: true,
      inputShape: [30, dataWidth],
      recurrentDropout: 0.15,
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }),
  );

  model.add(tf.layers.dropout({ rate: 0.4 }));

  // Layer 2: Second LSTM
  model.add(
    tf.layers.lstm({
      units: 48,
      returnSequences: false,
      recurrentDropout: 0.15,
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }),
  );

  model.add(tf.layers.dropout({ rate: 0.4 }));

  // Dense head
  model.add(
    tf.layers.dense({
      units: 48,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }),
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }),
  );
  model.add(
    tf.layers.dense({ units: uniqueLabels.length, activation: "softmax" }),
  );

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  model.summary();

  statusText.innerText = `Step 3: Training... train=${trainSamples.length}, val=${valSamples.length}. Check F12!`;
  await new Promise((r) => setTimeout(r, 50));

  let bestValAcc = 0;
  let bestValLoss = Infinity;
  let bestWeights = null;
  let epochsSinceImprovement = 0;
  const PATIENCE = 12; // stop if val_loss hasn't improved in this many epochs
  let stopRequested = false;

  await model.fit(xsTrain, ysTrain, {
    epochs: 80,
    batchSize: 16,
    validationData: xsVal ? [xsVal, ysVal] : undefined,
    sampleWeight: sampleWeightTensor,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const acc = (logs.acc * 100).toFixed(1);
        const hasVal = logs.val_acc !== undefined;
        const valAcc = hasVal ? (logs.val_acc * 100).toFixed(1) + "%" : "N/A";
        if (hasVal && logs.val_acc > bestValAcc) bestValAcc = logs.val_acc;

        // --- Manual early stopping + best-weights checkpoint ---
        // Watching val_loss (not val_acc) since accuracy pins at 100% long before
        // loss stops improving on a small/homogeneous dataset — loss is the more
        // sensitive signal for catching the point where it starts overfitting.
        if (hasVal) {
          if (logs.val_loss < bestValLoss - 1e-4) {
            bestValLoss = logs.val_loss;
            epochsSinceImprovement = 0;
            // Snapshot current weights as the best-so-far
            if (bestWeights) bestWeights.forEach((t) => t.dispose());
            bestWeights = model.getWeights().map((t) => t.clone());
          } else {
            epochsSinceImprovement++;
          }
          if (epochsSinceImprovement >= PATIENCE) {
            stopRequested = true;
            model.stopTraining = true;
          }
        }

        console.log(
          `Epoch ${epoch + 1}/80 | Loss: ${logs.loss.toFixed(4)} | Acc: ${acc}% | Val Acc: ${valAcc} | Val Loss: ${hasVal ? logs.val_loss.toFixed(4) : "N/A"}`,
        );
        statusText.innerText = stopRequested
          ? `Early stopping (val loss stalled) — restoring best epoch...`
          : `Epoch ${epoch + 1}/80 | Acc: ${acc}% | Val: ${valAcc} | Best Val Acc: ${(bestValAcc * 100).toFixed(1)}%`;
        // Yield so the UI can refresh
        await new Promise((r) => setTimeout(r, 0));
      },
    },
  });

  // Restore the best-validation-loss checkpoint instead of whatever the last epoch left us with
  if (bestWeights) {
    model.setWeights(bestWeights);
    bestWeights.forEach((t) => t.dispose());
    statusText.innerText = `Restored best checkpoint (val loss: ${bestValLoss.toFixed(4)}). Finalizing...`;
    await new Promise((r) => setTimeout(r, 50));
  } else if (!xsVal) {
    console.warn(
      "No validation set — couldn't checkpoint on val_loss, using final-epoch weights as-is. Results are unverified.",
    );
  }

  // Cleanup tensors
  xsTrain.dispose();
  ysTrain.dispose();
  if (xsVal) {
    xsVal.dispose();
    ysVal.dispose();
  }
  sampleWeightTensor.dispose();

  statusText.innerText = `Step 4: Training Complete! Best Val Acc: ${(bestValAcc * 100).toFixed(1)}%. Downloading files...`;

  // --- Dynamic export naming ---
  // {subModelName}_model.json / {subModelName}_model.weights.bin / {subModelName}_labels.json
  // when a sub-model name was provided; otherwise fall back to the default ksl_model naming.
  const modelExportName = usingSubModel ? `${subModelName}_model` : "ksl_model";
  const labelsExportName = usingSubModel
    ? `${subModelName}_labels.json`
    : "ksl_labels.json";

  // 1. Save model files (TFJS writes ${modelExportName}.json + ${modelExportName}.weights.bin)
  await model.save(`downloads://${modelExportName}`);

  // Wait to bypass browser download spam blocker
  await new Promise((r) => setTimeout(r, 1000));

  // 2. Save labels
  const blob = new Blob([JSON.stringify(uniqueLabels)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = labelsExportName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  if (usingSubModel) {
    statusText.innerText =
      `✅ Done! Sub-model "${subModelName}" — ${uniqueLabels.length} classes, ` +
      `best val accuracy: ${(bestValAcc * 100).toFixed(1)}%. ` +
      `Add "${subModelName}": { "model": "./${modelExportName}.json", "labels": "./${labelsExportName}" } ` +
      `to submodel-registry.json under the matching handshape_zone key to activate it.`;
  } else {
    statusText.innerText = `✅ Done! ${uniqueLabels.length} classes, best val accuracy: ${(bestValAcc * 100).toFixed(1)}%`;
  }
}
