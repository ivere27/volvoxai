function product(values) {
  return values.reduce((a, b) => a * b, 1);
}

function firstOutput(node) {
  return node.outputs?.out || Object.values(node.outputs || {})[0];
}

function linearInput(node) {
  return node.inputs?.input || node.inputs?.x || node.inputs?.data;
}

/**
 * Applies a pinned LoRA route to the output of a CPU linear operation.
 *
 * The adapter manager owns route selection and immutable factor snapshots;
 * this portable kernel only performs the low-rank matrix update.
 */
export function _cpuLoRALinear(node, plan) {
  if (!plan) return;
  const input = linearInput(node);
  const output = firstOutput(node);
  if (!input || !output || !node.inputs?.weight) return;
  const rows = product(input.shape.slice(0, -1));
  const routeCount = plan.kind === "batch" ? plan.routes.length : 1;
  const batchSize = input.shape.length > 1 ? input.shape[0] : 1;
  if (routeCount !== 1 && routeCount !== batchSize) {
    throw new Error(`Execution requires one adapter route or ${batchSize} routes for this batch; received ${routeCount}.`);
  }
  if (rows % batchSize !== 0) {
    throw new Error(`Execution cannot divide ${rows} linear rows across batch size ${batchSize}.`);
  }
  const rowsPerRoute = routeCount === 1 ? rows : rows / batchSize;
  const scratch = new Map();
  for (let row = 0; row < rows; row++) {
    const route = plan.kind === "single" ? plan.route : plan.routes[Math.floor(row / rowsPerRoute)];
    if (!route) continue;
    const { snapshot } = route;
    const target = snapshot.targetsByWeight.get(node.inputs.weight.name);
    if (!target) continue;
    let z = scratch.get(target);
    if (!z) { z = new Float32Array(target.rank); scratch.set(target, z); }
    z.fill(0);
    for (let k = 0; k < target.din; k++) {
      const x = input.buffer[row * target.din + k];
      for (let r = 0; r < target.rank; r++) z[r] += x * target.A[k * target.rank + r];
    }
    for (let j = 0; j < target.dout; j++) {
      let delta = 0;
      for (let r = 0; r < target.rank; r++) {
        delta += z[r] * target.B[r * target.dout + j];
      }
      const index = row * target.dout + j;
      output.buffer[index] += route.scale * target.scale * delta;
    }
  }
}
