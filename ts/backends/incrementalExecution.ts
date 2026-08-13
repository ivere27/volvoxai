import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import type { BackendExecutionOptions } from './BackendEngine.js';

interface IncrementalExecutionOptions extends BackendExecutionOptions {}

/* Compute the dependency closure for opt-in inference-only intermediate
 * caching. Executor tensor storage already survives execute() calls; selected
 * nodes are the consumers of changed graph inputs and all of their descendants. */
export function incrementalNodeSelection(
  graph: RuntimeGraph,
  inputs: Record<string, unknown>,
  options: IncrementalExecutionOptions,
  cacheValid: boolean,
): Set<number> | null {
  if (options?.incremental !== true || options.incrementalReset === true || cacheValid !== true) {
    return null;
  }
  if (options.changedInputs != null && !Array.isArray(options.changedInputs)) {
    throw new Error('Incremental changedInputs must be an array of graph-input names.');
  }
  const changed = options.changedInputs ?? Object.keys(inputs || {});
  for (const name of changed) {
    if (typeof name !== 'string' || name.length === 0) {
      throw new Error('Incremental changedInputs must contain non-empty graph-input names.');
    }
    const tensor = graph?.tensors?.get(name);
    if (!tensor?.isInput) {
      throw new Error(`Incremental changed input '${name}' is not a graph input.`);
    }
  }
  const dirty = new Set<string>(changed);
  const selected = new Set<number>();
  for (let index = 0; index < graph.nodes.length; index++) {
    const node = graph.nodes[index];
    if (!Object.values(node.inputs || {}).some((value) => value && dirty.has(value.name))) continue;
    selected.add(index);
    for (const value of Object.values(node.outputs || {})) {
      if (value?.name) dirty.add(value.name);
    }
  }
  return selected;
}

export function incrementalExecutionEnabled(
  options: IncrementalExecutionOptions,
  adapterPlan: unknown = null,
) {
  if (options?.incremental !== true) return false;
  if (adapterPlan != null) {
    throw new Error('Incremental execution is unavailable while an adapter is active.');
  }
  return true;
}
