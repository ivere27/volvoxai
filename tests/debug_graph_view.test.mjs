import assert from 'node:assert/strict';
import test from 'node:test';
import {layoutDebugGraph} from '../examples/common/DebugGraphView.js';

const step = (scheduleIndex, inputs, outputs) => ({scheduleIndex, inputs, outputs});

test('graph edges follow tensor dataflow, including branches and shared inputs, rather than schedule adjacency', () => {
  const plan = {steps: [step(0, [0, 1], [2]), step(1, [2], [3]), step(2, [2], [4]), step(3, [3, 4, 1], [5])]};
  const graph = layoutDebugGraph(plan);
  assert.deepEqual(graph.edges.map(edge => [edge.from.step.scheduleIndex, edge.to.step.scheduleIndex, edge.tensorId]),
    [[0, 1, 2], [0, 2, 2], [1, 3, 3], [2, 3, 4]]);
  assert.equal(graph.nodes[1].y, graph.nodes[2].y, 'parallel branches share a level');
  assert.notEqual(graph.nodes[1].x, graph.nodes[2].x, 'parallel branches do not overlap');
  assert.ok(graph.nodes[3].y > graph.nodes[2].y);
});

test('reused tensor identifiers connect to the latest earlier producer without self edges or duplicate edges', () => {
  const graph = layoutDebugGraph({steps: [step(0, [0], [1]), step(1, [1, 1], [1]), step(2, [1], [2])]});
  assert.deepEqual(graph.edges.map(edge => [edge.from.step.scheduleIndex, edge.to.step.scheduleIndex]), [[0, 1], [1, 2]]);
  assert.deepEqual(layoutDebugGraph({steps: []}).nodes, []);
});
