import unittest

from examples.tts_common.compare_profiles import (
    aggregate_volvox, centered_normalization_regions, match_ort_regions, region_attribution, summarize_ort_events,
)


class ProfileComparisonTests(unittest.TestCase):
    def test_name_join_reports_fusion_loss_and_sums_paired_runs(self):
        rows = [
            {"node": "first", "op": "Mul", "provider": "CPUExecutionProvider", "samples_per_run_host_us": [1000, 4000]},
            {"node": "second", "op": "Add", "provider": "CPUExecutionProvider", "samples_per_run_host_us": [3000, 2000]},
            {"node": "fused_replacement", "op": "LayerNormalization", "provider": "CUDAExecutionProvider", "samples_per_run_host_us": [2000, 3000]},
        ]
        region, = match_ort_regions(rows, {"norm": {"first", "second", "renamed"}})
        self.assertEqual(region["unmatched_source_names"], ["renamed"])
        self.assertEqual(region["matched_source_names"], 2)
        provider, = region["providers"]
        self.assertEqual(provider["samples_per_run_host_ms"], [4, 6])
        self.assertEqual(provider["mean_host_ms"], 5)
        rows[1]["samples_per_run_host_us"].pop()
        with self.assertRaises(ValueError):
            match_ort_regions(rows, {"norm": {"first", "second"}})

    def test_centered_normalization_requires_connected_square_and_matching_axes(self):
        import onnx
        from onnx import TensorProto, helper
        nodes = [
            helper.make_node("ReduceMean", ["x"], ["mean"], name="mean", axes=[-1]),
            helper.make_node("Sub", ["x", "mean"], ["center"], name="center"),
            helper.make_node("Pow", ["center", "power"], ["square"], name="square"),
            helper.make_node("ReduceMean", ["square"], ["variance"], name="variance", axes=[-1]),
            helper.make_node("Add", ["eps", "variance"], ["positive"], name="positive"),
            helper.make_node("Sqrt", ["positive"], ["scale"], name="scale"),
            helper.make_node("Div", ["center", "scale"], ["normalized"], name="normalized"),
            helper.make_node("Mul", ["normalized", "gamma"], ["affine"], name="affine"),
        ]
        model = helper.make_model(helper.make_graph(nodes, "normalization", [], [], [
            helper.make_tensor("eps", TensorProto.FLOAT, [], [1e-5]),
            helper.make_tensor("power", TensorProto.FLOAT, [], [2]),
        ]))
        region, = centered_normalization_regions(model)
        self.assertEqual(len(region["source_nodes"]), 7)
        self.assertNotIn("affine", region["source_nodes"])
        model.graph.node[2].input[0] = "unrelated"
        self.assertEqual(centered_normalization_regions(model), [])
        model.graph.node[2].input[0] = "center"
        model.graph.node[3].attribute[0].ints[0] = -2
        self.assertEqual(centered_normalization_regions(model), [])

    def test_overlapping_timing_layers_and_domains_remain_separate(self):
        events = [
            {"name": "pass", "device": {"elapsedNs": "10000000"}},
            {"name": "Conv2D", "node": {}, "host": {"durationNs": "5000000"}},
            {"name": "Conv2D", "node": {}, "device": {"elapsedNs": "9000000"}},
            {"name": "conv", "node": {}, "program": {}, "device": {"elapsedNs": "8000000"}},
        ]
        for domain, layer, expected in [("host", "node", 5), ("device", "node", 9),
                                        ("device", "program", 8), ("device", "operation", 10)]:
            self.assertEqual(sum(row["total_ms"] for row in aggregate_volvox(events, domain, layer)), expected)

    def test_ort_fences_and_device_records_are_not_host_node_time(self):
        events = [
            {"cat": "Node", "name": "conv_kernel_time", "dur": 1200,
             "args": {"provider": "CUDAExecutionProvider", "op_name": "Conv"}},
            {"cat": "Node", "name": "conv_kernel_time", "dur": 800,
             "args": {"provider": "CPUExecutionProvider", "op_name": "Conv"}},
            {"cat": "Node", "name": "conv_fence_before", "dur": 400},
            {"cat": "Session", "name": "model_run", "dur": 5000},
            {"cat": "Kernel", "name": "device_kernel", "dur": 900},
        ]
        result = summarize_ort_events(events)
        self.assertEqual(result["raw_kernel_category_event_count"], 1)
        rows = result["operator_provider_host_intervals"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(row["total_host_ms"] for row in rows), 2)
        self.assertEqual({row["provider"] for row in rows}, {"CPUExecutionProvider", "CUDAExecutionProvider"})

    def test_lowered_region_includes_layout_cost_but_not_unrelated_nodes(self):
        graph = {"nodes": [{"id": "c", "opType": "Conv2D"},
                           {"id": "t", "opType": "Transpose"},
                           {"id": "a", "opType": "Add"}]}
        provenance = {"node_sources": [
            {"id": "c", "source_op": "Conv", "source_node": "original_conv"},
            {"id": "t", "source_op": "Conv", "source_node": "original_conv"},
            {"id": "a", "source_op": "Add", "source_node": "original_add"}]}
        stage = {"plans": [{"steps": [
            {"sourceNodeIds": ["c"], "cost": {"multiplyAccumulates": "64"}},
            {"scheduleIndex": 1, "sourceNodeIds": ["t"]},
            {"scheduleIndex": 2, "sourceNodeIds": ["a"]}]}], "events": [
            {"name": "Conv2D", "node": {}, "host": {"durationNs": "2000000"}},
            {"name": "conv", "node": {}, "program": {}, "device": {"elapsedNs": "3000000"}},
            {"name": "Transpose", "node": {"scheduleIndex": 1}, "host": {"durationNs": "1000000"}},
            {"name": "Add", "node": {"scheduleIndex": 2}, "host": {"durationNs": "9000000"}}]}
        region, = region_attribution(stage, graph, provenance)
        self.assertEqual(region["source_node_count"], 1)
        self.assertEqual(region["runtime_node_count"], 2)
        self.assertEqual(region["total_host_node_ms"], 3)
        self.assertEqual(region["total_device_program_ms"], 3)
        self.assertEqual(region["host_observed_schedule_steps"], 2)
        self.assertEqual(region["device_program_observed_schedule_steps"], 1)
        stage["events"] = []
        self.assertIsNone(region_attribution(stage, graph, provenance)[0]["total_device_program_ms"])


if __name__ == "__main__":
    unittest.main()
