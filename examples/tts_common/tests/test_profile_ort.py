import unittest
import numpy as np
from examples.tts_common.profile_ort import summarize_trace, validate_outputs


class ProfileOrtTest(unittest.TestCase):
    def test_warmed_run_partition_and_cuda_host_timing(self):
        events = []
        for index, duration in enumerate([999, 777, 21, 31]):
            start = index * 2000
            events.append({'cat': 'Session', 'name': 'model_run', 'ph': 'X', 'ts': start, 'dur': 1500})
            events.append({'cat': 'Node', 'name': 'conv_kernel_time', 'ph': 'X', 'ts': start+10,
                'dur': duration, 'args': {'provider': 'CUDAExecutionProvider', 'op_name': 'Conv',
                    'node_index': '7', 'input_type_shape': [{'float': [1, 4, 8]}]}})
        # A CUDA provider and a "kernel_time" suffix still describe a host node.
        result = summarize_trace(events, warmups=1, repeats=2)
        self.assertFalse(result['device_timing_available'])
        row = result['summaries']['operators'][0]
        self.assertEqual(row['samples_per_run_host_us'], [21, 31])
        self.assertEqual(row['event_count'], 2)
        self.assertEqual(row['mean_per_run_host_us'], 26)
        events.append({'cat': 'Kernel', 'name': 'real_cuda_kernel', 'ph': 'X', 'ts': 4020, 'dur': 13})
        self.assertEqual(summarize_trace(events, 1, 2)['explicit_device_kernel_events'], 1)

    def test_missing_or_overlapping_run_boundaries_are_not_guessed(self):
        with self.assertRaisesRegex(ValueError, 'Expected'):
            summarize_trace([], 0, 1)
        events = [{'name': 'model_run', 'ph': 'X', 'ts': t, 'dur': 20} for t in [0, 10]]
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            summarize_trace(events, 0, 1)

    def test_changed_duration_is_a_visible_shape_difference(self):
        result = validate_outputs({'audio': np.zeros((1, 8), np.float32)},
                                  {'audio': np.zeros((1, 6), np.float32)})['audio']
        self.assertFalse(result['numerical_parity'])
        self.assertEqual(result['shape'], [1, 8])
        self.assertEqual(result['reference_shape'], [1, 6])
        self.assertNotIn('rmse', result)
        with self.assertRaisesRegex(ValueError, 'finiteness'):
            validate_outputs({'audio': np.array([np.nan], np.float32)},
                             {'audio': np.zeros(1, np.float32)})


if __name__ == '__main__':
    unittest.main()
