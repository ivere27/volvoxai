"""Python execution adaptation validates scopes and borrowed-copy metadata."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import volvoxai as vx
from volvoxai._execution import execution_scope, array_destinations, validate_copies

pb = vx.pb


class ExecutionContractTest(unittest.TestCase):
    def entry(self, name, effect):
        return pb.ExecutionEntrypoint(name=name, effect=effect,
            allowed_ranges=[pb.ExecutionRangeKind.EXECUTION_RANGE_KIND_FULL], inputs=['x'], outputs=['y'])

    def test_only_unambiguous_stateless_full_evaluation_has_a_default(self):
        evaluate = self.entry('evaluate', pb.ExecutionEffect.EXECUTION_EFFECT_EVALUATE)
        contract = pb.ModelExecutionContract(stateless=pb.Empty(), entrypoints=[evaluate])
        self.assertEqual(execution_scope(contract).entrypoint, 'evaluate')
        contract.entrypoints.append(self.entry('reference', pb.ExecutionEffect.EXECUTION_EFFECT_EVALUATE))
        with self.assertRaisesRegex(ValueError, 'explicit entrypoint'):
            execution_scope(contract)
        self.assertEqual(execution_scope(contract, entrypoint='reference').entrypoint, 'reference')
        stateful = pb.ModelExecutionContract(incremental=pb.IncrementalStateContract(active_lengths_input='lengths'),
                                             entrypoints=[evaluate])
        with self.assertRaisesRegex(ValueError, 'explicit entrypoint'):
            execution_scope(stateful)
        scope = pb.ExecutionScope(entrypoint='evaluate', prefix_rows=pb.PrefixRows(count=2))
        self.assertIs(execution_scope(stateful, scope=scope), scope)
        with self.assertRaises(TypeError):
            execution_scope(stateful, scope=scope, entrypoint='evaluate')

    def tensor(self, name='y', size=8):
        return pb.Tensor(name=name, shape=[2], dtype=pb.DataType.DATA_TYPE_F32,
                         buffer=pb.BufferView(buffer_id=5, length_bytes=size))

    def test_inconsistent_snapshot_metadata_is_rejected_before_host_copy(self):
        for outputs in ([self.tensor(size=12)], [self.tensor(), self.tensor()], [self.tensor(name='wrong')]):
            with self.subTest(outputs=outputs), self.assertRaises(ValueError):
                array_destinations(outputs, ['y'])

    def test_copy_receipt_must_match_owned_destination_exactly(self):
        arrays, request = array_destinations([self.tensor()], ['y'])
        receipt = pb.TensorBatch(outputs=[pb.Tensor(name='y', shape=[2], dtype=pb.DataType.DATA_TYPE_F32,
                                                    borrowed=request.into[0])])
        validate_copies(receipt, arrays)
        self.assertTrue(arrays['y'].flags.owndata)
        receipt.outputs[0].borrowed.resource.handle += 4
        with self.assertRaisesRegex(ValueError, 'destination'):
            validate_copies(receipt, arrays)

    def test_empty_selection_needs_no_destination_storage(self):
        arrays, request = array_destinations([], [])
        self.assertEqual(arrays, {})
        self.assertEqual(request.into, [])
        self.assertEqual(request.sources, [])


if __name__ == '__main__':
    unittest.main()
