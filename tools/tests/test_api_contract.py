import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_api_contract as contract


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        contract.load_dependencies()
        cls.fd = contract.descriptor()
        cls.pool = contract.descriptor_pool.DescriptorPool()
        cls.pool.Add(cls.fd)

    def modified_annotation(self, path, annotation):
        fd = contract.descriptor_pb2.FileDescriptorProto()
        fd.CopyFrom(self.fd)
        location = next(loc for loc in fd.source_code_info.location if list(loc.path) == path)
        location.leading_comments = '@api ' + json.dumps(annotation)
        return fd

    def test_unknown_field_paths_fail_generation(self):
        index = next(i for i, m in enumerate(self.fd.message_type) if m.name == 'LoadModelRequest')
        fd = self.modified_annotation([4, index], {'rules': [
            {'kind': 'EXACTLY_ONE', 'fields': ['graph_path', 'misspelled_package']},
        ]})
        with self.assertRaisesRegex(ValueError, 'unknown field path'):
            contract.contracts(fd, self.pool)

    def test_unknown_annotation_keys_fail_generation(self):
        index = next(i for i, m in enumerate(self.fd.message_type) if m.name == 'Tensor')
        fd = self.modified_annotation([4, index, 2, 0], {'requred': True})
        with self.assertRaisesRegex(ValueError, 'unsupported @api keys'):
            contract.contracts(fd, self.pool)

    def test_profile_metadata_keeps_only_available_method_type_closure(self):
        fd = contract.descriptor_pb2.FileDescriptorProto()
        fd.CopyFrom(self.fd)
        contract.filter_generated_descriptor(fd, set(contract.INFERENCE_SERVICES),
            set(contract.GENERATION_SPECS['c-inference'].omitted_enum_values))
        methods, types = contract.contracts(fd, self.pool)
        self.assertFalse(any(m.service == 'VxTrainingService' for m in methods))
        self.assertNotIn('volvoxai.v1.TrainStepRequest', types)
        self.assertIn('volvoxai.v1.InputValidationIssue', types)
        codes = types['volvoxai.v1.OperationCode'][1]
        self.assertFalse(any(v.name == 'OPERATION_CODE_TRAIN_STEP_FAILED' for v in codes.values))
        for _, _, dependencies in types.values():
            self.assertTrue(set(dependencies) <= set(types))


if __name__ == '__main__':
    unittest.main()
