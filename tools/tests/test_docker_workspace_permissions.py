"""Regression tests preventing root-owned artifacts from breaking persistent runners."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DockerWorkspacePermissionsTests(unittest.TestCase):
    def test_docker_run_disables_python_bytecode_writes(self):
        makefile = (ROOT / 'Makefile').read_text(encoding='utf-8')
        match = re.search(
            r'^DOCKER_RUN\s*=\s*(.*?)(?:\n(?!\t)|\Z)',
            makefile,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match, 'Makefile must define DOCKER_RUN')
        docker_run = match.group(1)
        self.assertIn(
            '-e PYTHONDONTWRITEBYTECODE=1',
            docker_run,
            'DOCKER_RUN must set PYTHONDONTWRITEBYTECODE=1 so root Python '
            'invocations inside Docker do not leave root-owned __pycache__ trees',
        )

    def test_root_container_cleanups_restore_workspace_ownership(self):
        makefile = (ROOT / 'Makefile').read_text(encoding='utf-8')
        self.assertIn('CHOWN_HOST_WORKSPACE_ROOT =', makefile)
        cleanups = re.findall(r'cleanup\(\)\s*\{(.*?)\n\t\t\};', makefile, re.DOTALL)
        self.assertGreaterEqual(len(cleanups), 10)
        for index, body in enumerate(cleanups):
            self.assertIn(
                '$(CHOWN_HOST_WORKSPACE_ROOT)',
                body,
                f'Makefile cleanup() trap #{index + 1} must invoke '
                '$(CHOWN_HOST_WORKSPACE_ROOT) so no root-owned files remain under /workspace',
            )

    def test_self_hosted_linux_jobs_heal_all_root_owned_files_before_and_after(self):
        workflows = [
            ROOT / '.github' / 'workflows' / 'gpu-smoke.yml',
            ROOT / '.github' / 'workflows' / 'build.yml',
        ]
        for workflow_path in workflows:
            text = workflow_path.read_text(encoding='utf-8')
            self.assertNotIn(
                'for dir in native build dist; do',
                text,
                f'{workflow_path.name} must scan the whole workspace for root-owned '
                'entries instead of hardcoding only native build dist',
            )
            self.assertEqual(
                len(re.findall(r'runs-on:\s*\[self-hosted,\s*linux,', text)),
                1,
                f'Expected one self-hosted Linux job in {workflow_path.name}',
            )
            self.assertIn(
                'Restore stale root-owned workspace files before checkout',
                text,
            )
            self.assertIn(
                'Restore workspace permissions on self-hosted runner',
                text,
            )
            pre_idx = text.index('Restore stale root-owned workspace files before checkout')
            checkout_idx = text.index('actions/checkout@v4', pre_idx)
            post_idx = text.index('Restore workspace permissions on self-hosted runner')
            self.assertLess(pre_idx, checkout_idx)
            self.assertLess(checkout_idx, post_idx)
            self.assertIn('-xdev -depth -user 0', text[pre_idx:checkout_idx])
            self.assertIn('-xdev -depth -user 0', text[post_idx:])


if __name__ == '__main__':
    unittest.main()
