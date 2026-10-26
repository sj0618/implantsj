"""Check command routing without loading models or starting training."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run


class DispatcherTests(unittest.TestCase):
    def test_registered_modules_exist(self):
        for command, module in run.COMMANDS.items():
            with self.subTest(command=command):
                self.assertTrue((run.REPO_ROOT / (module.replace('.', '/') + '.py')).is_file())

    def test_forwards_arguments_and_exit_status(self):
        with patch.object(sys, 'argv', ['run.py', 'train', '--epochs', '2']), \
             patch.object(run.subprocess, 'call', return_value=7) as call:
            with self.assertRaises(SystemExit) as result:
                run.main()
        self.assertEqual(result.exception.code, 7)
        call.assert_called_once_with(
            [sys.executable, '-m', 'pipelines.train.train_supervised', '--epochs', '2'],
            cwd=str(run.REPO_ROOT),
        )

    def test_unknown_command_does_not_launch_process(self):
        with patch.object(sys, 'argv', ['run.py', 'does-not-exist']), \
             patch.object(run.subprocess, 'call') as call, patch.object(sys, 'stderr'):
            with self.assertRaises(SystemExit) as result:
                run.main()
        self.assertEqual(result.exception.code, 2)
        call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
