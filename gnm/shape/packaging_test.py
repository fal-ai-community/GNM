# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packaging regression tests for the gnm-shape wheel.

Builds the wheel from this directory and verifies that it ships the
``gnm.shape.data.versions`` modules (``gnm_catalog``, ``gnm_specs``) alongside
the model data. Guards against a regression where ``[tool.setuptools]
packages`` omitted ``gnm.shape.data.versions``, so installed wheels raised
``ModuleNotFoundError`` on ``from gnm.shape.data.versions import gnm_catalog``
even though the ``.npz``/``.h5`` data files were present.
"""

import glob
import os
import subprocess
import sys
import tempfile
import zipfile

from absl.testing import absltest

_SHAPE_DIR = os.path.dirname(os.path.abspath(__file__))

_REQUIRED_MODULE_FILES = (
    'gnm/shape/data/versions/gnm_catalog.py',
    'gnm/shape/data/versions/gnm_specs.py',
)

_REQUIRED_DATA_FILES = (
    'gnm/shape/data/versions/v3_0/gnm_head.npz',
    'gnm/shape/data/semantic_sampler/identity_decoder_model.h5',
    'gnm/shape/data/semantic_sampler/expression_decoder_model.h5',
)


class PackagingTest(absltest.TestCase):
  """Builds the gnm-shape wheel once and inspects its contents."""

  _tmp_dir = None
  wheel_path = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._tmp_dir = tempfile.TemporaryDirectory()
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'pip',
            'wheel',
            '--no-deps',
            '--wheel-dir',
            cls._tmp_dir.name,
            _SHAPE_DIR,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
      raise RuntimeError(
          f'Building the gnm-shape wheel failed:\n{result.stdout}\n'
          f'{result.stderr}'
      )
    wheels = glob.glob(os.path.join(cls._tmp_dir.name, 'gnm_shape-*.whl'))
    if len(wheels) != 1:
      raise RuntimeError(f'Expected exactly one gnm-shape wheel, got {wheels}')
    cls.wheel_path = wheels[0]

  @classmethod
  def tearDownClass(cls):
    if cls._tmp_dir is not None:
      cls._tmp_dir.cleanup()
    super().tearDownClass()

  def test_wheel_ships_data_versions_modules(self):
    """The catalog/specs modules must be packaged, not just the model data."""
    with zipfile.ZipFile(self.wheel_path) as wheel:
      names = set(wheel.namelist())
    for module_file in _REQUIRED_MODULE_FILES:
      self.assertIn(module_file, names)

  def test_wheel_ships_model_data(self):
    """The .npz model file and .h5 sampler decoders must stay packaged."""
    with zipfile.ZipFile(self.wheel_path) as wheel:
      names = set(wheel.namelist())
    for data_file in _REQUIRED_DATA_FILES:
      self.assertIn(data_file, names)

  def test_data_versions_modules_import_from_wheel(self):
    """The modules must be importable from the wheel without the source tree."""
    with tempfile.TemporaryDirectory() as unpacked_dir:
      with zipfile.ZipFile(self.wheel_path) as wheel:
        wheel.extractall(unpacked_dir)

      env = dict(os.environ, PYTHONPATH=unpacked_dir)
      script = (
          'from gnm.shape.data.versions import gnm_catalog, gnm_specs; '
          "assert gnm_catalog.VARIANT_TO_MODEL_FILE_NAME_MAP['head'] == "
          "'gnm_head'; "
          "assert gnm_specs.GNMVersion('3.0') is gnm_specs.GNMVersion.V3_0"
      )
      # -S skips site initialization so an editable install of gnm-shape in the
      # test environment cannot satisfy the import; only the unpacked wheel on
      # PYTHONPATH can. Run from the unpacked dir so the implicit cwd sys.path
      # entry points at the wheel contents too.
      result = subprocess.run(
          [sys.executable, '-S', '-c', script],
          capture_output=True,
          text=True,
          check=False,
          cwd=unpacked_dir,
          env=env,
      )
      self.assertEqual(
          result.returncode,
          0,
          f'Importing from the wheel failed:\n{result.stdout}\n{result.stderr}',
      )


if __name__ == '__main__':
  absltest.main()
