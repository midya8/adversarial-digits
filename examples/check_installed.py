"""Check the installed tests and full CLI outputs outside the checkout.

Use --section tests or --section commands to run either half on its own.
The default runs both. These are same-runtime packaging comparisons, not
independent numerical or statistical evidence.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
ENV.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
MODULES = ['digitlab', 'digitlab.__main__', 'digitlab.data', 'digitlab.model',
           'digitlab.train', 'digitlab.attack', 'digitlab.evaluate', 'digitlab.box',
           'digitlab.radius', 'digitlab.attribution', 'digitlab.segment']
COMMANDS = [
    ['demo'], ['train', '--artifacts', 'fresh'], ['evaluate', '--artifacts', 'fresh'],
    ['predict', '--model', 'fresh/mlp.json', '--input', 'examples/input.json'],
    ['box'], ['box-audit', '--artifacts', 'box-fresh'],
    ['radius'], ['radius-audit', '--artifacts', 'radius-fresh'],
    ['attribution'], ['attribution-audit', '--artifacts', 'attribution-fresh'],
]
FILES = ['fresh/' + name for name in ['linear.json', 'mlp.json', 'protocol.json',
         'selection.json', 'evaluation-plan.json', 'report.json']]
FILES += ['box-fresh/' + name for name in ['box-report.json', 'box-0.json',
          'box-1-20.json', 'box-1-10.json', 'box-1-5.json']]
FILES += ['radius-fresh/radius-report.json', 'attribution-fresh/attribution-report.json']


def load(path):
    return json.loads(path.read_bytes())


def call(args, cwd, env=ENV):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--section', choices=['all', 'tests', 'commands'], default='all')
args = parser.parse_args()
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    wheels = work / 'wheels'
    wheels.mkdir()
    call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels), str(ROOT)], work)
    built = list(wheels.glob('*.whl'))
    assert len(built) == 1, built
    venv.create(work / 'env', with_pip=True)
    py = str(work / 'env/bin/python')
    call([py, '-m', 'pip', 'install', str(built[0])], work)
    outside = work / 'outside'
    outside.mkdir()
    for name in ['tests', 'examples', 'docs', 'assets']:
        shutil.copytree(ROOT / name, outside / name)
    for name in ['artifacts', 'data']:
        shutil.copytree(ROOT / 'digitlab' / name, outside / 'digitlab' / name)
    assert not list((outside / 'digitlab').rglob('*.py'))
    suite = '''
import importlib,importlib.util,json,pathlib,sys,unittest
outside=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(outside))
names=json.loads(sys.argv[1])
source=pathlib.Path(sys.argv[2]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
for name in names:importlib.import_module(name)
def check():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
check()
if sys.argv[3]=='tests':
    tests=unittest.defaultTestLoader.discover(str(outside/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.testsRun>0 and result.wasSuccessful()
    print(json.dumps(dict(tests=result.testsRun)))
check()
print(json.dumps(dict(installed_origins=len(names),checked_before_and_after=True)))
'''
    if args.section in ['all', 'tests']:
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'tests'], outside))
    if args.section in ['all', 'commands']:
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'imports'], outside))
        reference = work / 'reference'
        reference.mkdir()
        for name in ['examples', 'digitlab/artifacts']:
            shutil.copytree(ROOT / name, reference / name)
        source_env = ENV.copy()
        source_env['PYTHONPATH'] = str(ROOT)
        expected = []
        for command in COMMANDS:
            output = call([sys.executable, '-m', 'digitlab'] + command, reference, source_env).strip()
            expected.append(output)
        files = {name: load(reference / name) for name in FILES}
        # This package has no console-script entrypoint. Check its module CLI.
        for command, report in zip(COMMANDS, expected):
            output = call([py, '-I', '-m', 'digitlab'] + command, outside).strip()
            assert output == report, command
        for name, value in files.items():
            assert load(outside / name) == value, name
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'imports'], outside))
        print(json.dumps(dict(command_invocations=len(COMMANDS), command_kinds=10,
                              complete_output_files=len(FILES), entrypoints=1)))
