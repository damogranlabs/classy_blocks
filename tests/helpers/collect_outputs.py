#!/usr/bin/env python
# Run from main directory (a.k.a. "./tests/helpers/collect_outputs.py")
"""Collects output files from all examples and stores them into tests/outputs;

TODO: THIS IS NOT A TEST CASE!
TODO: Proper CI/CD testing scenario and environment"""

import glob
import importlib
import shutil
import os
import subprocess

current_dir = os.getcwd()
outputs_dir = 'tests/outputs/'
case_dir = 'examples/case'

def collect_dict(path):
    # >>> path = examples/chaining/tank.py

    # don't import __init__s
    if path.endswith('__init__.py'):
        return

    # >>> ['examples', 'chaining', 'tank.py']
    exploded_path = path.split('/')

    # >>> 'tank'
    example_name = exploded_path[-1][:-3]

    # >>> 'examples/chaining'
    example_dir = '/'.join(exploded_path[:-1]) + '/'

    # >>> 'tests/outputs/examples/chaining'
    output_dir = outputs_dir + example_dir
    
    # >>> 'tests/outputs/examples/chaining/tank'
    output_path = output_dir + example_name

    # import string: strip .py and replace slashes with dots
    # >>> examples.chaining.tank
    import_string = path[:-3].replace('/', '.')

    print(f"Running {import_string} > {output_path}")

    example_module = importlib.import_module(import_string)
    mesh = example_module.get_mesh()

    mesh.write('examples/case/system/blockMeshDict')

    # create mesh and confirm checkMesh output
    os.chdir(case_dir)
    subprocess.run("blockMesh", check=True, capture_output=True)
    result = subprocess.run('checkMesh', check=True, capture_output=True)

    # check_result must contain 'Mesh OK.'
    assert result.stdout.splitlines(keepends=False)[-4] == b'Mesh OK.', \
        f"checkMesh failed: {import_string}; {result.stdout}"

    # change back to this dir to prepare the next example
    os.chdir(current_dir)

    # copy this 'checked' example to outputs to test against
    # (currently not implemented as it doesn't seem 
    #os.makedirs(output_dir, exist_ok=True)
    #shutil.copyfile('examples/case/system/blockMeshDict', output_path)

if __name__ == '__main__':
    examples = glob.glob('examples/*/*.py')

    for example_path in examples:
        collect_dict(example_path)
