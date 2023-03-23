#!/usr/bin/env python
# Run from main directory (a.k.a. "./tests/helpers/collect_outputs.py")
"""Collects output files from all examples and stores them into tests/outputs;

TODO: THIS IS NOT A TEST CASE!
TODO: Proper CI/CD testing scenario and environment"""

import glob
import subprocess
import os
import shutil

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
    example_file = exploded_path[-1]
    example_name = example_file[:-3]

    # >>> 'examples/chaining'
    example_dir = '/'.join(exploded_path[:-1]) + '/'

    # >>> 'tests/outputs/examples/chaining'
    output_dir = outputs_dir + example_dir
    
    # >>> 'tests/outputs/examples/chaining/tank'
    output_path = output_dir + example_name

    os.chdir(example_dir)
    subprocess.run(['python', example_file], check=False)
    print(f"Running {output_path} > {output_path}")
    os.chdir(current_dir)


    # create mesh and confirm checkMesh output
    os.chdir(case_dir)
    subprocess.run("blockMesh", check=True, capture_output=True)
    result = subprocess.run('checkMesh', check=True, capture_output=True)

    # check_result must contain 'Mesh OK.'
    assert result.stdout.splitlines(keepends=False)[-4] == b'Mesh OK.', \
        f"checkMesh failed: {path}; {result.stdout}"

    # change back to this dir to prepare the next example
    os.chdir(current_dir)

    # copy this 'checked' example to outputs to test against
    # (currently not implemented as it doesn't seem 
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile('examples/case/system/blockMeshDict', output_path)

if __name__ == '__main__':
    examples = glob.glob('examples/*/*.py')

    for example_path in examples:
        collect_dict(example_path)
