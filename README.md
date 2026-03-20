# CSE 583 Project

## Installation (Great Lakes)

Start by installing `uv`, which will handle virtual environments and dependencies: 
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Ensure Python 3.10 is installed:
```
uv python install 3.10
```

Then, from the project root, run: 

```
make install
```

Finally, run 

```
source .venv/bin/activate
```

to activate the virtual environment. 

Verify your installation. Create a Python interpreter by running `python` and verify you get similar output: 
```
>>> import gym
>>> import compiler_gym
>>> compiler_gym.COMPILER_GYM_ENVS
['llvm-v0', 'llvm-ic-v0', 'llvm-autophase-ic-v0', 'llvm-ir-ic-v0']
```