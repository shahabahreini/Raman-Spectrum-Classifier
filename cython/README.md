# How to compile it into Cython

1- copy nobitex.py and setting.py to this directory.<br />
2- run the following code:<br />
`python3 generate.py build_ext --inplace`<br />
or<br />
`python generate.py build_ext --inplace`<br />
3- delete nobitex.py and change the generated '.so' or '.pyx' file to 'nobitex.so' or 'nobitex.pyx'.<br />
4- remove "built" folder and nobitex.c files.<br />
5- run the following code, also include csv files into csv folder in the target directory:<br />
`python3 run.py`<br />
or<br />
`python run.py`<br />
<br />
more info: https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
