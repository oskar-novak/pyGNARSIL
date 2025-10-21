1. pyGNARSIL is an updated version of our original algorithm in:
https://www.computer.org/csdl/proceedings-article/qce/2024/413701a109/23opYjjytyg

2. pyGNARSIL splits stabilizers into gauge operators!
*Useful for CSS and non-CSS codes.
Uses a new parallelized graph optimization strategy, which speeds things up a lot.
***Use $[Lx,S,Lz]$ form for input matrix:
   
See baconShorexample.py in tests for example code

3. Code parallelized using Numba: making the code scale much better.
** This requires the code to compile the first time!

4. Cite the original Gnarsil paper if you use this code!

5. install using pip install git+https://github.com/oskar-novak/pyGNARSIL.git







