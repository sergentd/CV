# import necessary packages
from pyimagesearch.pypeline import Pypeline
from pyimagesearch.pypeline import Step

s1 = Step("part_1.py", "first step", [["input", "input.txt"]],
    verbose=True, enable=True)
s2 = Step("part_10.py", "second step", [["input", "input.txt"],["output", "v5"]],
    verbose=True, enable=True)

p = Pypeline(verbose=True)
p.add(s1)
p.add(s2)

p.execute()
