from MergedDifferentialGrouping.mdg import MDG
import numpy as np
from benchmark.cec2013lsgo.cec2013 import Benchmark

benchmark = Benchmark()
for fun_id in range(1, 16):
    fun = benchmark.get_function(fun_id)
    info = benchmark.get_info(fun_id)


    mdg = MDG(fun, info)
    subspaces = mdg.run()
    print("Function ID:", fun_id)
    print("nonseps", len(subspaces["nonseps"]))
    print("sep", len(subspaces["seps"]))
