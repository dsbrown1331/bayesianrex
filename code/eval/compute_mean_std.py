import numpy as np
import sys
f = open(sys.argv[1])
returns = []
for line in f:
    returns.append(float(line))
print("mean, stdev {:.1f} ({:.1f})".format(np.mean(returns), np.std(returns)))
f.close()
