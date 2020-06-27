import os
import sys
fd = os.listdir(sys.argv[1])
for nname in fd:
    if " " in nname:
        print("Skipping " + str(nname) + " due to space")
        continue
    os.system("python DemoGraph.py " + sys.argv[1] + " " + str(nname))
