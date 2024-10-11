import os 
import subprocess 

if __name__ == "__main__":
    nInstances = 100
    nNodes = 120
    density = 10

    for i in range(9000, 9000 + nInstances):
        if not os.path.isdir(os.path.join("data", str(nNodes))):
            os.mkdir(os.path.join("data", str(nNodes)))
        instance_name = f"pm{nNodes}_{density}_{i}.maxcut"

        cmd = ["./rudy", "-rnd_graph", str(nNodes), str(density), str(i), "-random", str(0), str(10), str(i)]
        print(cmd)

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        out = p.communicate()

        with open(os.path.join("data", str(nNodes), instance_name), "w") as file:
            file.write(out[0].decode("utf-8"))
