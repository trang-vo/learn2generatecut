import subprocess
import os

if __name__ == "__main__":
    methods = ["portcgen"]
    nInstances = 999
    nNodes = 300

    for i in range(1, nInstances + 1):
        for method in methods:
            cmd = "./" + method
            data_folder = f"data/{nNodes}"
            if not os.path.isdir(data_folder):
                os.makedirs(data_folder)
            if method == "portgen":
                instance_name = f"data/{nNodes}/E{nNodes}_{i}.tsp"
            else:
                instance_name = f"data/{nNodes}/C{nNodes}_{i}.tsp"
            print(cmd, str(nNodes), str(i), instance_name)
            p = subprocess.Popen([cmd, str(nNodes), str(i), instance_name], stdout=subprocess.PIPE)
            out = p.communicate()
            with open(instance_name, "w") as file:
                file.write(out[0].decode())