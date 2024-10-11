# Learning to Cut Generation in Branch-and-Cut algorithms for Combinatorial Optimization
This repository contains the code for the paper "Learning to Cut Generation in Branch-and-Cut algorithms for Combinatorial Optimization".

## Requirements
The code is written in Python 3.7 and CPLEX 12.10. The academic version of CPLEX can be downloaded in https://www.ibm.com/academic/.
To use CPLEX Python API, change the path in the 'solvers/cplex_api.py' file to the path where the CPLEX is installed. For example:
```
sys.path.append('C:/Program Files/IBM/ILOG/CPLEX_Studio1210/cplex/python/3.7/x64_win64')
```
For the other packages, they are listed in the `requirements.txt` file and you can install them using the following command:
```
pip install -r requirements.txt
```
To simplify the installation of the packages, we recommend using Docker. The Dockerfile is provided in the repository.

## Datasets
We provide the datasets used in the paper in the 'data' folder. 
We also provide the scripts to generate the datasets in the 'tsp-generator' and 'maxcut-generator' folders.

## Running the experiments
We provide the bash scripts to run the experiments in the 'bash_scripts' folder.
To run the experiments, you can use the following command:
```
cd bash_scripts
bash file_name.sh
```


