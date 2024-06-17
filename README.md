## Required packages:

  1. numpy

  2. torch

  3. Levenshtein

  4. pickle

  5. dill

  6. tqdm

     

## Python Code:

 - `main.py` contain the execution of MOEA algorithms
 - `MOEA.py` contain the implementation of MOEAs (i.e., PVD-NSGA-II-WR, PVD-GSEMO, PVD-GSEMO-R and PVD-GSEMO-WR).
 - `/data` preprocessed real datasets are saved in this folder.



## Execution

#### Run the algorithms:

```
python main.py -ea alg_name -dataFile file_name 
```

For example: `python main.py -ea PVD-GSEMO-R -dataFile mhc1_credences ` 

All the default parameters are set based on the paper. You can change them in the code.

The results will be saved in `/result` folder. 