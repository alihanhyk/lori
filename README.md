# Inferring Lexicographically-Ordered Rewards from Preferences
Code author: Alihan Hüyük ([ah2075@cam.ac.uk](mailto:ah2075@cam.ac.uk))

This repository contains the source code necessary to replicate the main experimental results in the AAAI 2022 paper "[Inferring Lexicographically-Ordered Reward from Preferences]()." Our proposed method, *LORI*, is implemented in files `src/main-lori.py` and `src/main-lori-liver.py` for the problem settings considered in the paper: cancer treatment and organ transplantation respectively.

### Usage
First, install the required python packages by running:
```shell
    python -m pip install -r requirements.txt
```

Then, the experiments in the paper can be replicated by running:
```shell
    ./src/run.sh        # generates the results in Tables 2 and 3
    ./src/run-liver.sh  # generates the reward functions in (10) and (11)
```

Note that, in order to run the experiments for the transplantation setting, you need to get access to the [Organ Procurement and Transplantation Network (OPTN)](https://optn.transplant.hrsa.gov) dataset for liver transplantations as of December 4, 2020.

### Citing
If you use this software please cite as follows:
```
@inproceedings{huyuk2022inferring,
  author={Alihan H\"uy\"uk and William R. Zame and Mihaela van der Schaar},
  title={Inferring lexicographically-ordered rewards from preferences},
  booktitle={Proceedings of the 36th AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
