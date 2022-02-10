# dlts_paper_code (TF1 version)

This folder contains the TensorFlow 1.15 code used in the experiments in the paper ["Deep Learning in Target Space."](https://jmlr.org/papers/v23/20-040.html) Fairbank, Michael, Spyridon Samothrakis, and Luca Citi. Journal of Machine Learning Research 23.8 (2022): 1-46.   Also published as [arXiv:2006.01578](https://arxiv.org/abs/2006.01578).

Please cite the above paper if this code or future variants of it are used in future academic work.  Also, we welcome contact from anyone who's found this code or method useful.

### Purpose

The purpose of the code in this subfolder is to archive the experiments performed in the above paper, and to make those results replicable.  

- This code is not the most flexible for future projects.  

- The keras layers code in the rest of this repository is easier to use for future work.

## Running the code

- There are four runnable python scripts in this repository.  

- See the leading comments in each script for usage examples.


## Dependencies

All were built using tensorflow v.1.15, but should be compatible with Tensorflow versions 2.x.

Also used :

- Python 3.8

- numpy version 1.17, 

- pandas version 1.0.3 

## Two-Spirals result

When running the two spirals script, with the --graphical argument, we should see a result compatible with this figure from the paper:

![Two-Spirals image](../spirals_image.png)