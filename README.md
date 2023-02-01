# DPRN
This repository provides a PyTorch implementation of the paper [Maximization and restoration: Action segmentation through dilation passing and temporal reconstruction](https://www.sciencedirect.com/science/article/pii/S003132032200245X).

Tested with:
- PyTorch 0.4.1
- Python 2.7.12

## Training:

* Download the [data](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) folder, which contains the features and the ground truth labels. (~30GB) (If you cannot download the data from the previous link, try to download it from [here](https://zenodo.org/record/3625992#.Xiv9jGhKhPY))
* Extract it so that you have the `data` folder in the same directory as `main.py`.
* To train the model run `python main.py --action=train --dataset=DS --split=SP` where `DS` is `breakfast`, `50salads` or `gtea`, and `SP` is the split number (1-5) for 50salads and (1-4) for the other datasets.

## Prediction:

Run `python main.py --action=predict --dataset=DS --split=SP`. 

## Evaluation:

Run `python eval.py --dataset=DS --split=SP`. 

## Citation:

If you use the code, please cite

    Park, Junyong, et al. "Maximization and restoration: Action segmentation through dilation passing and temporal reconstruction." Pattern Recognition 129 (2022): 108764.

## Acknowledgement
The repository of [MS-TCN](https://github.com/yabufarha/ms-tcn) has been used for the general structure of this project