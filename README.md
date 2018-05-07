# Neural-K-Pop-Star-FFTNet
Implementation of FFTNet, [paper](http://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/fftnet-jin2018.pdf)

## FFTNet_split.py
 - implementation using LR split 1x1 conv, as in the [paper](http://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/fftnet-jin2018.pdf).
 - see [description](https://github.com/mimbres/Neural-K-Pop-Star-FFTNet/blob/master/Explanation_of_sum(split_1x1_conv).md) 
 - no generator
 
## FFTNet_dilconv.py
 - implementation using 2x1 dilated-conv.
 - generator avaialble
 
Note that 2x1 dilated-conv is equivalent with LR split 1x1 conv.

## In progress..
 - YesNoDataset
 - CMU Arctic (found a good parser in [r9y9's repository](https://github.com/r9y9/wavenet_vocoder))
