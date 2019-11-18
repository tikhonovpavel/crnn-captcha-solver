# crnn-captcha-solver

Solving captcha using CRNN + CTC loss

<!---## Results--->
The model has been trained on 3500 examples which look like this:

![example1color](https://github.com/tikhonovpavel/crnn-captcha-solver/raw/master/data/raw_color_images/2970.jpg)

![example2color](https://github.com/tikhonovpavel/crnn-captcha-solver/raw/master/data/raw_color_images/3691.jpg)

![example3color](https://github.com/tikhonovpavel/crnn-captcha-solver/raw/master/data/raw_color_images/2949.jpg)

They've been binarized to remove background noise:

![example1](https://raw.githubusercontent.com/tikhonovpavel/crnn-captcha-solver/master/data/train/h4q2h_2970.png)

![example2](https://raw.githubusercontent.com/tikhonovpavel/crnn-captcha-solver/master/data/train/qz45h_3691.png)

![example3](https://raw.githubusercontent.com/tikhonovpavel/crnn-captcha-solver/master/data/train/veq4n_2949.png)


The model shows about 76% accuracy on the validation set.

##### Augmentations used in the model: 
Random rotations by a small angle and piecewise affine transformation - to simulate distortions in captcha images.

## Train on your own data
- Move your data to `train` and `test` dirs. To handle the problem of images with the same texts, your files should be named as `<captcha_text>_<index>.png`, for example `veq4n_2949.png`
- Run `tool/create_dataset.py` for train and test and specify directory for output lmdb dataset
- You can specify your character set in `alphabets.py`
- You can change learning parameters in `params.py`

## Test
Use `inference.py` to predict your captcha images. Specify `plot_needed` parameter to visualize output of the model.

For example, for given image:

![example4](https://raw.githubusercontent.com/tikhonovpavel/crnn-captcha-solver/master/data/raw_color_images/3256.jpg)

we can visualize output probability distributions for channels `s`, `4`, `d`, `7` and `-` (empty):

![example5](https://raw.githubusercontent.com/tikhonovpavel/crnn-captcha-solver/master/data/raw_color_images/example_output.png)

## Reference

[meijieru/crnn.pytorch](<https://github.com/meijieru/crnn.pytorch>)
