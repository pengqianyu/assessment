# Assessment

## Problem statement

A website uses Captchas on a form to keep the web-bots away. However, the captchas it generates,
are quite similar each time:
- the number of characters remains the same each time
- the font and spacing is the same each time
- the background and foreground colors and texture, remain largely the same
- there is no skew in the structure of the characters.
- the captcha generator, creates strictly 5-character captchas, and each of the characters is either an
upper-case character (A-Z) or a numeral (0-9).

You are provided a set of twenty-five captchas, such that, each of the characters A-Z and 0-9 occur at
least once in one of the Captchas&#39; text. From these captchas, you can identify texture, nature of the
font, spacing of the font, morphological characteristics of the letters and numerals, etc. Download
this sample set from [here](http://hr-testcases.s3.amazonaws.com/2587/assets/sampleCaptchas.zip) for the purpose of creating a simple AI model or algorithm to identify the
unseen captchas.

## Method

There are four main methods: `_read_image`, `_preprocess_image`, `_split_data_and_learn`, and `_evaluate` in the Captcha class.

1. `_read_image`: Given the path to the image txt, read the image as a numpy array.
2. `_preprocess_image`: Normalize the image. 
3. `_split_data_and_learn`: Construct train/test dataset and train the SVM model. Since the number of features is larger than the number of training examples, a linear kernel is used. 
4. `_evaluate`: Evaluate the model performance on the test dataset. 

## Usage
Suppose an unseen captcha is located at `input/input100.jpg` and in `main.py` we set
```
    captcha = Captcha()
    captcha('input/input100.jpg', 'output/output.txt')
```
Run `main.py` will produce a file `output.txt` that contains all characters/numerals in the unseen captcha.
