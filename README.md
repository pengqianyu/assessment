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

There are four main methods: `_read_image`, `_grayscale_image`, `_learn` and `_find_chars` in the Captcha class.

1. `_read_image`: Given the path to the image txt, read the image as a numpy array;
2. `_grayscale_image`: Convert the image to grayscale;
3. `_learn`: Use the sample set images and find pixel-wise representations where we need to identidy the start/end columns for each character/numeral. It creates a dictionary that maps character/numeral from to numpy array;
4. `_find_chars`: Given a new image, find all characters/numerals in the image. Here a similar technique is used to identidy the start/end columns for each character/numeral as in `_learn`.

## Usage
Suppose an unseen captcha is located at `input/input00.jpg` and in `main.py` we set
```
    captcha = Captcha()
    captcha('input/input00.jpg', 'output/output.txt')
```
Run `main.py` will produce a file `output.txt` that contains all characters/numerals in the unseen captcha.
