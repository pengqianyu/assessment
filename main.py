"""An algorithm to identify unseen captchas.
"""

from array import array
from matplotlib import pyplot as plt
import numpy as np
import os

os.chdir(os.path.abspath(os.path.dirname(__file__)))

class Captcha(object):
    """
    Method to learn representations for character/numeral
    and predict unseen captchas.
    """
            
    def __init__(self):
        filenames_input_raw = sorted('input/'+ filename for filename in 
                                    filter(lambda x: str.endswith(x, '.txt'),
                                           os.listdir('input')))
        filenames_output_raw = sorted('output/'+ filename for filename in 
                                    filter(lambda x: str.endswith(x, '.txt'),
                                           os.listdir('output')))
        in_number = [filename[-6:-4] for filename in filenames_input_raw]
        out_number = [filename[-6:-4] for filename in filenames_output_raw]
        self.valid_number = sorted(list(set(in_number).intersection(out_number)))
        self._char_dict = {}
        self._learn()
    
    def _learn(self) -> None:
        """
        Learn representation for each character/numeral.
        """
        inputs = ['input/input' + v_n + '.txt' for v_n in self.valid_number]
        outputs = ['output/output' + v_n + '.txt' for v_n in self.valid_number]
        image = self._grayscale_image(self._read_image(inputs[0]))
        row_idxs = list(np.where(image.sum(axis=1) == 60)[0])
        row_idx = np.argmax(np.array(row_idxs) - np.array(range(len(row_idxs))) > 0)
        self._row_start = row_idxs[row_idx - 1] + 1
        self._row_end = row_idxs[row_idx]

        for filename_input, filename_output in zip(inputs, outputs):
            image_raw = self._read_image(filename_input)
            image_processed = self._grayscale_image(image_raw)

            with open(filename_output) as file:
                labels = file.readlines()[0][:-1]
            
            column_indices = [-1] + list(np.where(image_processed.sum(axis=0) == 30)[0])
            
            j = 0
            for start_column, end_column in zip(column_indices[:-1], column_indices[1:]):
                if start_column + 1 != end_column:
                    character_represent = image_processed[self._row_start : self._row_end,
                                                          start_column + 1 : end_column]
                    self._char_dict[labels[j]] = character_represent
                    j += 1

    def _read_image(self, filename: str) -> np.array:
        """
        Read image from file.
        """
        with open(filename) as f:
            content = f.readlines()
        # first row contains the image shape
        rows, columns = [int(x) for x in content[0].split(' ')]
        image_raw = np.empty([rows, columns, 3])
        # the rest rows contain the image
        for i, line in enumerate(content[1:]):
            image_raw[i, :, :] = np.array([list(map(int,
                values.split(','))) for values in line.split(' ')])
        return image_raw

    def _grayscale_image(self, image_raw: np.array) -> np.array:
        """
        Grayscale the image.
        """
        image_new = image_raw.copy() / 255.0
        image_raw = (0.2126 * image_new[:,:,0] + 0.7152 * image_new[:,:,1] 
                        + 0.0722 * image_new[:,:,2])
        return np.where(image_raw < 0.5, 0, 1)

    def _find_chars(self, im_path: str) -> list:
        """
        Find characters/numerals in the image.
        """
        chars = []
        image = plt.imread(im_path)
        image_processed = self._grayscale_image(image)
        column_indices = [-1] + list(np.where(image_processed.sum(axis=0) == 30)[0])
        for start_column, end_column in zip(column_indices[:-1], column_indices[1:]):
            if start_column + 1 != end_column:
                img_char = image_processed[self._row_start : self._row_end,
                                                        start_column + 1 : end_column]
                for char, char_representation in self._char_dict.items():
                    if (img_char.shape[1] == char_representation.shape[1]) \
                         and (img_char == char_representation).all():
                        chars.append(char)
        return ''.join(chars)

    def __call__(self, im_path: str, save_path: str) -> None:
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        image = plt.imread(im_path)
        image = self._grayscale_image(image)
        chars = self._find_chars(im_path)
        with open(save_path, 'w+') as f:
            f.write(chars)


if __name__ == '__main__':
    captcha = Captcha()
    captcha('input/input00.jpg', 'output/out.txt')