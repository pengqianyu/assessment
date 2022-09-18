"""An algorithm to identify unseen captchas.
"""

import numpy as np
import os

from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


os.chdir(os.path.abspath(os.path.dirname(__file__)))

class Captcha(object):
    """
    An algorithm to learn representations for character/numeral
    and predict unseen captchas.
    """
            
    def __init__(self, ratio=0.2, num_chars=5):
        filenames_input_raw = sorted('input/'+ filename for filename in 
                                    filter(lambda x: str.endswith(x, '.txt'),
                                           os.listdir('input')))
        filenames_output_raw = sorted('output/'+ filename for filename in 
                                    filter(lambda x: str.endswith(x, '.txt'),
                                           os.listdir('output')))
        in_number = [filename[-6:-4] for filename in filenames_input_raw]
        out_number = [filename[-6:-4] for filename in filenames_output_raw]
        self.valid_number = sorted(list(set(in_number).intersection(out_number)))

        self._inputs = ['input/input' + v_n + '.txt' for v_n in self.valid_number]
        self._outputs = ['output/output' + v_n + '.txt' for v_n in self.valid_number]
        image = self._preprocess_image(self._read_image(self._inputs[0]))
        image = np.where(image < 0.5, 0, 1)
        row_idxs = list(np.where(image.sum(axis=1) == 60)[0])
        row_idx = np.argmax(np.array(row_idxs) - np.array(range(len(row_idxs))) > 0)
        self._row_start = row_idxs[row_idx - 1] +1
        self._row_end = row_idxs[row_idx]
        col_idxs = list(np.where(image.sum(axis=0) == 30)[0])
        self._col_start = np.argmax(np.array(col_idxs) - np.array(range(len(col_idxs))) > 0)
        self._size = max(np.diff(np.array(col_idxs) - np.array(range(len(col_idxs))))) + 1
        assert (ratio > 0 and ratio < 1), 'train/test ratio should be in (0,1).'
        self._ratio = ratio
        self._num_chars = num_chars

        self._split_data_and_learn()
        self._evaluate()
    
    def _split_data_and_learn(self) -> None:
        """
        Construct train/test datasets and train the SVM model.
        """
        X = []
        Y = []
        for filename_input, filename_output in zip(self._inputs, self._outputs):
            image_raw = self._read_image(filename_input)
            image_processed = self._preprocess_image(image_raw)

            with open(filename_output) as file:
                labels = file.readlines()[0][:-1]

            col_num = self._col_start
            for i in range(self._num_chars):
                X.append(image_processed[self._row_start : self._row_end,
                    col_num : col_num + self._size].flatten())
                Y.append(labels[i])
                col_num += self._size
        
        X = np.array(X)
        X_train, self._X_test, Y_train, self._Y_test = train_test_split(
            X, Y, test_size=self._ratio, random_state=42)
        self._clf = svm.SVC(kernel="linear")
        self._clf.fit(X_train, Y_train)

    def _evaluate(self) -> None:
        """
        Evaluate the model.
        """
        #  Not able to do cross validation here due to small sample size of each class. 
        Y_pred = self._clf.predict(self._X_test)
        f1 = f1_score(self._Y_test, Y_pred, average = 'macro')
        print('Using {:.1f}% of data, test accuracy is {:.2f}%, F1 score is {:.2f}'.format(
            (1-self._ratio)*100, accuracy_score(self._Y_test, Y_pred)*100, f1))

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

    def _preprocess_image(self, image_raw: np.array) -> np.array:
        """
        Preprocess the image.
        """
        return image_raw.copy()[:, :, 0] / 255.0

    def __call__(self, im_path: str, save_path: str) -> None:
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        image = plt.imread(im_path)
        image_processed = self._preprocess_image(image)
        col_num = self._col_start
        chars = []
        for _ in range(self._num_chars):
            test = image_processed[self._row_start : self._row_end,
                col_num : col_num + self._size].reshape(1, -1)
            chars.append(self._clf.predict(test)[0])
            col_num += self._size
        out = ''.join(chars)
        with open(save_path, 'w+') as f:
            f.write(out)


if __name__ == '__main__':
    captcha = Captcha(ratio=0.2)
    captcha('input/input100.jpg', 'output/out.txt')