# Predict Sarcasm using Python and Machine Learning
This Python script uses Machine Learning to determine whether a given text input is either sarcastic or not sarcastic. It was trained using a dataset of 1.3 million Reddit comments.

The text input for this program should ideally be brief, like a social media comment. This program is meant for detecting sarcasm in social media comments.

Do note that it is not always 100% correct. Some cases of human sarcasm depend on additional context, body language, emoji, and other such things which this program does not take into account.

# Examples
See ```example-screenshot.PNG``` for a screenshot with example input and outputs.


# How can I run this script?

- Have NumPy, Pandas, and [scikit-learn](https://scikit-learn.org/stable/install.html) installed, as well as Python 3.10 or above.
- Save the dataset .csv file, which you can get on Kaggle (linked below).
- Make sure the correct path to the .csv is specified in the sarcasm.py script.
- Run the script.

# Credits
Dataset found [here](https://www.kaggle.com/datasets/danofer/sarcasm)

I also followed a tutorial to make this code, [here](https://thecleverprogrammer.com/2021/08/24/sarcasm-detection-with-machine-learning/) is the tutorial. Note that the tutorial uses a *different* dataset than the one here, with a different purpose- the tutorial linked is for a project to detect sarcasm in *news headlines* as opposed to a social media comment.
