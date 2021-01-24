from flask import Flask, render_template, request
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    paragraph = ""
    book = ""
    if request.method == "POST":
        paragraph = request.form['paragraph']
        book = request.form['book']
        paragraphToLower = paragraph.lower()
        newParagraphToLower = paragraphToLower.replace('\n', '').replace('\r', '').replace('_', '')
        tokenizedSentences = sent_tokenize(newParagraphToLower)

        # Removing stopwords
        stopWords = stopwords.words('english')
        sentencesWithoutStopwords = list()
        for sent in tokenizedSentences:
            splitSentences = sent.split(' ')
            for word in stopWords:
                if word in splitSentences:
                    splitSentences.remove(word)
            joinedSentences = ' '.join(splitSentences)
            sentencesWithoutStopwords.append(joinedSentences)

        # Removing numbers
        sentencesWithoutNumbers = list()
        for sent in sentencesWithoutStopwords:
            tempString = re.sub(r"\d+", "", sent)
            sentencesWithoutNumbers.append(tempString)

        # Removing punctuations
        sentencesWithoutPunctuations = list()
        punctuations = '''!()-[]{};:'"\\, <>./?@#$%^&*_~'''
        for sent in sentencesWithoutNumbers:
            for punc in punctuations:
                if punc in sent:
                    newSent = sent.replace(punc, "")
            sentencesWithoutPunctuations.append(newSent)

        # Removing whitespaces
        sentencesWithoutWhitespaces = list()
        for sent in sentencesWithoutNumbers:
            tempString = sent.strip()
            sentencesWithoutWhitespaces.append(tempString)

        # Sentiment Analysis
        analyzedSentences = list()
        sa = SentimentIntensityAnalyzer()
        for ind, sent in enumerate(sentencesWithoutWhitespaces):
            score = sa.polarity_scores(sent)
            tempDict = {"Sentences": tokenizedSentences[ind], "Positive %": (score["pos"] * 100), "Negative %": (score["neg"] * 100),
                        "Neutral %": (score["neu"] * 100)}
            analyzedSentences.append(tempDict)

        data = pd.DataFrame(analyzedSentences)

        return render_template("success.html", data=data, book=book)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
