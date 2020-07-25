#!/usr/local/bin/python3
#
# Code by: Vivek Shresta, Akhil Mokkapati, Vijay Sai Kondamadugu  - vivband, akmokka, vikond
#
import sys
from os import listdir
from os.path import isfile, join
from math import log


def getWordCountInMails(directory):
    mailNames = getFileNamesInDirectory(directory)
    wordCount = {}
    numberOfMails = 0

    for mailName in mailNames:
        numberOfMails += 1
        uniqueWordsInMail = set([])
        with open(directory + "/" + mailName, 'r') as mail:
            for line in mail:
                words = line.split(" ")
                for word in words:
                    word = word.lower()
                    # DATA CLEANING
                    # if shouldWordBeCleaned(word):
                    #     continue
                    if word not in uniqueWordsInMail:
                        uniqueWordsInMail.add(word)
                        count = wordCount.get(word)
                        if count is None:
                            wordCount[word] = 1
                        else:
                            count += 1
                            wordCount[word] = count
    return wordCount, numberOfMails


def getFileNamesInDirectory(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f))]


def spamClassification(testDirectory, spamMails, notSpamMails, outputFileName):
    mailNames = getFileNamesInDirectory(testDirectory)
    outputFile = open(outputFileName, "w+")

    for mailName in mailNames:
        spamProbability = 0
        notSpamProbability = 0
        with open(testDirectory + "/" + mailName, 'r') as mail:
            for line in mail:
                words = line.split(" ")
                for word in words:
                    word = word.lower()
                    # DATA CLEANING
                    # if shouldWordBeCleaned(word):
                    #     continue
                    countInSpamDict = spamMails[0].get(word)
                    countInNonSpamDict = notSpamMails[0].get(word)
                    if countInSpamDict is None:
                        countInSpamDict = 0
                    if countInNonSpamDict is None:
                        countInNonSpamDict = 0
                    # When there is no word in the training data set, we add a very small alpha (Laplace Smoothing)
                    # Calculating the probabilities of all the words
                    countInSpamDict += 0.1
                    countInNonSpamDict += 0.1
                    spamProbability = spamProbability + log(countInSpamDict) - log(spamMails[1])
                    notSpamProbability = notSpamProbability + log(countInNonSpamDict) - log(notSpamMails[1])

            # Calculating the probability of being a spam mail or a not spam mail
            spamProbability = spamProbability + log(spamMails[1])
            notSpamProbability = notSpamProbability + log(notSpamMails[1])

            if spamProbability > notSpamProbability:
                outputFile.write(mailName + " spam\n")
            else:
                outputFile.write(mailName + " notspam\n")

    outputFile.close()


def shouldWordBeCleaned(word):
    if len(word) == 1 or len(word) == 0 or not word.isalpha():
        return True

# def calculateAccuracy(finalOutput, expectedOutput):
#     dict = {}
#     accuracy = 0
#     with open(expectedOutput, 'r') as file:
#         for line in file:
#             words = line.split(" ")
#             dict[words[0]] = words[1][0:-1]
#
#     with open(finalOutput, 'r') as output:
#         for line in output:
#             words = line.split(" ")
#             if dict[words[0]] == words[1][0:-1]:
#                 accuracy += 1
#
#     print float(accuracy*100)/2554


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception("Error: expected 3 arguments")

    spamMails = getWordCountInMails(sys.argv[1] + "/spam")
    notSpamMails = getWordCountInMails(sys.argv[1] + "/notspam")

    spamClassification(sys.argv[2], spamMails, notSpamMails, sys.argv[3])

    # calculateAccuracy(sys.argv[3], "/Users/vivekshresta/Downloads/CodeBase/vivband-akmokka-vikond-a3/part3/test-groundtruth.txt")