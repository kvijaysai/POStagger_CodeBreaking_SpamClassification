# Part 1: POS Tagger
 #### Part of Speech tagging is achieved by using the following algorithms:
- Simplified (the most-probable tag for each word)
- Viterbi Algorithm (maximum a posteriori (MAP))
- Complex (Max Marginal using Gibbs Sampling)
#### Processed the given training data to learn the probabilities required to implement the algorithms mentioned above.
- The probability that a particular POS_Tag comes as the first POS_Tag in a sentence
- The probability that a particular POS_Tag follows a given POS_Tag
- The probability of word given POS_Tag
- The probability of given POS_Tag

#### We use the above mentioned probabilities for our algorithms to follow.
- Simplified (the most-probable tag for each word): 
 - Here we are selecting a POS by finding most probable tag for each word.
 - Si*  = arg max P(Si=si | W) = argmax [ P(Wi | Si) * P(Si) ]
- Viterbi Algorithm (maximum a posteriori (MAP))
 - Here we are selecting POS by finding MAP labeling for the sentence.
 - (S1*  , S2*  , S3*  …, Sn*  )= arg max P(Si=si | W) = argmax [ P(Wi | Si) * P(Si | Si-1) ]
- Complex (Max Marginal using Gibbs Sampling)
 - To calculate marginal distribution from samples, we first generated 2000 samples using Gibbs Sampling. The first 500 samples were discarded to pass the healing period and improve sampling accuracy. From the remaining 1500 samples we calculate max probability of each tag corresponding to each word in the test sentence. We assign the pos tag which has the maximum probability for a word. And combining them we get the pos tags for the whole sentence.
#### Posterior Probabilities Calculation: Sum over each word in a sentence of function F
 - Simplified: 
  - F = ( log(p(word|tag) * p(tag)) )
 - Viterbi Algorithm:
  - F = log( "p(word|tag) * p(tag|previous tag)") if other than first word
  - F = log( "p(word|tag) * p_start(tag)") if first word
 - Complex: 
  - F = log( (p(word|tag)* p(tag|previous tag))  for 2nd to N-1 words
  - F = log( p(word|tag) * p(tag))      for first word  
  - F = log(p(word|tag) * p(tag|first tag, previous tag))   for last word

#### Results:
Results: Scored 2000 sentences with 29442 words.
                    Words correct:     Sentences correct:
1. Ground truth:      100.00%              100.00%
2. Simple:             93.92%               47.50%
3. HMM:                95.50%               57.55%
4. Complex:            94.04%               47.55%

#### Assumptions:
1.	If we encounter a new word in test set, then we take associated POS tag as "noun" (“noun” is the high probable tag in training set).
2.	If we encounter a new word in test set, then we take priors and emissions as very small probability number:  Pow(10,-10)
3.	And for any combinations in priors ,emissions  and transition probabilities tables if values are zeros then its replaced with very low probability =Pow(10,-10)

#### Problems Faced:
1.	It is very difficult to correctly classify a word that has never been in our training set. After multiple trials, found out that setting tag to high probable tag in training set i.e "noun” produced better results.
2.	Large number of iterations are required to attain higher accuracy through Gibbs Sampling. Tried out different combinations of iterations for healing period, Actual Sampling to check the impact on Runtime and Accuracy. Finally taken iterations for healing period as 500 and Actual Sampling Iterations =1500 which resulted in approx. 3 secs per sentence.


# Part 2: Code Breaking
We are given a secret message that is encrypted using both of two techniques. Replacement, where each letter of the alphabet is replaced with another letter of the alphabet. In Rearrangement, the order of the characters is scrambled for each consecutive sequence of 4 characters. We decrypt this message using probabilistic, simple Markov chain over alphabets

We calculate this document probabilities based on:
  
![equation]( https://latex.codecogs.com/gif.latex?P(D)&space;=&space;\prod&space;_{i}&space;P(W_{i}))
  
![equation]( https://latex.codecogs.com/gif.latex?P%28W_%7Bi%7D%29%20%3D%20P%28W_%7Bi%7D%5E%7B0%7D%29%5Cprod_%7Bj%3D1%7D%5E%7B%7CW_%7Bi%7D%7C-1%7DP%28W_%7Bi%7D%5E%7Bj&plus;1%7D%7CW_%7Bi%7D%5E%7Bj%7D%29)
  
using the given English corpus. 

We use these probabilities in decrypting. To simplify breaking the code, we use the Metropolis-Hastings algorithm:

1.	Start with a guess about the encryption tables. Call the guess T.
- We generate a random guess of both replacement and rearrangement table as an initial guess to decoding process
2.	Modify T to produce a new guess T_hat. 
- In this process, we guess either replacement or rearrangement based on generating a random number between 0 & 1
- If this number is greater than 0.5, we do replacement, else rearrangement
  - Replacement in this step is just switching 2 letters randomly in guess T
  - Rearrangement in this step is just swapping two numbers in in rearrangement table of guess T

3. Decrypt the encoded document using T to produce document D, and using T_hat to produce D_hat
- We decode the input encrypted text using guess T to produce D and T_hat to produce D_hat
- We calculate probability of documents D and D_hat as P(D) and P(D_hat) respectively
- In calculating this we use log values of transition probabilities calculated above and add them up, to avoid decimal value underflow
- We split the document using space to get the words and calculate this probabiltiy using above formulas

4. If P(D_hat) > P(D); then replace T with T_hat

4.1. Here we take the replacement and rearrangement tables used for T_hat and send it to the next iteration to improve the guess

4.2. Otherwise, we are taking a bernouli(exp(log(P(D_hat))- log(P(D)))) random variable 
- if this is 1 we replace T with T_hat and follow the same procedure as 4.1

5. We go to step 2
- In this way, we iterate for over 9 minutes and keep saving top 100 decoded strings based on their probabilities
- After 9 minutes we exit and give the decrypted output as the string with highest probability out of 100 populations saved.


# Part 3: Spam Classification
The first step of Spam Classification is to parse all the mails in the training directory and maintain the occurrences of each word in spam 
and not spam mail.

This is best done using a dictionary, since the count retrieval in a dictionary takes O(1) time.(Constant amount of time)

So, first we read through all the files in the spam folder, parse each file and retrieve MEANINGFUL words and put them into the dictionary. If there 
is an existing entry of the word in the dictionary, the count is just incremented. This way, an accurate count of all the words in all the files in 
the spam folder are created.

The same process is used to get the count of each word in the not spam folder.

The most important part of this process is the data cleaning. If no data cleaning is done at all, the current accuracy is '98.512%'. Since the most 
important part of any email classification is the content, the data cleaning we introduced(Please look at the function 'shouldWordBeCleaned()') makes 
sure only real words are considered for our classification. Any words with special characters or any single letter words such as 'a' are not considered 
which gave us an accuracy of '97.611%'. Using third party libraries might obviously give us better accuracy, but since SICE servers doesn't have those 
libraries available, we're not going to use our simple data cleaning method. The code for data cleaning is commented out in the code currently.
(Looks like the same thing is mentioned on Piazza.
https://piazza.com/class/jzswlrc6gg6pe?cid=330)

So once we have all the counts with us, all we need to do to classify an email as a spam or not spam mail. This can be done with the 'Naive Bayes' 
classifier. 
The naive bayes formula is:
P(S|W1,W2..Wn) = P(W1|S)P(W2|S)..P(Wn|S)P(S) / P(W1|S)P(W2|S)..P(Wn|S)P(S) + P(W1|nS)P(W2|nS)..P(Wn|S)P(nS)
P(nS|W1,W2..Wn) = P(W1|nS)P(W2|nS)..P(Wn|nS)P(nS) / P(W1|S)P(W2|S)..P(Wn|S)P(S) + P(W1|nS)P(W2|nS)..P(Wn|S)P(nS)
Here S = Spam
nS = not Spam
W1 = Word 1
W2 = Word 2
Wn = Word n

If the whole probability is greater than 50%, then according to Naive bayes classifier, we can classify the email as a Spam. Or else, not spam.

The better approach would be to just compare P(S|W1,W2..Wn) and P(nS|W1,W2..Wn). Since, the denominators are same, we can just calculate 
P(W1|S)P(W2|S)..P(Wn|S)P(S) and P(W1|nS)P(W2|nS)..P(Wn|S)P(nS) and compare them. This is a better approach since division involves more CPU cycles 
when compared to multiplication.

If we keep on multiplying less values, the resultant would be a very very less value which would lead to underflow. This is when 'logs' come into 
picture. Since logs monotonically increase, we can just compare the resultant logarithmic values and need not apply any anti-log. This process 
reduces the time complexity by a little and mitigates any underflow which might happen.

During the classification, we might come across variables which might not be in our training set. When such a word occurs, since the count of the word 
is 0, the whole probability would be 0 which might lead the mail to be classified as not spam; which might clear not be the case. Basically, one word 
is deciding the fate of the whole mail which is very wrong. This is when 'Smoothing parameter or Laplace Smoothing' comes into picture. By introducing 
a small alpha values; a very small probability, the probability of a word which is never seen in the training set is reduced to a very small number, 
but not 0. This way we mitigate the issue of extremes during classifying. Another approach could be to just ignore such words, but there's an 
issue with this approach too. Basically ignoring the word would mean that we're giving it a probability of 1, which would mean, this word will only 
come in a spam mail when in reality it didn't occur even once. 
That is why the Laplace Smoothing Parameter is very important here. 
