###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids: Akhil Mokkapati, Vijay Sai Kondamadugu, Vivek Shreshta - akmokka, vikond, vivband
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import numpy as np
import copy as dp

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    #posterior probability was caluclated in the same way the probalilities were caluclated in the individual functions in the first place
    def posterior(self, model, sentence, label):
    #This function log(probability (proportional measure)) for given sentence by taking given tag of each word. probability proportional to "p(word/tag) * p(tag)"
            if model == "Simple":
                a=math.log(self.pos_Prob[self.pos_dict[label[0]]])
                if sentence[0] in self.word_dict:
                    a += math.log(self.EP[self.word_dict[sentence[0]],self.pos_dict[label[0]]])
                else:
                    a +=math.log(math.pow(10,-10))
                for i in range(1, len(sentence)):
                    if sentence[i] in self.word_dict:
                        a+= math.log(self.EP[self.word_dict[sentence[i]],self.pos_dict[label[i]]])
                    else:
                        a +=math.log(math.pow(10,-10))
                    a+= math.log(self.pos_Prob[self.pos_dict[label[i]]])
                return a
    
    #This function gives log(probability (proportional measure)) for given sentence.
    #probability(proportional)  = (p(word/tag) * p(tag/previous tag)) for 2nd to N-1 words and p(word/tag) * p(tag)for first word  
    #and p(word/tag) * p(tag/first tag,previous tag) for last word
            elif model == "Complex":
                a = math.log(self.Start_Prob[self.pos_dict[label[0]]])
                if sentence[0] in self.word_dict:
                    a += math.log(self.EP[self.word_dict[sentence[0]],self.pos_dict[label[0]]])
                else:
                    a +=math.log(math.pow(10,-10))
                for i in range(1, len(sentence)):
                    if i <len(sentence)-1:
                        a += math.log(self.Trans_Prob[self.pos_dict[label[i-1]],self.pos_dict[label[i]]])
                        if sentence[i] in self.word_dict:
                            a+= math.log(self.EP[self.word_dict[sentence[i]],self.pos_dict[label[i]]])
                        else:
                            a +=math.log(math.pow(10,-10))
                    else:
                        pos_1_N = label[0]+'_'+ label[-2]
                        if pos_1_N in self.pos_count_1_N and label[-1] in self.pos_count_1_N[pos_1_N ]:         
                            a += math.log(self.pos_count_1_N[pos_1_N][label[-1]])
                        else:
                            a +=math.log(math.pow(10,-10))
          
                return a
     
    #This function gives log(probability (proportional measure)) for given sentence. "p(word/tag) * p(tag/previous tag)"
            elif model == "HMM":
                if sentence[0] in self.word_dict:
                    a = math.log(self.EP[self.word_dict[sentence[0]],self.pos_dict[label[0]]])
                else:
                    a =math.log(math.pow(10,-10))                
                a+=math.log(self.Start_Prob[self.pos_dict[label[0]]])
                for i in range(1,len(label)):
                    try:
                        if sentence[i] in self.word_dict:
                            a+= math.log(self.EP[self.word_dict[sentence[i]],self.pos_dict[label[i]]])
                        else:
                            a +=math.log(math.pow(10,-10))
                        a+=math.log(self.Trans_Prob[self.pos_dict[label[i-1]],self.pos_dict[label[i]]])
                        
                    except:
                        continue
                return a
            else:
                print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        self.word_dict ={} # Words in train and corresponding unique ID
        self.pos_dict ={}   # Tags in train and corresponding unique ID
        self.pos_dict_rev ={}  # List of ID and their corresponding Tags ( Reverse to pos_dict)
        self.word_pos ={} # Words and Tags cobinations count
        self.pos_pos_trans ={} # Transition Probability between 2 Tags
        self.pos_start ={} # Probability for a Tag at start of Sentence
        self.word_id =0 # Total number of words in Train data 
        self.pos_id =0 #Total number of tags in train data
        self.pos_count_1_N = {} # Conditional probability of P(Sn/S1,Sn-1)
        for sentence,pos_seq in data:
            i=0
            for word in sentence:
                if i>0:
                    # If Left_Tag is not present then insert Left_Tag,Right_Tag combo count to pos_pos_trans
                    if pos_seq[i-1] not in self.pos_pos_trans.keys():
                       self.pos_pos_trans[pos_seq[i-1]] = {pos_seq[i] :1} 
                    # If Left_Tag is present then insert Right_Tag as child combo count to pos_pos_trans   
                    elif pos_seq[i] not in self.pos_pos_trans[pos_seq[i-1]].keys():
                        self.pos_pos_trans[pos_seq[i-1]][pos_seq[i]] = 1
                    # If Left_Tag and Right_Tag is present then increment count to pos_pos_trans   
                    else:
                        self.pos_pos_trans[pos_seq[i-1]][pos_seq[i]] += 1
                else:
                    # i=0 means first word, so calculating Prob of start for a Tag ( no transition calculation for 1st word)
                    if pos_seq[i] not in self.pos_start.keys():
                       self.pos_start[pos_seq[i]] = 1
                    else:
                        self.pos_start[pos_seq[i]] += 1
                #Populating Word and Tag combinations count(will be used for emission prob calculation)
                if word not in self.word_pos.keys():
                    self.word_pos[word] = {pos_seq[i] :1}
                    self.word_dict[word] =self.word_id
                    self.word_id+=1
                    if pos_seq[i] not in self.pos_dict.keys():
                        self.pos_dict[pos_seq[i]] =self.pos_id
                        self.pos_dict_rev[self.pos_id] =pos_seq[i]
                        self.pos_id+=1
                else:
                    if pos_seq[i] in self.word_pos[word].keys():                
                        self.word_pos[word][pos_seq[i]]+=1
                    else:
                        self.word_pos[word][pos_seq[i]]=1
                        if pos_seq[i] not in self.pos_dict.keys():
                            self.pos_dict[pos_seq[i]] =self.pos_id
                            self.pos_dict_rev[self.pos_id] =pos_seq[i]
                            self.pos_id+=1
                i+=1
                # calculating transition probability of (S1,Sn-1) and Sn  (** P(Sn/S1,Sn-1))
                #This is valid only if len(sentence)>2
            if i >3:
                pos_1_N =pos_seq[0] +'_'+pos_seq[-2]
                if pos_1_N not in self.pos_count_1_N:
                    self.pos_count_1_N[pos_1_N] ={pos_seq[-1]: 1}
                elif pos_seq[-1] not in self.pos_count_1_N[pos_1_N]:
                    self.pos_count_1_N[pos_1_N][pos_seq[-1]] =1
                else:
                    self.pos_count_1_N[pos_1_N][pos_seq[-1]] =+1
        
        self.word_pos_matrix = np.array([[0 for j in range(self.pos_id)] for i in range(self.word_id)])
        self.pos_pos_matrix = np.array([[0 for j in range(self.pos_id)] for i in range(self.pos_id)])
        self.pos_start_count = np.array([0 for j in range(self.pos_id)])
        
        for pos_1_N in self.pos_count_1_N:
            total = 0
            for pos_N in self.pos_count_1_N[pos_1_N]:
                total += self.pos_count_1_N[pos_1_N][pos_N]
            for pos_N in self.pos_count_1_N[pos_1_N]:
                # Conditional probability of P(Sn/S1,Sn-1)
                self.pos_count_1_N[pos_1_N][pos_N] = float(self.pos_count_1_N[pos_1_N][pos_N]) / float(total)
        # Words and tags matrix with corresponding counts 
        for word,pos_count in self.word_pos.items():
            for pos,count in pos_count.items():
                self.word_pos_matrix[self.word_dict[word]][self.pos_dict[pos]] =count
        # left tags and right tags matrix with corresponding counts
        for pos,pos_T_count in self.pos_pos_trans.items():
            for pos_T,count in pos_T_count.items():
                self.pos_pos_matrix[self.pos_dict[pos]][self.pos_dict[pos_T]] =count
        #tag wise start word tags
        for pos,pos_count in self.pos_start.items():
            self.pos_start_count[self.pos_dict[pos]] =pos_count
                
        pos_count =[np.sum(self.word_pos_matrix[:,x]) for x in range(self.pos_id)]
        max_pos_id = np.where(pos_count == np.amax(pos_count))[0]
        # tag with max count in training
        self.max_pos = self.pos_dict_rev[int(max_pos_id)]
        #tag wise Probabilities
        self.pos_Prob = np.true_divide(pos_count,sum(pos_count))
        
        #calculating emission probabilities
        self.EP = dp.deepcopy(self.word_pos_matrix)    
        self.EP = np.true_divide(self.EP,self. EP.sum(axis=0, keepdims=True))
        self.EP[self.EP == 0] = math.pow(10,-10)

        #calculating Transition probabilities
        self.Trans_Prob = dp.deepcopy(self.pos_pos_matrix)    
        self.Trans_Prob = np.true_divide(self.Trans_Prob, self.Trans_Prob.sum(axis=1, keepdims=True))
        self.Trans_Prob[self.Trans_Prob == 0] = math.pow(10,-10)
        # Calculating tags starting probabilities
        self.Start_Prob = dp.deepcopy(self.pos_start_count)
        self.Start_Prob = np.true_divide(self.Start_Prob, self.Start_Prob.sum()) 
        #if anyone of them are 0 replace with negligible probability math.pow(10,-10) to avoid log errors
        self.Start_Prob[self.Start_Prob == 0] = math.pow(10,-10)


    #MCMC and Gibbs Sampling
    def mcmc(self, sentence, sample_count):
#        sample = ["noun"] * len(sentence) 
        sample = dp.copy(self.pos_seq_simplified)  # initial sample output of simplified 
        for i in range(500):  # ignore first 500 samples (healing period)
            sample = self.sampling(sentence, sample)
        samples = []
        # Will use following Samples for prob calculations 
        for p in range(sample_count):
            sample = self.sampling(sentence, sample)
            samples.append(sample)
        return samples

    def sampling(self, sentence, sample):
            n_words = len(sentence)
            pos_tags = list(self.pos_pos_trans)
#            i=0
            num =list(range(n_words))
#            for word in sentence:
            for j in range(n_words):
                #Randomly selecting whcich tag to be changed fixing others.
                [i]=random.sample(num,1)
                num.remove(i)
                word = sentence[i]
                pos_prob_sample = np.array([0] * self.pos_id)
    
                for j in range(self.pos_id):  # try by assigning every tag  
                    if i==0:
                        parent_Trans = self.Start_Prob[j]
                        if n_words>1:
                            Child_Trans = self.Trans_Prob[j, self.pos_dict[sample[i+1]]]
                        else:
                            Child_Trans=1
                    elif i < n_words-1:
                        parent_Trans = self.Trans_Prob[self.pos_dict[sample[i-1]],j]
                        Child_Trans = self.Trans_Prob[j, self.pos_dict[sample[i+1]]]
                    else:
#                        parent_Trans = self.Trans_Prob[self.pos_dict[sample[i-1]],j]
                        pos_1_N = sample[0]+'_'+ sample[-2]
                        if pos_1_N in self.pos_count_1_N and sample[-1] in self.pos_count_1_N[pos_1_N ]:         
                            parent_Trans = self.pos_count_1_N[pos_1_N][sample[-1]]
                        else:
                            parent_Trans = math.pow(10,-10)
                        Child_Trans = 1
                    if word in self.word_dict:
                        emis_prob = self.EP[self.word_dict[word], j]
                    else:
                        emis_prob = self.pos_Prob[j]
                        
                    pos_prob_sample[j] = np.log(parent_Trans)+ np.log(Child_Trans)+ np.log(emis_prob)
    
                pos_prob_sample_ratio = np.true_divide(np.exp(pos_prob_sample), np.exp(pos_prob_sample).sum())
                rand = np.random.random()
                cum_prob = 0
                for j in range(self.pos_id):
                    cum_prob += pos_prob_sample_ratio[j]
                    if rand < cum_prob:
                        sample[i] = pos_tags[j]
                        break
#                i+=1
            return sample


    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        self.pos_seq_simplified=[]        
        for word in sentence:
            if word in self.word_dict.keys():
                arr = self.word_pos_matrix[self.word_dict[word],:]
                pos = np.where(arr == np.amax(arr))[0][-1]
                self.pos_seq_simplified.append(self.pos_dict_rev[int(pos)])
            else:
                self.pos_seq_simplified.append(self.max_pos)
        return self.pos_seq_simplified

    def complex_mcmc(self, sentence):
        sample_count = 1000
        samples = self.mcmc(sentence, sample_count)
        probabilities = []
        final_sample = []
    
        for i in range(len(sentence)):
            tag_count = dict.fromkeys(self.pos_pos_trans, 0)
            for sample in samples:
                tag_count[sample[i]] += 1
            final_sample.append(max(tag_count, key=tag_count.get))
            probabilities.append(tag_count[final_sample[i]] / sample_count)
            
        return final_sample

    def hmm_viterbi(self, sentence):
        V = [[[0, []] for col in range(len(sentence))] for row in range(self.pos_id)]
        i=0
        for word in sentence:
            if word in self.word_dict.keys():
                if i==0:
                 for p_id in range(self.pos_id):
                    V[p_id][i] = [np.log(self.EP[self.word_dict[word], p_id]) + np.log(self.Start_Prob[p_id]),[self.pos_dict_rev[p_id]]]
                else:    
                    for p_id in range(self.pos_id):
                        [max_arg_P, Max_arg_seq] = max([[V[p_id_2][i-1][0] + np.log(self.Trans_Prob[p_id_2, p_id]), V[p_id_2][i-1][1]] for p_id_2 in range(self.pos_id)])
                        V[p_id][i][0] = np.log(self.EP[self.word_dict[word], p_id]) + max_arg_P
                        V[p_id][i][1] = Max_arg_seq + [self.pos_dict_rev[p_id]]
                        
            else:
                if i==0:
                 for p_id in range(self.pos_id):
                    V[p_id][i] = [np.log(self.pos_Prob[p_id])+ np.log(self.Start_Prob[p_id]),[self.pos_dict_rev[p_id]]]
                else:    
                    for p_id in range(self.pos_id):
                        [max_arg_P, Max_arg_seq] = max([[V[p_id_2][i-1][0] + np.log(self.Trans_Prob[p_id_2, p_id]), V[p_id_2][i-1][1]] for p_id_2 in range(self.pos_id)])
                        V[p_id][i][0] = np.log(self.pos_Prob[p_id])+ max_arg_P
                        V[p_id][i][1] = Max_arg_seq + [self.pos_dict_rev[p_id]]
            i=i+1
        [self.max_arg_P, [self.Max_arg_seq]] = max([[V[p_id][i-1][0], [V[p_id][i-1][1]]] for p_id in range(self.pos_id)])
                
        return self.Max_arg_seq


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

