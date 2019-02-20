import nltk
import numpy as np
import random
import string # to process standard python strings
import warnings
warnings.filterwarnings("ignore")


# We will read in the corpus.txt file and convert the entire corpus into a list of sentences 
# and a list of words for further pre-processing.

f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase
# nltk.download('punkt') # first-time use only
# nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

sent_tokens[:2]
word_tokens[:5]





# We shall now define a function called LemTokens which will take as input the tokens 
# and return normalized tokens.

lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))



# Next, we shall define a function for a greeting by the bot i.e if a user’s input is a greeting, 
# the bot shall return a greeting response.ELIZA uses a simple keyword matching for greetings. 
# We will utilize the same concept here.

GREETING_INPUTS = ("oi", "oie", "eae", "eai", "tudo bem",)
GREETING_RESPONSES = ["oi", "oie", "eae", "eai", "tudo bem"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)  



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# We define a function response which searches the user’s utterance for one or more known keywords 
# and returns one of several possible responses. If it doesn’t find the input matching any of the keywords, 
# it returns a response:” I am sorry! I don’t understand you”

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words = nltk.corpus.stopwords.words("portuguese"))
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"Não entendi, abaixa o volume da TV e me escuta pelo telefone"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response




# Finally, we will feed the lines that we want our bot to say while starting 
# and ending a conversation depending upon user’s input.

flag=True
print("GALVÃO: Bem amigos da rede Globo, meu nome é Galvão Bueno, qual é a sua dúvida sobre futebol?")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='tchau'):
        if(user_response=='obrigado' or user_response=='valeu' ):
            flag=False
            print("GALVÃO: A gende se vê por aqui..")
        else:
            if(greeting(user_response)!=None):
                print("GALVÃO: "+greeting(user_response))
            else:
                print("GALVÃO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("GALVÃO: A gente se vê por aqui..")