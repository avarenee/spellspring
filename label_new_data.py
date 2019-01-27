import pickle, sys, ast

def label_new_data(sentiments):
    sentiments = sentiments.split(",")
    log_model = pickle.load(open("log_model.pkl", "rb"), encoding='latin1')
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"), encoding='latin1')
    sentiments = vectorizer.transform(sentiments).toarray()
    labels = log_model.predict(sentiments)
    for i in range(len(labels)):
        print(labels[i])

label_new_data(sys.argv[1])


