def random_deletion(sentence,p):
    
    if len(sentence) == 1 :
        return sentence
    
    words = sentence.split()
    new_words = []
    
    for word in words:
        r = random.uniform(0,1)
        if r > p :
            new_words.append(word)
    
    if len(new_words) == 0 :
        rand_int = random.randint(0,len(words)-1)
        return " ".join([words[rand_int]])
    
    return " ".join(new_words)