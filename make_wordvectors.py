
def read_data(filename, encoding='utf-8', sep='\t', isSentence=0):
    '''
    this method have no any dependencies
    reading a e'ncoding type' of strings from 'filename'
        
    Arg : read_data('train_data.txt', 'utf-8-sig')
    print(read_data('train_data.txt', 'utf-8-sig')
    
    :encoding # when you wanna remove \ufeff, use 'utf-8-sig'
    :return: listed lists as a line
    [['날씨가 정말 덥구나'], ['비가 오네']]
    
    
    '''
    import codecs
    if isSentence == 1 :
        with codecs.open(filename, encoding=encoding,
                         mode='r') as f:  # can't aware BOM on utf-16 if it is big or little endian

            data = ""
            for line in f.read().splitlines():
                if len(line) == 0:
                    continue
                else:
                    data += " " + line
    else :
        with codecs.open(filename, encoding=encoding, mode='r') as f:
            data = [line.split(sep) for line in f.read().splitlines()]
    return data


def tokenize(doc):
    '''
    need to import konlp as a dependency
    spliting a string into taged morpheme
     
    Args:
      sent: A string. A sentence.
      
    Returns: A list of words.
    ['날씨/Noun', '가/Josa', '정말/Noun', '덥다/Adjective']
    '''
    from konlpy.tag import Twitter
    tagger = Twitter()
    return ['/'.join(t) for t in tagger.pos(doc, norm=True, stem=True)]

def clean_text(text):
    import regex
    # Common
    text = regex.sub("(?s)<ref>.+?</ref>", "", text)  # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text)  # remove html tags
    text = regex.sub("&[a-z]+;", "", text)  # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text)  # remove markup tags
    text = regex.sub("(?s){.+?}", "", text)  # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text)  # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text)  # remove media links
    text = regex.sub("[']{5}", "", text)  # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text)  # remove bold symbols
    text = regex.sub("[']{2}", "", text)  # remove italic symbols
    text = regex.sub(u"[^ \r\n\p{Hangul}.?!]", " ", text)  # Replace unacceptable characters with a space.
    return text

def sentence_segment(text):
    '''
    Args:
      text: A string. A unsegmented paragraph.

    Returns:
      A list of sentences.
    '''
    import regex
    sents = regex.split("([.?!])?[\n]+|[.?!] ", text)
    return sents


def input_data(dir_list, numOfFile, encoding="utf-8"):
    '''      
    :param dir_list: list of file name(size is the number of label)
    dir_list=[['normal'], ['sadness']]     
    :param length: the number of file in each directory
    
    :return: list of contents in each directory
    [[I'm a boy], [You look like sad]]
    '''

    import os
    filename = os.getcwd()
    data = [[] for i in range(len(dir_list))]
    for file_i in range(len(dir_list)):
        for iter in range(numOfFile):
            filename = os.getcwd() + '\\'+dir_list[file_i][0]+ '\\' + str(iter + 1) + '.txt'
            temp = read_data(filename, encoding, isSentence=1)
            temp = clean_text(temp)
            data[file_i].append(temp)
    return data


def make_wordvectors(filename, data, numOfFile):
    '''
    generating word2vec Model and save the corpus dictionary file
    if L:num of label, N:num of File
    :param data: lists of sentences, shape is L*N    
    :return: None 
    '''
    from gensim.models import word2vec
    sentences = []
    for label in range(len(data)):
        for EachFile in range(numOfFile):
            EachSentence = data[label][EachFile]
            ListOfWords = tokenize(EachSentence)
            #print(ListOfWords)
            sentences.append(ListOfWords)
    model = word2vec.Word2Vec(sentences, min_count=1)
    model.save(filename)








