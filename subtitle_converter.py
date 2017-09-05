import re
import csv
import numpy as np
import konlpy
import json

def clean_text(text):
    cond = "([a-z])|([&n])|<[^>]+>" # matching with eng song, <w*>, &w* part
    p = re.compile(cond, re.I)
    m = p.search(text)
    if m:
        text = ""
        return text
    elif len(text)==0:
        text=""
        return text
    else:
        return text

#1. 질문 응답 셋을 만들어야한다. ? 나오면 질문, 그 다음 . or ! 나올때까지 받아놓고 응답으로 본다. (이후는 수작업가자.)
def extract_conversations(data):
    OQ=[]
    OA=[]

    last_query = ""
    answer = ""
    for line in data:
        #answer이 비었을때만 last_query 넣는다. ? 랑 !가 연속이면 저장해봄. 문제점1. !가 저장되버림 문제점2. 서로 물음표 쓰면서 대화하고 있을 때. 문제점3. 서로 마침표 쓰면서 대화할 때.
        #?를 포함하면 바로 선택, 바로 그 다음 응답만 선택
        pattern = "([/?]+)"
        p = re.compile(pattern)
        m = p.search(line)
        if m and len(answer) == 0:
            if len(last_query) == 0:
                last_query = line
            else:
                last_query += " " + line
        elif len(last_query) != 0 and len(answer) != 0 and m:
            #conversations.append((last_query, answer))
            OQ.append(last_query)
            OA.append(answer)
            last_query = line
            answer = ""
        else:
            if len(last_query) != 0 and m is None and len(answer) == 0:
                answer = line
            elif len(last_query) != 0 and m is None and len(answer) != 0:
                answer += " " + line
    return (OQ, OA)

#2. 같은 사람이 이어서 말한건데 자막이 줄을 바꿨으면?  모든 라인을 돌면서, 마침표('.')가 없이 마치면 잘린걸로 보고 다음 line을 그 앞 line에 붙인다.
#근데 .로 마치거나 -로 마치면 정상적으로 넘어간걸로 본다. (문제점 : !로 이어지더라도 잘라버림.)
def attach_sameturn(data):
    '''    
    :param data: list of data 
    :return: revised caption 
    '''
    result = []
    temp =""
    for line in data:
        if line[-1] == '.' or line[-1]=='-' or line[-1]=='?' or line[-1]=='!':
            if len(temp) != 0:
                temp += " "+line
                result.append(temp)
                temp = ""
            else :
                result.append(line)
        else :
            if len(temp) != 0:
                temp += " " + line
            else :
                temp += line
    return result

#3. 너무 긴 질문이나 응답은 세트로 날리기(마지막)
def QA_cutter(query, answer):
    remove_query_list = []
    remove_answer_list = []
    for oq, oa, in zip(query, answer):
        if len(oq) > 40 or len(oa) > 40 or len(oq) < 5: # 짧은거에 대해서는 질문만 고려
            remove_query_list.append(oq)
            remove_answer_list.append(oa)

    for rem_oq, rem_oa in zip(remove_query_list, remove_answer_list):
        query.remove(rem_oq)
        answer.remove(rem_oa)
    return (query, answer)

#자막 내 인물 이름, 특수지명 등을 다른걸로 대체한다.
def noun_fetcher(datas, targets, substitutions):
    for target, substitution in zip(targets, substitutions):
        pattern = target
        p = re.compile(pattern)
        for iter in range(len(datas)):
            try:
                target_ = p.search(datas[iter]).group()
            except AttributeError as ae:
                target_ = None
            except Exception as e:
                print(e)
            if target_ == target:  # maching!
                new = p.sub(substitution, datas[iter])
                datas[iter] = new
            elif target_ is None:
                continue
            else:
                print("check error", iter, datas[iter])
                break
    return datas

def subtitle_converter(input_filename, output_filename):
    f = open(input_filename, 'r')

    #getting data preprocessed
    data = []
    for line in f.read().splitlines():
        line = clean_text(line)
        if len(line)==0 or line==None:
            continue
        #print(line)
        data.append(line)

    result = attach_sameturn(data)

    # separating into two parts (OQ, OA)
    (query, answer) = extract_conversations(result)
    #여기서 길이가 너무 길거나 짧으면 제외 (아, 그래요? 네? 와 같이 자막 상에서 나름의 문맥상 진행되는 연속대화를 데이터로 뽑아선 X)
    (query, answer) = QA_cutter(query, answer)

    # 붙으면 대명사(받침있으면 그 녀석 없으면 걔), 공백으로 둘러싸이면 호출로 생각
    targets=['천송이', '천송아', '송이']
    substitutions=['걔', '야', '걔']
    #여기서 특수인명, 지명에 대한 정보를 대체해야한다. 따로 메소드 만들고, 이 메소드는 대체 정보에 대한 딕셔너리를 파라미터로 받아야 할거다. 대체 작업은 regex사용.
    query = noun_fetcher(query, targets, substitutions)
    answer = noun_fetcher(answer, targets, substitutions)
    #이후 저장해서 적절하지 않은건 수작업 제거(연속?에 대한 문제-앞이 의도인지 뒤가 의도인지 모호함, 3인 이상대화 포함문제, 대화중 제3인이 끼어드는 문제)


    with open(output_filename, 'w', encoding='cp949', newline='') as f:
        writer = csv.writer(f)
        for oq, oa in zip(query, answer):
            cont = [oq, oa]
            writer.writerow(cont)


def createDictFromSentences(datas):
    temp=[]
    for sent in datas:
        temp.extend(sent)
    dict = {c:i for i, c in enumerate(temp)}
    return dict

#그리고 disintegrator를 따로 만들어야할 듯. 거기서 다시 로딩
#이후 FQ, FA를 만들어야한다. (서버에서 유지할 리소스로서 전처리된 리소스를 가지고 있어야 할 듯 하다, 즉 load_data는 리소스를 가져오고, 임베딩하고, train, test 데이터로 구분해서 주는 작업 정도만 포함해야할 것이다.)
def disintegrator(sentences):
    '''    
    :param sentence: list of sentences to be trimmed
    :return: FS, OS In order to implement Grammar Generator
    '''
    original_sentences = [] # GG를 위한 톤 보존
    disintegrated_sentences = [] # BE를 위한 톤, 조사 삭제
    for sentence in sentences:
        original_sentences.append(['/'.join(tags) for tags in konlpy.tag.Twitter().pos(sentence)])
        disintegrated_sentences.append(['/'.join(tags) for tags in konlpy.tag.Twitter().pos(sentence, norm=True, stem=True)])

    inputData=[]
    for sent in disintegrated_sentences:
        result = []
        for token in sent:
            if 'Eomi' not in token and 'Josa' not in token and 'Number'not in token and 'KoreanParticle'not in token and 'Punctuation' not in token:
                result.append(token)
        inputData.append(result)

    outputData = []
    for sent in original_sentences:
        result=[]
        for token in sent:
            if 'Number' not in token and 'Punctuation' not in token:
                result.append(token)
        outputData.append(result)
    return (inputData, outputData)


def get_emotion_score(sentence):
    return None

# FQ, FA 잘린거로부터 가져와서 train, test 나눠서 반환할 메소드(근데 이건 load_data()에서 받아서 해도 된다.
# 한편 부경대 팀이 Sentence Generator를 만들려면 FA에 대한 OA도 필요하다.X FS에 대한 OS가 있으면 된다. For Grammar Generator
# 가장 중요한 것은 전달할 파라미터를 맞추는 것이다, Set이다. (V차원의 단어벡터(FQ), 각 단어의 감정 스코어, 각 단어의 4차원 스코어, FA, FS, OS) 4개는 Bot Engine, 2개는 Grammar Generator
def load_n_embedding(filename, save=0):

    # 수작업 전처리된 CSV 로딩
    original_query = []
    original_answer=[]
    with open(filename, 'r', encoding='cp949') as f:
        datareader = csv.reader(f, delimiter=',')
        for row in datareader:
            original_query.append(row[0])
            original_answer.append(row[1])

    # Bot Engine을 위한 데이터 : oq_input, oa_input
    (fq_input, oq_output) = disintegrator(original_query)
    (fa_input, oa_output) = disintegrator(original_answer)

    # Grammar Generator를 위한 데이터
    inputdata_forGG = fq_input + fa_input
    outputdata_forGG = oq_output + oq_output

    if save==1:
        #각 단어를 임베딩해서 저장해야함.
        Dict_forBE = createDictFromSentences(inputdata_forGG)
        with open("dict_for_be.txt", 'w') as f:
            json.dump(Dict_forBE, f)

        Dict_forGG = createDictFromSentences(inputdata_forGG+outputdata_forGG)
        with open("dict_for_gg.txt", 'w') as f:
            json.dump(Dict_forGG, f)

    return (fq_input, fa_input, inputdata_forGG, outputdata_forGG)

def beEmbedded(sentences, dict):
    result=[]
    for sent in sentences:
        embeddedsent = []
        for token in sent:
            embeddedsent.append(dict[token])
        result.append(embeddedsent)
    return result

# 최종적으로 load_data는 적절한 질의응답이 결정되고, 이들로부터 임베딩된 단어들을 반환해줘야한다. 그리고 감정 스코어도 저장된 파일로부터 산출해서 반환한다.
def load_data(QAfile, BEdictfile, GGdictfile):
    FQ, FA, inputdata_forGG, outputdata_forGG = load_n_embedding(QAfile)

    with open(BEdictfile) as f:
        BEDict = json.load(f)
        embedded_FQ = beEmbedded(FQ, BEDict)
        embedded_FA = beEmbedded(FA, BEDict)

    with open(GGdictfile) as f:
        GGDict = json.load(f)
        embedded_input_forGG = beEmbedded(inputdata_forGG, GGDict)
        embedded_output_forGG = beEmbedded(outputdata_forGG, GGDict)

    # 각 단어에 대한 감정 산출 ** 이 부분이 이야기 되어야한다. 각 단어별 감정 스코어 갈지, 문장의 감정 스코어 갈지. 그리고 4차원 감정 스코어는 각 단어에 대해서 넣을지.(대표값 선택은 최대값을 선택해야할 것 임.)
    # 감정 스코어는 FQ 문장에 대한 감정 스코어를 가져와야한다.
    # get_emotion_score(sentence)가 문장에 대한 스코어를 주면 될 듯 하다.

    return (embedded_FQ, embedded_FA, embedded_input_forGG, embedded_output_forGG)

#subtitle_converter("1.txt", "data.csv")
#load_n_embedding("data.csv", save=1)
#(embedded_FQ, embedded_FA, embedded_input_forGG, embedded_output_forGG) = load_data("data.csv", "dict_for_be.txt","dict_for_gg.txt")
#print(len(embedded_FQ))
#print(embedded_FQ)

'''
필요한 라이브러리
Numpy, koNLPy
작업 시나리오
1. *.smi 파일을 모조리 가져와서 하나의 txt파일로 만들어라.
2. subtitle_converter로 그걸 가져와서 어느정도는 프로그램으로 전처리한다. 이후 csv로 저장(타임라인 제거, 태그제거, 영어 노래 제거, ?가 연속일때 포함해서 쿼리로 처리, 그 다음 문장을 응답으로 처리, 특수한 고유명사에 대해서 대체어로 대체, 너무 긴 질의응답 제거)     
3. 수작업으로 나머지 전처리한다.(어색한 질의응답 제거, 파악하지 못한 특수 인명 사용 여부)
4. load_n_embedding으로 위 3번 작업후 csv파일 로딩하고 Bot Engine과 Grammar Engine에 필요한 데이터로 각각 나누고 적당히 자른다음 태그를 붙여서 원핫으로 임베딩한다. 그리고 딕셔너리를 저장한다.
5. load_data로 딕셔너리만 가져온다. 딕셔너리는 Bot Engine과 Grammar Engine을 위한 각각의 train, test용 데이터를 적용직전 상태로 저장되어있다.

Bot Engine 개발 팀은 해당 파일을 임포트하고 딕셔너리 파일 받아서 load_data()만 호출하면된다.
그러기 위해서 패딩이 들어가야한다. 이를 위해서는 봇 엔진팀과의 상의가 필요하다. 패딩은 뭘로 넣어야하는가. seq2seq 명세를 다루며 이야기가 되어야할 것이다. 

다만 Bot Engine에서 seq2seq 구현시 사용할 플래그(e.g <GO>,<EOS> etc)가 필요하다면 필요에따라 딕셔너리에 추가해야한다. 

'''