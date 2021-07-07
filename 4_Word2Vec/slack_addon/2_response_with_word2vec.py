import json

# pip install Flask
from flask import Flask, request, make_response
from konlpy.tag import Twitter
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from PIL import Image
from io import BytesIO
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from scipy.sparse import data
from sklearn.metrics.pairwise import cosine_similarity
import pickle



####################################################################################
# 코사인 유사도
####################################################################################
df = pd.read_csv("data.csv")
print('전체 문서의 수 :', len(df))


def _removeNonAscii(s):
    return "".join(i for i in s if ord(i) < 128)

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

df['cleaned'] = df['Desc'].apply(_removeNonAscii)
df['cleaned'] = df.cleaned.apply(make_lower_case)
df['cleaned'] = df.cleaned.apply(remove_stop_words)
df['cleaned'] = df.cleaned.apply(remove_punctuation)
df['cleaned'] = df.cleaned.apply(remove_html)
df['cleaned'].replace('', np.nan, inplace=True)
df = df[df['cleaned'].notna()]


corpus = []
for words in df['cleaned']:
    corpus.append(words.split())

# with open('document_embedding_list.pickle', 'rb') as f:
with open('document_embedding_list_full.pickle', 'rb') as f:
    document_embedding_list = pickle.load(f)

cosine_similarities = cosine_similarity(document_embedding_list, document_embedding_list)
print('코사인 유사도 매트릭스의 크기 :', cosine_similarities.shape)

def recommendations(title):
    books = df[['title', 'image_link']]

    # 책의 제목을 입력하면 해당 제목의 인덱스를 리턴받아 idx에 저장.
    indices = pd.Series(df.index, index = df['title']).drop_duplicates()    
    print("HERE ?")
    # TODO 이후 진입 불가?
    # [TITLE]That Will Never Work: The Birth of Netflix and the Amazing Life of an Idea
    # 외에는 아래에서 에러 나는듯?
    idx = indices[title]
    print("3")

    # 입력된 책과 줄거리(document embedding)가 유사한 책 5개 선정.
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    print("sim_scores", sim_scores)

    # 가장 유사한 책 5권의 인덱스
    book_indices = [i[0] for i in sim_scores]

    # 전체 데이터프레임에서 해당 인덱스의 행만 추출. 5개의 행을 가진다.
    recommend = books.iloc[book_indices].reset_index(drop=True)

    # fig = plt.figure(figsize=(20, 30))

    # 데이터프레임으로부터 순차적으로 이미지를 출력
    dataList =[]
    for index, row in recommend.iterrows():
        dataList.append({"title": row["title"], "image_link": row["image_link"]})
    # for x in dataList:
    #     print("recommendations()\n", x)
    return dataList
#         response = requests.get(row['image_link'])
#         img = Image.open(BytesIO(response.content))
#         fig.add_subplot(1, 5, index + 1)
#         plt.imshow(img)
#         plt.title(row['title'])
    # # 데이터프레임으로부터 순차적으로 이미지를 출력
    # for index, row in recommend.iterrows():
    #     response = requests.get(row['image_link'])
    #     img = Image.open(BytesIO(response.content))
    #     fig.add_subplot(1, 5, index + 1)
    #     plt.imshow(img)
    #     plt.title(row['title'])


# recommendations("That Will Never Work: The Birth of Netflix and the Amazing Life of an Idea")

####################################################################################
# 코사인 유사도
####################################################################################

from handler import slack_handler

# 플라스크 인스턴스 생성 
app = Flask(__name__)

@app.route('/', methods=['POST'])
def default_listener():
	# 슬랙에서 보낸 request 데이터를 json으로 파싱한다.
	slack_event = json.loads(request.data)
	print(">> slack_event\n", slack_event)

	# 인자 중 challenge가 있으면 해당 인자의 값을 반환한다.
	# slack api specification. 참고:https://api.slack.com/
	if "challenge" in slack_event:
		return make_response(slack_event["challenge"], 200, {"content_type": "application/json"})

	# slack에서 발생한 event를 통한 request에 대한 핸들링
	if "event" in slack_event:
		event_type = slack_event["event"]["type"]
		# bot_mention일 경우에 대한 핸들링
		if event_type == 'app_mention':
			try:
				# 멘션을 남긴 채널 읽어오기
				channel = slack_event['event']['channel']
				# 유저가 멘션과 함께 남긴 텍스트 읽어오기
				user_query = slack_event['event']['blocks'][0]['elements'][0]['elements'][1]['text']

				####################################################################################

				#								여기에 전처리 등 추가!								   #

				####################################################################################

				try:
					print("user_query\t", user_query)
					print("user_query[1:]\t", user_query[1:])
					recommendList = recommendations(user_query[1:])
					print(recommendList)
					slack_handler.post_slack_message(message= recommendList[0]["title"], channel=channel)
				except Exception as e: 
					print(e)
					slack_handler.post_slack_message(message="Error Occured :(", channel=channel)
				
				# 정상적으로 완료했음에 대한 http response
				return make_response("response made :)", 200, )
			except IndexError:
				# 멘션은 했지만 텍스트는 남기지 않은 경우에 대한 에러.
				# do nothing
				pass
		# 그 외 event에 대한 핸들링: 404 error
		msg = f"[{event_type}] cannot find event handler"
		return make_response(msg, 404, {"X-Slack-No-Retry": 1 })


	# 그 외 request 핸들링: 404 error
	return make_response("No Slack request events", 404, {"X-Slack-No-Retry": 1 })



# 실행
if __name__ == '__main__':
    app.run(debug=True)