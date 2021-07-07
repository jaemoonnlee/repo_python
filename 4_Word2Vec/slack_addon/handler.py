# handler.py
# 슬랙과 통신하는 모듈을 정의하는 코드

import os

# pip install python-dotenv
from dotenv import load_dotenv 

# pip install slack_sdk
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# 클래스 생성
class SlackHandler:
	token: str
	client: WebClient

	# default constructor
	def __init__(self):
		load_dotenv(verbose=True) # .env 파일에서 환경변수 읽어옴
		self.token = os.getenv('SLACK_TOKEN') # SLACK_TOKEN이라는 환경변수를 token으로 지정
		self.client = WebClient(self.token) # 해당 토큰으로 slack과 통신하는 웹클라이언트 생성

	# 1. 지정된 채널에 메시지를 보내는 메소드
	def post_slack_message(self, channel: str, message: str):
		try:
			response = self.client.chat_postMessage(channel=channel, text=message)
			print(response)
		except SlackApiError as e:
			assert e.response["ok"] is False
			assert e.response["error"]
			print(f"ERROR: {e.response['error']}")

slack_handler = SlackHandler()

# 메인함수
if __name__ == '__main__':
	print("ERROR::직접호출금지.")