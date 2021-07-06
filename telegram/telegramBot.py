from flask import Flask, request

# pip install python-telegram-bot
# pip install telegram
import telegram
import os
from credentials import URL
from dotenv import load_dotenv 

load_dotenv(verbose=True)
TOKEN = os.getenv('bot_token')
bot = telegram.Bot(token=TOKEN)

app = Flask(__name__)

# 텔레그램 봇과 url 연동 
@app.route('/setwebhook', methods=['GET', 'POST'])
def set_webhook():
    s = bot.setWebhook(f'{URL}/{TOKEN}')
    if s:
        return "webhook setup ok"
    else:
        return "webhook setup failed"

# 실제로 텔레그램에서 request를 보내는 URL
# 텔레그램의 경우 <URL>/<TOKEN> 으로 http request 전송
@app.route(f'/{TOKEN}', methods=['POST'])
def autoResponse():
	# request를 json 형식으로 파싱
    resp = request.get_json(force=True)
    print(resp)
    if 'message' in resp.keys():
        msgtext = resp["message"]["text"]
        sendername = resp["message"]["from"]["first_name"]
        chat_id = resp["message"]["chat"]["id"]
    elif 'channel_post' in resp.keys():
        msgtext = resp["channel_post"]["text"]
        sendername = resp["channel_post"]["chat"]["username"]
        chat_id = resp["channel_post"]["chat"]["id"]
    try:
        bot.sendMessage(chat_id=chat_id, text=msgtext)
    except Exception:
        # 에러가 발생했을 때
        bot.sendMessage(chat_id=chat_id, text="문제가 발생하였습니다 :(")

    return "ok"

if __name__ == '__main__':
	app.run(debug=True)
