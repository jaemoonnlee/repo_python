import time
import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from yeongcheon.oh72 import conn_browser
from yeongcheon.oh72 import open_browser
from yeongcheon.oh72 import log_in
from yeongcheon.oh72 import log_out
from yeongcheon.oh72 import close_browser
import yeongcheon.oh72

# variables
browser_path = 'utils/ms/edge/msedgedriver'
url = 'http://www.oceanhills.com/login/login.asp?returnurl=/pagesite/reservation/live.asp?'
user_id = 'sh4220'
user_pw = 'tlsgud4220'


def conn_test(date, term):
    y4m2d2 = time.strftime('%Y%m%d', time.localtime())
    driver = yeongcheon.oh72.conn_browser(browser_path)
    yeongcheon.oh72.open_browser(driver, url)
    yeongcheon.oh72.log_in(driver, user_id, user_pw)
    yeongcheon.oh72.into_tab_reservation(driver)

    yeongcheon.oh72.find_date(driver, date, y4m2d2)
    # TODO 날짜 선택까지 오케이... 시간 선택 해야함
    yeongcheon.oh72.find_term(driver, term)
    # //*[@id="cm_reservation_02"] 시간 선택

    yeongcheon.oh72.log_out(driver)
    time.sleep(1)
    yeongcheon.oh72.close_browser(driver)


# main method
if __name__ == '__main__':
    # 날짜(yyyyMMdd)
    # 시간(24hhmm)
    target_date = '20210619'  # 날짜
    target_date = '20210614'
    # target_date = input()
    target_term = '0710'  # 시간
    target_term = '0710'
    # target_term = input()
    conn_test(target_date, target_term)
