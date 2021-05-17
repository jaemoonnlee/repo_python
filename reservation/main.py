import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait


def conn_test():
    target_url = 'http://www.oceanhills.com/index_yeongcheon.asp'

    # open browser
    driver = webdriver.Edge(executable_path='utils/ms/edge/msedgedriver')
    # open target webpage
    driver.get(url=target_url)
    # 암묵적 대기
    # driver.implicitly_wait(time_to_wait=5)
    # 명시적 대기
    idx = 1
    try:
        print(idx)
        idx += 1
        element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'gLFyf'))
        )
    finally:
        driver.quit()

    print(driver.current_url)

    # close browser
    driver.close()


# main method
if __name__ == '__main__':
    conn_test()
