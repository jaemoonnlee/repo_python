from selenium import webdriver
from selenium.webdriver.common.keys import Keys


def conn_browser(path):
    driver = webdriver.Edge(executable_path=path)
    return driver


def open_browser(driver, target):
    driver.get(url=target)


def close_browser(driver):
    if driver is not None:
        driver.close()


def log_in(driver, param_id, param_pw):
    while True:
        try:
            button_login = driver.find_element_by_xpath('//*[@id="header"]/div[1]/div/a[1]')
            if button_login is not None:
                button_login.click()
                break
        except Exception:
            print(Exception)
            continue
    # id tag
    while True:
        try:
            input_id = driver.find_element_by_xpath('//*[@id="log_id"]')
            if input_id is not None:
                break
        except Exception:
            print(Exception)
            continue
    # pw tag
    while True:
        try:
            input_pw = driver.find_element_by_xpath('//*[@id="login_pw"]')
            if input_pw is not None:
                break
        except Exception:
            print(Exception)
            continue
    # type id, pw then press enter
    input_id.send_keys(param_id)
    input_pw.send_keys(param_pw)
    input_pw.send_keys(Keys.ENTER)


def log_out(driver):
    while True:
        try:
            button_logout = driver.find_element_by_xpath('//*[@id="header"]/div[1]/div/a[1]')
            if button_logout is not None:
                button_logout.click()
                break
        except:
            continue


def into_tab_reservation(driver):
    while True:
        try:
            tab_yeongcheon = driver.find_element_by_xpath('//*[@id="cm_reservation"]/ul/li[1]/a/img')
            if tab_yeongcheon is not None:
                tab_yeongcheon.click()
                break
        except Exception:
            print(Exception)
            continue


def find_date(driver, target='20210530', today='20210511'):
    # set calendar(?)
    # calendar_change(?,1번 달력,2번 달력,오늘)
    # calendar_change('yyyyMM','yyyyMM','yyyyMM','yyyyMMdd')
    cur_yyyyMM = today[:6]
    target_yyyyMM = target[:6]
    target_dd = target[6:]
    # driver.execute_script(
    #     "calendar_change('" + cur_yyyyMM + "','" + cur_yyyyMM + "','" + target_yyyyMM + "','" + today + "')")

    # calculate date
    week_idx = 1  # 1-6
    day_idx = 7  # 1-7
    path_base = ''
    if cur_yyyyMM == target_yyyyMM:
        path_base = '//*[@id="calendar_view_ajax_1"]/div/div[3]/table/tbody/'
    elif cur_yyyyMM != target_yyyyMM:
        path_base = '//*[@id="calendar_view_ajax_2"]/div/div[3]/table/tbody/'
    target_path = path_base + 'tr[' + str(week_idx) + ']/td[' + str(day_idx) + ']//a'
    print(target_path)
    firstWeekLastDay = driver.find_element_by_xpath(target_path)
    gap = int(target_dd) - int(firstWeekLastDay.text)
    # ex1) 20210530(target), 20210511(open), 1(firstWeekLastDay)
    # gap = 30 - 1 = 29
    # ex2) 20210606(target), 20210518(open), 5(firstWeekLastDay)
    # gap = 6 - 5 = 1
    # ex2) 20210601(target), 20210518(open), 5(firstWeekLastDay)
    # gap = 1 - 5 = -4
    print(target_dd + "-" + firstWeekLastDay.text + "=" + str(gap))
    if gap == 0:
        firstWeekLastDay.click()
    else:
        if gap > 0:
            week_idx += gap // 7
            day_idx = gap % 7
            if day_idx > 0:
                week_idx += 1
            target_path = path_base + 'tr[' + str(week_idx) + ']/td[' + str(day_idx) + ']//a'
            target_element = driver.find_element_by_xpath(target_path)
        elif gap < 0:
            day_idx += gap
            target_path = path_base + 'tr[1]/td[' + str(day_idx) + ']//a'
            target_element = driver.find_element_by_xpath(target_path)
        target_element.click()
