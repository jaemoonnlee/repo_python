from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


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
            button_login = driver.find_element_by_xpath(
                '//*[@id="header"]/div[1]/div/a[1]'
            )
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
    try:
        WebDriverWait(driver, 3).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        alert.accept()
        print("Login")
    except Exception:
        print("No alert")


def log_out(driver):
    while True:
        try:
            button_logout = driver.find_element_by_xpath(
                '//*[@id="header"]/div[1]/div/a[1]'
            )
            if button_logout is not None:
                button_logout.click()
                break
        except Exception:
            continue
    try:
        WebDriverWait(driver, 3).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        alert.accept()
        print("Logout")
    except Exception:
        print("No alert")


def into_tab_reservation(driver):
    while True:
        try:
            tab_yeongcheon = driver.find_element_by_xpath(
                '//*[@id="cm_reservation"]/ul/li[1]/a/img'
            )
            if tab_yeongcheon is not None:
                tab_yeongcheon.click()
                break
        except Exception:
            print(Exception)
            continue


def find_date(driver, target, today):
    cur_yyyyMM = today[:6]
    target_yyyyMM = target[:6]
    target_dd = target[6:]

    # calculate date
    week_idx = 1  # 1-6
    day_idx = 7  # 1-7
    path_base = ""
    if cur_yyyyMM == target_yyyyMM:
        path_base = '//*[@id="calendar_view_ajax_1"]/div/div[3]/table/tbody/'
    elif cur_yyyyMM != target_yyyyMM:
        path_base = '//*[@id="calendar_view_ajax_2"]/div/div[3]/table/tbody/'
    target_path = path_base + "tr[" + str(week_idx) + "]/td[" + str(day_idx) + "]//a"
    # print(target_path)
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
            target_path = (
                path_base + "tr[" + str(week_idx) + "]/td[" + str(day_idx) + "]//a"
            )
            target_element = driver.find_element_by_xpath(target_path)
        elif gap < 0:
            day_idx += gap
            target_path = path_base + "tr[1]/td[" + str(day_idx) + "]//a"
            target_element = driver.find_element_by_xpath(target_path)
        target_element.click()


def find_term(driver, tm):
    # ex) tm = '0710'
    tm = "1410"
    hh24 = tm[:2]

    # 1. 시간대 선택(1부, 2부)
    # term_path = '//*[@id="cm_time_live"]/div[2]/div[1]/*[@class="yc"]'  # 1부 경로
    # term_path = '//*[@id="cm_time_live"]/div[2]/div[2]/*[@class="yc"]'  # 2부 경로
    # 1부, 2부 상관없이 수집
    term_path = '//*[@id="cm_time_live"]/div[2]//*[@class="yc"]'
    try:
        # term_arr: 1, 2부를 합쳐 현재 선택 가능한 모든 옵션
        term_arr = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.XPATH, term_path))
        )
    except Exception:
        print(Exception)
    # 0 <= len(term_arr) <= 3
    if len(term_arr) == 0:
        print("해당 날짜 예약 불가능")
        return

    checker = False
    for a in term_arr:
        print(a.text)
        if a.text[:2] == hh24:
            checker = True
            a.click()
    if not checker:
        for a in term_arr:
            gap = int(a.text[:2]) - int(hh24)
            if gap == 1:
                a.click()
                break
            if gap == -1:
                a.click()
                break
            if gap == -2:
                a.click()
                break
            if gap == 2:
                a.click()
                break
            if gap == -3:
                a.click()
                break
            if gap == -3:
                a.click()
                break
        # 1부: 06, 07, 08
        # 2부: 11, 12, 13, 14
        term_arr[0].click()  # TODO 이건 그냥 제일 빠른거...

    # 2. 상세 시간 선택
    theIdx = 0
    time_path = (
        '//*[@id="timelist_course_ajax"]/div[2]/table/tbody//*[@class="cm_tlrks"]'
    )
    btn_path = (
        '//*[@id="timelist_course_ajax"]/div[2]/table/tbody//*[@class="cm_dPdir"]/a'
    )
    # table/tbody/tr[theIdx]/td[4]/a
    # the_btn = btn_reserve[theIdx]
    while True:
        try:
            time_arr = driver.find_elements_by_xpath(time_path)
            if time_arr is not None:
                break
        except Exception:
            print(Exception)
            continue
    while True:
        try:
            btn_arr = driver.find_elements_by_xpath(btn_path)
            if btn_arr is not None:
                break
        except Exception:
            print(Exception)
            continue
    print(len(btn_arr))
    # 0 <= len(time_arr) <= ?
    for t in time_arr:
        theIdx += 1
        print(theIdx, t.text, btn_arr[theIdx - 1].text)
    btn_arr[4].click()
    return 0
    # TODO 예약 버튼 누른 뒤 아래 코드 실행
    driver.execute_script("javascript:subcmd1('R')")
    try:
        WebDriverWait(driver, 3).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        alert.accept()
        print("Confirm reservation")
    except Exception:
        print("No alert")
