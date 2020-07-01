import requests
from bs4 import BeautifulSoup


def print_stock_price(code, page_num):
    result = [[], [], [], [], [], [], [], [], []]

    for n in range(page_num):
        url = 'https://finance.naver.com/item/sise_day.nhn?code=' + code + '&page=' + str(n + 1)
        # print(url)

        r = requests.get(url)

        if not r.ok:
            print('Not more data !')
            break

        html = r.content
        soup = BeautifulSoup(html, 'html.parser')
        tr = soup.select('table > tr')

        for i in range(1, len(tr) - 1):
            td = tr[i].select('td')
            if td[0].text.strip():
                result[0].append(td[0].text.strip())  # 날짜
                result[1].append(td[1].text.strip())  # 종가

                img = td[2].select('img')
                if len(img) != 0:
                    if 'src' in img[0].attrs:
                        src = img[0]['src']
                        if 'up' in src:
                            result[2].append('상승')
                        else:
                            result[2].append('하락')
                else:
                    result[2].append('보합')

                result[3].append(td[2].text.strip())  # 전일대비
                result[4].append(td[3].text.strip())  # 시장가
                result[5].append(td[4].text.strip())  # 최고가
                result[6].append(td[5].text.strip())  # 최저가
                result[7].append(td[6].text.strip())  # 거래량

    for i in range(len(result[0])):
        #     날짜          종가             시가         최고가        최저가        거래량
        print(result[0][i], result[1][i], result[4][i], result[5][i], result[6][i],
              result[7][i])


for _ in range(1, 100):
    print_stock_price(code='054210', page_num=_)

