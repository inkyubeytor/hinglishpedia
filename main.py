import os

import requests
from bs4 import BeautifulSoup

import time

from util import BASE_URL, load_links, title


def make_url(year, month):
    return f"{BASE_URL}/{year}/{month:02}"


def get_links():
    links = set()
    for year in range(2010, 2022):
        for month in range(1, 13):
            time.sleep(5)
            response = requests.get(make_url(year, month))
            soup = BeautifulSoup(response.content, "lxml")
            entries = soup.find_all("h2", class_="post-title")
            links |= {e.a["href"] for e in entries}
    with open("links.txt", "w+") as f:
        f.write("\n".join(links))


def scrape_page(link):
    raw = requests.get(link).content
    t = title(link)
    with open(f"data_raw/{t}", "wb+") as f:
        f.write(raw)


def scrape_pages():
    already_have = set(os.listdir("data_raw"))
    for link in load_links():
        t = title(link)
        if t not in already_have:
            time.sleep(5)
            scrape_page(link)


if __name__ == "__main__":
    # get_links()
    scrape_pages()
