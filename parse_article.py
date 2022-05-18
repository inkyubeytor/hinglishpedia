from bs4 import BeautifulSoup
from util import load_links, title

LINKS = load_links()


for link in LINKS:
    t = title(link)
    with open(f"data_raw/{t}", "r") as f:
        soup = BeautifulSoup(f.read(), "lxml")
        text = soup.find("div", class_="post-body").text
    with open(f"data_clean/{t}", "w+") as g:
        g.write(text)
