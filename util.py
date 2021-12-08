def load_links():
    with open("links.txt", "r") as f:
        links = f.readlines()
    return [s.strip() for s in links]


def title(link):
    return link.split("/")[-1]


BASE_URL = "https://hinglishpedia.blogspot.com"
