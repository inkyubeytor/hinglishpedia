# Hinglishpedia
This repository contains the tools needed to scrape the Hinglishpedia website.
The required environment can be built with `hinglishpedia.yml`.

## `main.py`
To scrape the data, use `main.py`.
The `get_links` function will scrape a link index for files that have not been scraped.
The `scrape_pages` function will download the raw articles in HTML format.

## `parse_article.py`
To clean the data, use `parse_article.py`.
This will turn the HTML files into text-format articles.

# Transliteration
This repository also contains the means to train a transliteration model on the Hinglishpedia data.
The environment for these can be created with `transliterate.yml`.

## `transliterate.py`
First, all tokens are read and split from the Hinglishpedia data.
Then, these are passed through the Google Transliteration API to convert Romanized Hindi tokens to Devanagari Hindi tokens.

## `filter_eng.py`
All tokens which were English tokens (and have no Devanagari counterparts) can be filtered out of the mapping with this file.

## `train_transliterate_model.py`
This file can be used to train transliteration models on the data produced by the previous two steps.
The filenames used are the same defaults for the previous two files, but can be modified in `load_dict.py`.

## `transliterator.py`
This file can be used to apply the transliteration model created by the previous step onto text with Devanagari characters.

