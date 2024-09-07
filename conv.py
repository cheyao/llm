import ebooklib
from ebooklib import epub

book = epub.read_epub('ln/aob-01.epub')

for doc in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
    print doc
