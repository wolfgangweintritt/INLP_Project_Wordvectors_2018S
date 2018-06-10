#!/usr/bin/env python3

from gensim.models import FastText, Word2Vec
from gensim.models.word2vec import LineSentence
import argparse
import datetime
import gensim.corpora
import glob
import multiprocessing
import os.path

def vprint(*args, **kwargs):
    if verbose:
        print(datetime.datetime.now().isoformat(' ', timespec="seconds"), end=": ")
        print(*args, **kwargs)

descr = "Introduction to Natural Language Processing, Project 2"
epi   = "by Maximilian Moser (01326252) and Wolfgang Weintritt (01327191)"
ap = argparse.ArgumentParser(description=descr, epilog=epi)
ap.add_argument("--wiki-dump-file", "-w", help="Location of the Wikipedia Dump file", default="dumps/enwiki-*-pages-articles.xml.bz2")
ap.add_argument("--verbose", "-v", help="Enable verbose output", action="store_true")
ap.add_argument("--use-cache", "-c", help="Skip the Corpus Creation and use dumped cache file (base directory: out/)")

args      = ap.parse_args()
wiki_file = args.wiki_dump_file
verbose   = args.verbose
cache     = args.use_cache

if not cache or not os.path.exists(cache):
    if not glob.glob(wiki_file):
        print("Could not find Wiki Dump: %s" % wiki_file)
        exit(1)
    else:
        # take the last matching file
        wiki_file = glob.glob(wiki_file)[-1]
        vprint("Using Wiki Dump: %s" % wiki_file)

    vprint("Creating Corpus from Wiki dump...")
    # parse the wiki dump file into a corpus
    wiki = gensim.corpora.wikicorpus.WikiCorpus(wiki_file, lemmatize=False, dictionary={})
    vprint("Done.")

    # in the following guide, the wiki object was serialized text by text
    # with each text in its own line, without punctuation
    # (LineSentences(file) reads this file and creates a list of lists of words -> iterable of iterables;
    #  the same as wiki.get_texts())
    # https://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
    with open("out/cache.txt", "w") as dump_file:
        vprint("Dumping the Corpus Cache to 'out/cache.txt'...")
        # save the created corpus for good measure
        # (don't want the script die on me and lose all progress)
        for text in wiki.get_texts():
            dump_file.write(b" ".join(text).decode("utf-8") + "\n")
        vprint("Done.")

    # we can use the corpus directly
    inp = wiki

else:
    # if the cache was specified, use it
    vprint("Using cached corpus: %s" % cache)
    inp = LineSentence(cache)

vprint("Training Word2Vec...")
model = Word2Vec(inp, size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
vprint("Done.")