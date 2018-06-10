#!/usr/bin/env python3

from gensim.models import FastText, Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.decomposition import PCA
from util.config import Config
import argparse
import datetime
import gensim.corpora
import glob
import multiprocessing
import os.path

def vprint(*args, **kwargs):
    # logging function that is only enabled when the verbose flag is given
    if verbose:
        print("%s" % datetime.datetime.now().isoformat(' ', timespec="seconds"), end=": ")
        print(*args, **kwargs)

descr = "Introduction to Natural Language Processing, Project 2"
epi   = "by Maximilian Moser (01326252) and Wolfgang Weintritt (01327191)"

ap = argparse.ArgumentParser(description=descr, epilog=epi)
ap.add_argument("--wiki-dump-file", "-w", help="Location of the Wikipedia Dump file", default="dumps/enwiki-*-pages-articles.xml.bz2")
ap.add_argument("--verbose", "-v", help="Enable verbose output", action="store_true")
ap.add_argument("--use-cache", "-c", help="Skip the Corpus Creation and use dumped cache file (base directory: out/)")
ap.add_argument("--cache-output", "-o", help="File name for the cache ouptput (base directory: out/)", default="cache.txt")
ap.add_argument("--word-vector-output", "-O", help="File name for the saved Word Vector model (base directory: model/)", default="word2vec")
ap.add_argument("--word-vector-model", "-m", help="File name for the Word Vectors to load instead of train")
ap.add_argument("--config", "-C", help="Configuration file to use (base directory: config/)", default="default")

args         = ap.parse_args()
wiki_file    = args.wiki_dump_file
verbose      = args.verbose
cache        = args.use_cache
cache_output = args.cache_output
model_input  = args.word_vector_model
model_output = args.word_vector_output
config       = args.config

# make the variables usable
if not model_output.startswith("model/"):
    model_output = "model/" + model_output

if cache is not None and not cache.startswith("out/"):
    cache = "out/" + cache

if model_input is not None and not model_input.startswith("model/"):
    model_input = "model/" + model_input

if not config.startswith("config/"):
    config = "config/" + config

if not cache or not os.path.exists(cache):
    # if no corpus cache was specified, build one
    if not glob.glob(wiki_file):
        print("Could not find Wiki Dump: %s" % wiki_file)
        exit(1)
    else:
        # take the last matching file
        wiki_file = glob.glob(wiki_file)[-1]

    # parse the wiki dump file into a corpus
    vprint("Creating Corpus from Wiki Dump '%s'..." % wiki_file)
    wiki = gensim.corpora.wikicorpus.WikiCorpus(wiki_file, lemmatize=False, dictionary={})
    vprint("Done.")

    # in the following guide, the wiki object was serialized text by text
    # with each text in its own line, without punctuation
    # (LineSentences(file) reads this file and creates a list of lists of words -> iterable of iterables;
    #  the same as wiki.get_texts())
    # https://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
    cache_name = "out/%s" % cache_output
    with open(cache_name, "w") as dump_file:
        vprint("Dumping the Corpus Cache to '%s'..." % cache_name)
        # save the created corpus for good measure
        # (don't want the script die on me and lose all progress)
        for text in wiki.get_texts():
            dump_file.write(" ".join(text) + "\n")
        vprint("Done.")

    # we can use the corpus directly
    inp = wiki

else:
    # if the cache was specified, use it
    vprint("Loading cached corpus: %s" % cache)
    inp = LineSentence(cache, max_sentence_length=100000)
    vprint("Done.")

if model_input is None or not os.path.exists(model_input):
    # if no input model is supplied, we have to train it...
    vprint("Training Word2Vec...")
    model = Word2Vec(inp, workers=multiprocessing.cpu_count(), **cfg.word2vec)
    vprint("Done.")
else:
    # else, we just fetch the input model
    vprint("Loading Word Vector Model '%s'..." % model_input)
    model = Word2Vec.load(model_input)
    vprint("Done.")

if model_input is None:
    # there is no sense in saving a model which we already read from a file...
    vprint("Saving Word Vector Model to '%s'" % model_output)
    model.save(model_output)
    vprint("Done.")

print(model)
print(model.wv.most_similar("mouse"))
# reduce the vectors to a space where we can plot it
pca = PCA(n_components=2)
