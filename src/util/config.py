from typing import List
import json

class Topic:
    """A topic containing words"""
    def __init__(self, name: str, words: List):
        self.name  = name
        self.words = words


class Config:
    """Class for reading and parsing configurations"""
    def __init__(self, config_file: str):
        if not config_file.startswith("config/"):
            config_file = "config/" + config_file

        with open(config_file, "r") as cfg_file:
            tmp = json.load(cfg_file)

        self.word2vec = tmp["word2vec"]
        self.topics = []

        self.words = {}
        self.overlapping_words = []

        for topic in tmp["topics"]:
            # build the topics
            tpc = Topic(topic["name"], topic["words"])
            self.topics.append(tpc)

            for word in tpc.words:
                # find the overlapping words
                if word not in self.words:
                    self.words[word] = []
                else:
                    self.overlapping_words.append(word)

                self.words[word].append(tpc.name)
        
        # filter out the overlapping words
        self.non_overlapping_words = self.words.copy()
        for word in self.overlapping_words:
            del self.non_overlapping_words[word]
        
        # make the lists into single values
        for word in self.non_overlapping_words:
            self.non_overlapping_words[word] = self.non_overlapping_words[word][0]
        
    def topic(self, name: str):
        """Fetch the topic with the specified name"""
        for tpc in self.topics:
            if tpc.name == name:
                return tpc
        
        return None