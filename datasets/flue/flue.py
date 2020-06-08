"""FLUE: French Language Understanding Evaluation."""

from __future__ import absolute_import, division, print_function

import csv
import os
import textwrap
import xml.etree.cElementTree as ET

import nlp


_FLUE_CITATION = """\
@misc{le2019flaubert,
    title={FlauBERT: Unsupervised Language Model Pre-training for French},
    author={Hang Le and Loïc Vial and Jibril Frej and Vincent Segonne and Maximin Coavoux and Benjamin Lecouteux and Alexandre Allauzen and Benoît Crabbé and Laurent Besacier and Didier Schwab},
    year={2019},
    eprint={1912.05372},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_FLUE_DESCRIPTION = """\
FLUE is an evaluation setup for French NLP systems similar to the popular GLUE benchmark.
The goal is to enable further reproducible experiments in the future and to share models and progress on the French language.
The tasks and data are obtained from existing works, please refer to our Flaubert paper for a complete list of references.
"""

_CLS_CATEGORIES = ["books", "dvd", "music"]
_FWSD_CATEGORIES = ["wordnet", "semcor"]

_NAMES = ["PAWS-X", "XNLI", "FSE"]
for cat in _CLS_CATEGORIES:
    _NAMES.append(f"CLS.{cat}")
for cat in _FWSD_CATEGORIES:
    _NAMES.append(f"FWSD.{cat}")


_DESCRIPTIONS = {
    "CLS": textwrap.dedent(
        """
        This is a binary classification task.
        It consists in classifying Amazon reviews for three product categories: books, DVD, and music.
        Each sample contains a review text and the associated rating from 1 to 5 stars.
        Reviews rated above 3 is labeled as positive, and those rated less than 3 is labeled as negative.
        The train and test sets are balanced, including around 1k positive and 1k negative reviews for a total of 2k reviews in each dataset.
        """
    ),
    "PAWS-X": textwrap.dedent(
        """
        The task consists in identifying whether the two sentences in a pair are semantically equivalent or not.
        The train set includes 49.4k examples, the dev and test sets each comprises nearly 2k examples.
        """
    ),
    "XNLI": textwrap.dedent(
        """
        The Natural Language Inference (NLI) task, also known as recognizing textual entailment (RTE), \
        is to determine whether a premise entails, contradicts or neither entails nor contradicts a hypothesis.
        We take the French part of the XNLI corpus to form the development and test sets for the NLI task in FLUE.
        The train set includes 392.7k examples, the dev and test sets comprises 2.5k and 5k examples respectively.
        """
    ),
    "FSE": textwrap.dedent(
        """
        FrenchSemEval : An evaluation corpus for French verb disambiguation.
        """
    ),
    "FWSD": textwrap.dedent(
        """
        This is a dataset for the Word Sense Disambiguation of French using Princeton WordNet identifiers.
        It contains two training corpora : the SemCor and the WordNet Gloss Corpus, both automatically translated from their original English version, and with sense tags automatically aligned.
        It contains also a test corpus : the task 12 of SemEval 2013, originally sense annotated with BabelNet identifiers, converted into Princeton WordNet 3.0.
        """
    ),
}

_CITATIONS = {
    "CLS": textwrap.dedent(
        """
        @inproceedings{prettenhofer2010cross,
          title={Cross-language text classification using structural correspondence learning},
          author={Prettenhofer, Peter and Stein, Benno},
          booktitle={Proceedings of the 48th annual meeting of the association for computational linguistics},
          pages={1118--1127},
          year={2010}
        }
        """
    ),
    "PAWS-X": textwrap.dedent(
        """
        @inproceedings{pawsx2019emnlp,
          title = {{PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification}},
          author = {Yang, Yinfei and Zhang, Yuan and Tar, Chris and Baldridge, Jason},
          booktitle = {Proc. of EMNLP},
          year = {2019}
        }
        """
    ),
    "XNLI": textwrap.dedent(
        """
        @inproceedings{conneau2018xnli,
          author = "Conneau, Alexis and Rinott, Ruty and Lample, Guillaume and Williams, Adina and Bowman, Samuel R. and Schwenk, Holger and Stoyanov, Veselin",
          title = "XNLI: Evaluating Cross-lingual Sentence Representations",
          booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
          year = "2018",
          publisher = "Association for Computational Linguistics",
          location = "Brussels, Belgium",
        }
        """
    ),
    "FSE": textwrap.dedent(
        """
        @inproceedings{segonne2019using,
          title={Using Wiktionary as a resource for WSD: the case of French verbs},
          author={Segonne, Vincent and Candito, Marie and Crabb{\'e}, Benoit},
          booktitle={Proceedings of the 13th International Conference on Computational Semantics-Long Papers},
          pages={259--270},
          year={2019}
        }
        """
    ),
    "FWSD": textwrap.dedent(
        """
        @dataset{loic_vial_2019_3549806,
          author = {Loïc Vial},
          title = {{French Word Sense Disambiguation with Princeton
          WordNet Identifiers}},
          month = nov,
          year = 2019,
          publisher = {Zenodo},
          version = {1.0},
          doi = {10.5281/zenodo.3549806},
          url = {https://doi.org/10.5281/zenodo.3549806}
        }
        """
    ),
}

_TEXT_FEATURES = {
    "CLS": ["text", ],
    "PAWS-X": ["sentence1", "sentence2"],
    "XNLI": ["premise", "hypothesis", "label"],
    "FSE": ["label", ],
    "FWSD": [],
}

_DATA_URLS = {
    "CLS": "https://zenodo.org/record/3251672/files/cls-acl10-unprocessed.tar.gz",
    "PAWS-X": "https://storage.googleapis.com/paws/pawsx/x-final.tar.gz",
    "XNLI": "https://dl.fbaipublicfiles.com/XNLI",
    "FSE": "http://www.llf.cnrs.fr/dataset/fse/FSE-1.1-10_12_19.tar.gz",
    "FWSD": "https://zenodo.org/record/3549806/files",
}


class FlueConfig(nlp.BuilderConfig):
    """BuilderConfig for FLUE."""

    def __init__(
        self, citation, data_url, text_features, **kwargs,
    ):
        super(FlueConfig, self).__init__(version=nlp.Version("1.0.0"), **kwargs)
        self.citation = citation
        self.data_url = data_url
        self.text_features = text_features


class Flue(nlp.GeneratorBasedBuilder):
    """FLUE: French Language Understanding Evaluation."""

    BUILDER_CONFIGS = [
        FlueConfig(
            name=name,
            description=_DESCRIPTIONS[name.split(".")[0]],
            citation=_CITATIONS[name.split(".")[0]],
            data_url=_DATA_URLS[name.split(".")[0]],
            text_features=_TEXT_FEATURES[name.split(".")[0]],
        )
        for name in _NAMES
    ]

    def _info(self):
        features = {text_feature: nlp.Value("string") for text_feature in self.config.text_features}
        if self.config.name.startswith("CLS"):
            features["label"] = nlp.features.ClassLabel(names=["neg", "pos"])
        if self.config.name == "PAWS-X":
            features["label"] = nlp.Value("int32")
        if self.config.name == "XNLI":
            features["label"] = nlp.features.ClassLabel(names=["entailment", "neutral", "contradiction"])
        if self.config.name == "FSE":
            features["tokens"] = nlp.Sequence(nlp.Value("string"))
            features["pos"] = nlp.Value("int32")
        if self.config.name.startswith("FWSD"):
            features["tokens"] = nlp.Sequence(nlp.Value("string"))
            features["labels"] = nlp.Sequence(nlp.Value("string"))

        return nlp.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=self.config.description + "\n" + _FLUE_DESCRIPTION,
            # nlp.features.FeatureConnectors
            features=nlp.Features(features),
            supervised_keys=None,
            homepage="https://github.com/getalp/Flaubert/tree/master/flue",
            citation=self.config.citation + "\n" + _FLUE_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.name.startswith("CLS"):
            dl_dir = dl_manager.download_and_extract(self.config.data_url)
            data_dir = os.path.join(dl_dir, "cls-acl10-unprocessed", "fr")
            category = self.config.name.split(".")[1]
            return [
                nlp.SplitGenerator(
                    name=nlp.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, category, "train.review")}
                ),
                nlp.SplitGenerator(
                    name=nlp.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, category, "test.review")}
                ),
            ]

        if self.config.name == "PAWS-X":
            dl_dir = dl_manager.download_and_extract(self.config.data_url)
            data_dir = os.path.join(dl_dir, "x-final", "fr")
            return [
                nlp.SplitGenerator(
                    name=nlp.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, "translated_train.tsv")},
                ),
                nlp.SplitGenerator(
                    name=nlp.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir, "dev_2k.tsv")},
                ),
                nlp.SplitGenerator(
                    name=nlp.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, "test_2k.tsv")},
                ),
            ]

        if self.config.name == "XNLI":
            urls_to_download = {
                "train": os.path.join(self.config.data_url, "XNLI-MT-1.0.zip"),
                "dev_test": os.path.join(self.config.data_url, "XNLI-1.0.zip"),
            }
            dl_dirs = dl_manager.download_and_extract(urls_to_download)
            train_data_dir = os.path.join(dl_dirs["train"], "XNLI-MT-1.0")
            dev_test_data_dir = os.path.join(dl_dirs["dev_test"], "XNLI-1.0")
            return [
                nlp.SplitGenerator(
                    name=nlp.Split.TRAIN,
                    gen_kwargs={"filepath": os.path.join(train_data_dir, "multinli", "multinli.train.fr.tsv")},
                ),
                nlp.SplitGenerator(
                    name=nlp.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(dev_test_data_dir, "xnli.dev.tsv")}
                ),
                nlp.SplitGenerator(
                    name=nlp.Split.TEST, gen_kwargs={"filepath": os.path.join(dev_test_data_dir, "xnli.test.tsv")}
                ),
            ]

        if self.config.name == "FSE":
            dl_dir = dl_manager.download_and_extract(self.config.data_url)
            data_dir = os.path.join(dl_dir, "FSE-1.1-191210")
            return [
                nlp.SplitGenerator(
                    name=nlp.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "wiktionary-190418.data.xml"),
                        "labelpath": os.path.join(data_dir, "wiktionary-190418.gold.key.txt"),
                    },
                ),
                nlp.SplitGenerator(
                    name=nlp.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "FSE-1.1.data.xml"),
                        "labelpath": os.path.join(data_dir, "FSE-1.1.gold.key.txt"),
                    },
                ),
            ]

        if self.config.name.startswith("FWSD"):
            category = self.config.name.split(".")[1]
            train_file = "wngt.fr.xml"
            if category == "semcor":
                train_file = "semcor.fr.xml"
            urls_to_download = {
                "train": os.path.join(self.config.data_url, train_file),
                "test": os.path.join(self.config.data_url, "semeval2013task12.fr.xml"),
            }
            dl_files = dl_manager.download(urls_to_download)
            return [
                nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath": dl_files["train"]}),
                nlp.SplitGenerator(name=nlp.Split.TEST, gen_kwargs={"filepath": dl_files["test"]}),
            ]

    def _generate_examples(self, filepath, labelpath=None):
        if self.config.name.startswith("CLS"):
            idx = 0
            for _, elem in ET.iterparse(filepath):
                if elem.tag == "item":
                    id_ = idx
                    text = elem.findtext("text")
                    rating = float(elem.findtext("rating"))
                    label = "pos" if rating > 3 else "neg"
                    idx += 1
                    yield id_, {"text": text, "label": label}

        if self.config.name == "PAWS-X":
            with open(filepath, mode="r") as f:
                data = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                for id_, row in enumerate(data):
                    yield id_, {
                        "sentence1": row["sentence1"],
                        "sentence2": row["sentence2"],
                        "label": int(row["label"]),
                    }

        if self.config.name == "XNLI":
            with open(filepath, mode="r") as f:
                data = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                headers = data.fieldnames
                # dev & test files
                if "language" in headers:
                    idx = 0
                    for row in data:
                        if row["language"] == "fr":
                            id_ = idx
                            idx += 1
                            yield id_, {
                                "premise": row["sentence1"],
                                "hypothesis": row["sentence2"],
                                "label": row["gold_label"],
                            }
                # train file
                else:
                    for id_, row in enumerate(data):
                        yield id_, {
                            "premise": row["premise"],
                            "hypothesis": row["hypo"],
                            "label": row["label"] if row["label"] != "contradictory" else "contradiction",
                        }

        if self.config.name == "FSE":
            with open(labelpath, mode="r") as label_file:
                reader = csv.reader(label_file, delimiter=" ")
                for _, elem in ET.iterparse(filepath):
                    if elem.tag == "sentence":
                        id_ = elem.get("idx")
                        tokens = []
                        for i, child in enumerate(elem):
                            tokens.append(child.text)
                            if child.tag == "instance":
                                pos = i
                                # label_id = child.get("id")
                                label = next(reader, None)[1]
                        yield id_, {
                            "tokens": tokens,
                            "pos": pos,
                            "label": label,
                        }

        if self.config.name.startswith("FWSD"):
            for _, elem in ET.iterparse(filepath):
                idx = 0
                if elem.tag == "sentence":
                    id_ = idx
                    tokens, labels = [], []
                    for child in elem:
                        tokens.append(child.get("surface_form"))
                        labels.append(child.get("wn30_key") or "O")
                    idx += 1
                    yield id_, {
                        "tokens": tokens,
                        "labels": labels,
                    }
