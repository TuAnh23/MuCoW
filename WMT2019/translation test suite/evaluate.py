#! /usr/bin/env python3

import sys, glob
from qe_tokenization import perform_tokenization
import argparse
import stanza
from spacy_stanza import StanzaLanguage
import pandas as pd
from utils import load_text_file
from spacy_dummy_tokenizer import WhitespaceTokenizer


def processLanguagePair(lgpair, keyfile_prefix, rawtranslations_glob, output_path):
    """
    params:
        lgpair: translation language direction, e.g., en-de
        keyfile_prefix: prefix of the provided key files
        rawtranslations_glob: raw, detokenized translations
        output_path: path to store the WSD indices and their correct/incorrect labels. a dataframe with
            columns ['Sentence', 'Correct WSD output', 'Wrong WSD words indices']
    """

    # load sense keys from file
    sense_keys = []
    k = load_text_file(keyfile_prefix + ".key.txt")
    for line in k:
        elements = line.strip().split("\t")
        t = (elements[0], elements[1], elements[2], tuple(elements[3].split(" ")), tuple(elements[4].split(" ")))
        sense_keys.append(t)

    # load domain keys from file
    indomain_keys = set()
    outdomain_keys = set()
    d = load_text_file(keyfile_prefix + ".domain.txt")
    for line in d:
        elements = line.strip().split("\t")
        if elements[2] == "in":
            indomain_keys.add((elements[0], elements[1]))
        else:
            outdomain_keys.add((elements[0], elements[1]))

    # load lemmatizer
    snlp = stanza.Pipeline(lang=lgpair[-2:])
    nlp = StanzaLanguage(snlp)
    # Replace the default tokenizer in the pipeline with the dummy tokenizer, since we will use this on
    # pre-tokenized text
    nlp.tokenizer = WhitespaceTokenizer(nlp)

    # load and process submissions
    results = {}
    rawsubmissions = sorted(glob.glob(rawtranslations_glob))
    for rawsubmission in rawsubmissions:
        # Create the df to store the sentence and word level WSD correct/incorrect result
        wsd_labels_df = pd.DataFrame(columns=['Sentence', 'Correct WSD output', 'Wrong WSD words indices'])

        counts = {"pos_in": 0, "pos_out": 0, "neg_in": 0, "neg_out": 0, "unk_in": 0, "unk_out": 0}
        trans_sentences = load_text_file(rawsubmission)
        trans_sentences_tok = perform_tokenization(lang=lgpair[-2:], inlist=trans_sentences)

        wsd_labels_df['Sentence'] = trans_sentences

        for i, (trans_sentence, trans_sentence_tok, key) in \
                enumerate(zip(trans_sentences, trans_sentences_tok, sense_keys)):
            if (key[2], " ".join(key[3])) in indomain_keys:
                suffix = "_in"
            elif (key[2], " ".join(key[3])) in outdomain_keys:
                suffix = "_out"
            else:
                print("Domain not found:", (key[2], " ".join(key[3])))

            # first look in tokenized data
            tokwords = [x.lower() for x in trans_sentence_tok]
            posfound = any([posword in tokwords for posword in key[3]])
            negfound = any([negword in tokwords for negword in key[4]])

            negative_indices = []
            # Store the indices of the negative words in the tokenized sentence
            if negfound:
                for tokword_i, tokword in enumerate(tokwords):
                    if tokword in key[4]:
                        negative_indices.append(tokword_i)

            # if not found, look in lemmatized data
            if (not posfound) and (not negfound):
                posfound = False
                negfound = False

                # Perform lemmatization
                doc = nlp(trans_sentence)
                for token_i, token in enumerate(doc):
                    if token.lemma_.lower() in key[3]:
                        posfound = True
                    if token.lemma_.lower() in key[4]:
                        negfound = True
                        negative_indices.append(token_i)

            if posfound and not negfound:
                counts["pos" + suffix] += 1
                wsd_labels_df['Correct WSD output'].iloc[i] = True
            elif negfound:
                counts["neg" + suffix] += 1
                wsd_labels_df['Correct WSD output'].iloc[i] = False
            else:
                counts["unk" + suffix] += 1
                wsd_labels_df['Correct WSD output'].iloc[i] = None

            wsd_labels_df['Wrong WSD words indices'].iloc[i] = negative_indices

        wsd_labels_df.to_csv(output_path)

        counts["cov_in"] = (counts["pos_in"] + counts["neg_in"]) / (
                    counts["pos_in"] + counts["neg_in"] + counts["unk_in"])
        counts["cov_out"] = (counts["pos_out"] + counts["neg_out"]) / (
                    counts["pos_out"] + counts["neg_out"] + counts["unk_out"])
        counts["cov_all"] = (counts["pos_in"] + counts["neg_in"] + counts["pos_out"] + counts["neg_out"]) / (
                    counts["pos_in"] + counts["neg_in"] + counts["unk_in"] + counts["pos_out"] + counts["neg_out"] +
                    counts["unk_out"])

        # Precision = pos / (pos+neg)
        counts["prec_in"] = 0 if counts["pos_in"] == 0 else counts["pos_in"] / (counts["pos_in"] + counts["neg_in"])
        counts["prec_out"] = 0 if counts["pos_out"] == 0 else counts["pos_out"] / (
                    counts["pos_out"] + counts["neg_out"])
        counts["prec_all"] = 0 if (counts["pos_in"] + counts["pos_out"]) == 0 else (counts["pos_in"] + counts[
            "pos_out"]) / (counts["pos_in"] + counts["neg_in"] + counts["pos_out"] + counts["neg_out"])

        # RecallA = pos / (pos+unk)
        # This is the definition of recall that was used to compute the results tables
        # in the papers, but *does not* correspond to the definition given in the papers.
        counts["recA_in"] = 0 if counts["pos_in"] == 0 else counts["pos_in"] / (counts["pos_in"] + counts["unk_in"])
        counts["recA_out"] = 0 if counts["pos_out"] == 0 else counts["pos_out"] / (
                    counts["pos_out"] + counts["unk_out"])
        counts["recA_all"] = 0 if (counts["pos_in"] + counts["pos_out"]) == 0 else (counts["pos_in"] + counts[
            "pos_out"]) / (counts["pos_in"] + counts["unk_in"] + counts["pos_out"] + counts["unk_out"])

        # RecallB = pos / (pos+unk+neg)
        # This formula corresponds to the definition given in the papers,
        # but is *not* the one that was used to compute the results tables.
        counts["recB_in"] = 0 if counts["pos_in"] == 0 else counts["pos_in"] / (
                    counts["pos_in"] + counts["unk_in"] + counts["neg_in"])
        counts["recB_out"] = 0 if counts["pos_out"] == 0 else counts["pos_out"] / (
                    counts["pos_out"] + counts["unk_out"] + counts["neg_out"])
        counts["recB_all"] = 0 if (counts["pos_in"] + counts["pos_out"]) == 0 else (counts["pos_in"] + counts[
            "pos_out"]) / (counts["pos_in"] + counts["unk_in"] + counts["neg_in"] + counts["pos_out"] + counts[
            "unk_out"] + counts["neg_out"])

        # F1A is based on RecallA
        counts["f1A_in"] = 0 if (counts["prec_in"] + counts["recA_in"]) == 0 else 2 * counts["prec_in"] * counts[
            "recA_in"] / (counts["prec_in"] + counts["recA_in"])
        counts["f1A_out"] = 0 if (counts["prec_out"] + counts["recA_out"]) == 0 else 2 * counts["prec_out"] * counts[
            "recA_out"] / (counts["prec_out"] + counts["recA_out"])
        counts["f1A_all"] = 0 if (counts["prec_all"] + counts["recA_all"]) == 0 else 2 * counts["prec_all"] * counts[
            "recA_all"] / (counts["prec_all"] + counts["recA_all"])

        # F1B is based on RecallB
        counts["f1B_in"] = 0 if (counts["prec_in"] + counts["recB_in"]) == 0 else 2 * counts["prec_in"] * counts[
            "recB_in"] / (counts["prec_in"] + counts["recB_in"])
        counts["f1B_out"] = 0 if (counts["prec_out"] + counts["recB_out"]) == 0 else 2 * counts["prec_out"] * counts[
            "recB_out"] / (counts["prec_out"] + counts["recB_out"])
        counts["f1B_all"] = 0 if (counts["prec_all"] + counts["recB_all"]) == 0 else 2 * counts["prec_all"] * counts[
            "recB_all"] / (counts["prec_all"] + counts["recB_all"])

        submissionName = rawsubmission.split("/")[-1]
        results[submissionName] = counts

    print(lgpair.upper())
    print()
    print(
        "Submission\t\tInPos\tInNeg\tInUnk\tInCoverage\tInPrecision\tInRecallA\tInRecallB\tInFscoreA\tInFscoreB\t"
        "\tOutPos\tOutNeg\tOutUnk\tOutCoverage\tOutPrecision\tOutRecallA\tOutRecallB\tOutFscoreA\tOutFscoreB\t"
        "\tAllPos\tAllNeg\tAllUnk\tAllCoverage\tAllPrecision\tAllRecallA\tAllRecallB\tAllFscoreA\tAllFscoreB")
    for submission, result in sorted(results.items(), key=lambda x: x[1]["f1A_all"], reverse=True):
        s = submission
        s += "\t\t{}\t{}\t{}\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}%".format(result["pos_in"],
                                                                                           result["neg_in"],
                                                                                           result["unk_in"],
                                                                                           100 * result["cov_in"],
                                                                                           100 * result["prec_in"],
                                                                                           100 * result["recA_in"],
                                                                                           100 * result["recB_in"],
                                                                                           100 * result["f1A_in"],
                                                                                           100 * result["f1B_in"])
        s += "\t\t{}\t{}\t{}\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}%".format(result["pos_out"],
                                                                                           result["neg_out"],
                                                                                           result["unk_out"],
                                                                                           100 * result["cov_out"],
                                                                                           100 * result["prec_out"],
                                                                                           100 * result["recA_out"],
                                                                                           100 * result["recB_out"],
                                                                                           100 * result["f1A_out"],
                                                                                           100 * result["f1B_out"])
        s += "\t\t{}\t{}\t{}\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}%".format(
            result["pos_in"] + result["pos_out"], result["neg_in"] + result["neg_out"],
            result["unk_in"] + result["unk_out"], 100 * result["cov_all"], 100 * result["prec_all"],
            100 * result["recA_all"], 100 * result["recB_all"], 100 * result["f1A_all"], 100 * result["f1B_all"])
        print(s)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lgpair', type=str, default='en-de')
    parser.add_argument('--keyfileprefix',
                        type=str,
                        default='txt/en-de',
                        help='path of the *.key.txt and *.domain.txt files')
    parser.add_argument('--rawtranslations', type=str,
                        help='path of the detokenized translation output')
    parser.add_argument('--output_path', type=str,
                        help='path to store the WSD correct/incorrect labels and erroneous token indices')

    args = parser.parse_args()
    print(args)

    processLanguagePair(args.lgpair, args.keyfileprefix, args.rawtranslations, args.output_path)
