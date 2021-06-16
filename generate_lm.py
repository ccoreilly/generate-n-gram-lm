import argparse
import gzip
import io
import os
import sys
import subprocess
from collections import Counter

from tqdm import tqdm


def convert_and_filter_topk(args):
    """ Convert to lowercase, count word occurrences and save top-k words to a file """

    counter = Counter()
    data_lower = os.path.join(
        args.output_dir, f"{args.input_txt}_lower.txt.gz")

    print("\nConverting to lowercase and counting word occurrences ...")
    if not os.path.exists(data_lower):
        with io.TextIOWrapper(
            io.BufferedWriter(gzip.open(data_lower, "w+")), encoding="utf-8"
        ) as file_out:

            # Open the input file either from input.txt or input.txt.gz
            _, file_extension = os.path.splitext(args.input_txt)
            if file_extension == ".gz":
                file_in = io.TextIOWrapper(
                    io.BufferedReader(gzip.open(args.input_txt)), encoding="utf-8"
                )
            else:
                file_in = open(args.input_txt, encoding="utf-8")

            for line in tqdm(file_in):
                line_lower = line.lower()
                counter.update(line_lower.split())
                file_out.write(line_lower)

            file_in.close()
    else:
        print(f"\nSkipping as {data_lower} already exists...")

    # Save top-k words
    print(f"\nSaving top {args.top_k} words ...")
    vocab_path = f"{args.input_txt}_vocab-{args.top_k}.txt"
    vocab_path = os.path.join(args.output_dir, vocab_path)

    if not os.path.exists(vocab_path):
        top_counter = counter.most_common(args.top_k)
        vocab_str = "\n".join(word for word, count in top_counter)
        with open(vocab_path, "w+") as file:
            file.write(vocab_str)

        print("\nCalculating word statistics ...")
        total_words = sum(counter.values())
        if total_words == 0:
            sys.exit(f"Aborting! Your text file has 0 words, cannot continue")

        print(f"  Your text file has {total_words} words in total")
        print(f"  It has {len(counter)} unique words")
        top_words_sum = sum(count for word, count in top_counter)
        word_fraction = (top_words_sum / total_words) * 100
        print(
            f"  Your top-{ args.top_k} words are {word_fraction:.4f} percent of all words"
        )
        print(
            f"  Your most common word \"{top_counter[0][0]}\" occurred {top_counter[0][1]} times")
        last_word, last_count = top_counter[-1]
        print(
            f"  The least common word in your top-k is \"{last_word}\" with {last_count} times"
        )
        for i, (w, c) in enumerate(reversed(top_counter)):
            if c > last_count:
                print(
                    "  The first word with {c} occurrences is \"{w}\" at place {len(top_counter) - 1 - i}"
                )
                break
    else:
        print(f"\nSkipping as {vocab_path} already exists...")
        vocab_file = open(vocab_path)
        print(f"\nReading vocabulary file...")
        vocab_str = vocab_file.read()

    return data_lower, vocab_str


def build_lm(args, data_lower):
    print("\nCreating ARPA file ...")
    lm_path = os.path.join(
        args.output_dir, f"{args.input_txt}_lm_{args.arpa_order}_{args.arpa_prune}.arpa")
    subargs = [
        os.path.join(args.kenlm_bins, "lmplz"),
        "--order",
        str(args.arpa_order),
        "--temp_prefix",
        args.output_dir,
        "--memory",
        args.max_arpa_memory,
        "--text",
        data_lower,
        "--arpa",
        lm_path,
        "--prune",
        *args.arpa_prune.split("|"),
    ]
    if args.discount_fallback:
        subargs += ["--discount_fallback"]
    subprocess.check_call(subargs)

    return lm_path


def filter_lm(args, lm_path, vocab_str):
    """ Filter LM using vocabulary of top-k words """

    print("\nFiltering ARPA file using vocabulary of top-k words ...")
    filtered_path = os.path.join(
        args.output_dir, f"{args.input_txt}_lm_{args.arpa_order}_{args.arpa_prune}_filtered_{args.top_k}.arpa")
    subprocess.run(
        [
            os.path.join(args.kenlm_bins, "filter"),
            "single",
            f"model:{lm_path}",
            filtered_path,
        ],
        input=vocab_str.encode("utf-8"),
        check=True,
    )

    return filtered_path


def build_binary_lm(args, filtered_path):
    """ Quantize and produce trie binary """

    print("\nBuilding lm.binary ...")
    binary_path = os.path.join(
        args.output_dir, f"{args.input_txt}_lm_{args.arpa_order}_{args.arpa_prune}_filtered_{args.top_k}.binary")
    subprocess.check_call(
        [
            os.path.join(args.kenlm_bins, "build_binary"),
            "-a",
            str(args.binary_a_bits),
            "-q",
            str(args.binary_q_bits),
            "-v",
            args.binary_type,
            filtered_path,
            binary_path,
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate LM with KenLM"
    )
    parser.add_argument(
        "--input_txt",
        help="Path to a file.txt or file.txt.gz with sample sentences",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir", help="Directory path for the output", type=str, required=False, default="."
    )
    parser.add_argument(
        "--top_k",
        help="Use top_k most frequent words for the vocab.txt file. These will be used to filter the ARPA file.",
        type=int,
        required=False,
        default=500000
    )
    parser.add_argument(
        "--kenlm_bins",
        help="File path to the KENLM binaries lmplz, filter and build_binary",
        type=str,
        required=False,
        default="/opt/kenlm/bin/"
    )
    parser.add_argument(
        "--arpa_order",
        help="Order of k-grams in ARPA-file generation",
        type=int,
        required=False,
        default=3
    )
    parser.add_argument(
        "--max_arpa_memory",
        help="Maximum allowed memory usage for ARPA-file generation",
        type=str,
        required=False,
        default="85%"
    )
    parser.add_argument(
        "--arpa_prune",
        help="ARPA pruning parameters. Separate values with '|'",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--binary_a_bits",
        help="Build binary quantization value a in bits",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--binary_q_bits",
        help="Build binary quantization value q in bits",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--binary_type",
        help="Build binary data structure type",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--discount_fallback",
        help="To try when such message is returned by kenlm: 'Could not calculate Kneser-Ney discounts [...] rerun with --discount_fallback'",
        action="store_true",
    )

    args = parser.parse_args()

    data_lower, vocab_str = convert_and_filter_topk(args)
    lm_path = build_lm(args, data_lower)
    filtered_path = filter_lm(args, lm_path, vocab_str)
    build_binary_lm(args, filtered_path)


if __name__ == "__main__":
    main()
