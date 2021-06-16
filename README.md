# Generate n-gram LM

Simple tooling for generating n-gram language models with KenLM.

Generated LMs for the catalan language can be found in the releases section.

## Building the Docker Image

```
docker build . -f Dockerfile -t kenlm
```

## Building a language model
```
docker run -it --rm -v `pwd`:/io -w /io kenlm python generate_lm.py --input_txt catalan_textual_corpus.txt \ --output_dir . --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" --binary_a_bits 255 --binary_q_bits 8 \ --binary_type trie
```