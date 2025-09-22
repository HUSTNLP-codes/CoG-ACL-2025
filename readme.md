# Correcting on Graph

Complex multi-hop questions often require comprehensive retrieval and reasoning. As a result, effectively parsing such questions and establishing an efficient interaction channel between large language models (LLMs) and knowledge graphs (KGs) is essential for ensuring reliable reasoning. In this paper, we present a novel semantic parsing framework Correcting on Graph (CoG), aiming to establish faithful logical queries that connect LLMs and KGs. We first propose a structured knowledge decoding that enables the LLM to generate fact-aware logical queries during inference, while leveraging its parametric knowledge to fill in the blank intermediate entities. Then, we introduce a knowledge path correction that combines the logical query with KGs to correct hallucination entities and path deficiencies in the generated content, ensuring the reliability and comprehensiveness of the retrieved knowledge. Extensive experiments demonstrate that CoG outperforms the state-of-the-art KGQA methods on two knowledge-intensive question answering benchmarks. CoG achieves a high answer hit rate and exhibits competitive F1 performance for complex multi-hop questions.

## Datasets
We use the KGQA datasets from RoG.

[WebQSP](https://huggingface.co/datasets/rmanluo/RoG-webqsp)   
[CWQ](https://huggingface.co/datasets/rmanluo/RoG-cwq)

You can also retrieve subgraphs from FreeBase following previous studies.

## Index Graph
We first build a graph for each QA pair and search for ground-truth paths from question entities to answer entities. The retrieved paths are added to the ground_truth_paths field, and then saves the processed data as a Dataset.
```
python build_train_paths.py \
    --d webqsp \
    --split train \
    --output_path your/output/dir/result.json \
```

## Build Train Dataset
Then, we convert the dataset into a training-friendly Alpaca format (instruction–input–output). It loads a HuggingFace Dataset from disk, extracts the question, question entities, answer entities, and the ground-truth paths for each sample, formats each path as a string like Entity1 -> relation1 -> Entity2 -> ... -> AnswerEntity, and constructs an instruction describing the task with the question included. 
```
python build_train_dataset.py \
    --input_dir your/input/dir \
    --output_file your/output/file.jsonl

```

## Fine-tuning LLM
We fine-tuned LLaMA-3.1-8B-Instruct using LLaMA-Factory. Since our dataset had already been preprocessed into the training format required by LLaMA-Factory, the model could be trained directly without additional conversion.
```
llamafactory-cli train examples/train_lora/llama3_full_sft.yaml
```

# Correction
During the reasoning stage, when the model generates multiple reasoning paths for a given question, we use a correction module to extract entities and multi-hop relations from these paths to form a logical query. By executing this query over Freebase, we can refine and correct the generated paths, ultimately producing the final query.
```
python your_script.py \
    --data_path your/input/dir \
    --save_path your/output/dir
```