import argparse
import json
from datasets import Dataset
from tqdm import tqdm

def process_data(input_dir, output_dir):
    index_dataset = Dataset.load_from_disk(input_dir)
    res = []

    for data in tqdm(index_dataset):
        question = data['question']
        question_entity = data['q_entity']
        answer_entity = data['a_entity']
        GT_paths = data['ground_truth_paths']
        
        if not question_entity or not answer_entity or not GT_paths:
            continue
        
        for path in GT_paths:
            if not path:
                continue
            answer = f"{path[0][0]} -> {path[0][1]} -> {path[0][2]}"
            if len(path) > 1:
                for h, r, t in path[1:]:
                    answer += f" -> {r} -> {t}"

            instruction = f'''Task: Generate a structured knowledge graph path from the given question entities to the answer entities.
Instructions: 1. Start from one of the question entities provided. 2. Use existing relations in Freebase. 3. Generate a path that leads to the answer entity. 4. Represent the path in the format: Entity1 -> relation1 -> Entity2 -> relation2 -> ... -> AnswerEntity
# Question: {question}
# Question Entity: {data['q_entity']}'''
            
            res.append({
                "instruction": instruction,
                "input": "",
                "output": answer
            })

    with open(output_dir, 'w') as f:
        for item in res:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert shortest path dataset into training format")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input dataset")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file")
    args = parser.parse_args()

    process_data(args.input_dir, args.output_file)
