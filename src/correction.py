import argparse
from datasets import Dataset
from tqdm import tqdm
from utils.graph_utils import build_graph, bfs_with_rule

def retrieve_paths(example):
    graph = build_graph(example['graph'])
    prediction = example['prediction']
    path_set = set()
    used_entity = set()

    for pred in prediction:
        parts = [p.strip() for p in pred.split('->')]
        start_node = parts[0]
        target_rule = parts[1::2]

        res = bfs_with_rule(graph, start_node, target_rule)
        if len(res) == 0:
            continue

        for path_list in res:
            if len(path_list) == 0:
                continue
            path = f"{path_list[0][0]} -> {path_list[0][1]} -> {path_list[0][2]}"
            candidate_answer = path_list[0][2]
            for h, r, t in path_list[1:]:
                path = path + f" -> {r} -> {t}"
                candidate_answer = t
            if candidate_answer not in used_entity:
                path = f"Answer: {candidate_answer}, Path: {path}"
                path_set.add(path)
                used_entity.add(candidate_answer)

    return {"paths": list(path_set)}

def build_prompt2(example):
    instruction = (
        "Based on the reasoning paths and your own knowledge, please answer the given question. "
        "Please keep the answer as simple as possible and return all the possible answers as a list.\n"
    )
    if len(example["paths"]) > 0:
        instruction += "# Paths:\n"
        instruction += "\n".join(example["paths"])
        instruction += "\n"
    else:
        instruction += "# Paths: No paths available, please answer the question based on your knowledge. Please keep the answer as simple as possible and return all the possible answers as a list\n"
    instruction += f"# Question: {example['question']}"
    return {"prompt2": instruction}


def main(data_path, save_path):
    dataset = Dataset.load_from_disk(data_path)
    dataset = dataset.map(retrieve_paths, desc="retrieving paths")
    dataset = dataset.map(build_prompt2, desc="building prompts")
    dataset.save_to_disk(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve paths and build prompts from predictions dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the predictions dataset")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the processed dataset")
    args = parser.parse_args()

    main(args.data_path, args.save_path)
