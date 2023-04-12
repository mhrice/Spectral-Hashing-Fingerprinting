def evaluate_results(output_file):
    with open(output_file, "r") as f:
        lines = f.readlines()
    correct = 0
    top_1_correct = 0
    for line in lines:
        line = line.split("\t")
        query = line[0].split("-")[0]
        results = line[1:-1]
        if query == results[0]:
            top_1_correct += 1
        if query in results:
            correct += 1
    print(f"Top 1 Accuracy: {top_1_correct / len(lines)}")
    print(f"Top 3 Accuracy: {correct / len(lines)}")
