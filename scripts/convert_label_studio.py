import json

def tokens_to_text_and_spans(tokens, ner_tags):
    text = " ".join(tokens)
    
    # Build token start positions
    positions = []
    current_pos = 0
    
    for token in tokens:
        start = current_pos
        end = start + len(token)
        positions.append((start, end))
        current_pos = end + 1  # +1 for space
    
    results = []
    
    for start_idx, end_idx, label in ner_tags:
        start_char = positions[start_idx][0]
        end_char = positions[end_idx][1]
        
        entity_text = text[start_char:end_char]
        
        results.append({
            "value": {
                "start": start_char,
                "end": end_char,
                "text": entity_text,
                "labels": [label]
            },
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
        })
    
    return text, results


def convert_jsonl_to_labelstudio(input_file, output_file):
    output_data = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            tokens = record["tokenized_text"]
            ner_tags = record["ner"]
            
            text, results = tokens_to_text_and_spans(tokens, ner_tags)
            
            ls_format = {
                "data": {
                    "text": text
                },
                "annotations": [
                    {
                        "result": results
                    }
                ]
            }
            
            output_data.append(ls_format)
    
    # Write JSON (not JSONL)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


# Example usage
convert_jsonl_to_labelstudio("data/processed/gliner/valid.jsonl", "data/processed/gliner/labelstudio/valid.json")