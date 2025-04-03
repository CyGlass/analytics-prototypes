import json
import sys
import os

def detect_es_to_os_mapping_differences(es_json_str, os_json_str):
    """
    Detects differences in mappings when migrating from Elasticsearch to OpenSearch.

    Args:
        es_json_str (str): Elasticsearch JSON string.
        os_json_str (str): OpenSearch JSON string.

    Returns:
        list: A list of mapping fields that are different (es -> os).
        Returns empty list if there is no difference.
    """

    try:
        es_data = json.loads(es_json_str)
        os_data = json.loads(os_json_str)
    except json.JSONDecodeError as e:
        return ["Invalid JSON: " + str(e)]

    es_mappings = es_data.get("mappings", {}).get("cyglass", {}).get("properties", {})
    os_mappings = os_data.get("template", {}).get("mappings", {}).get("properties", {})

    diff_fields = []

    # Compare mappings
    es_keys = set(es_mappings.keys())
    os_keys = set(os_mappings.keys())

    added_keys = es_keys - os_keys
    common_keys = es_keys.intersection(os_keys)

    diff_fields.extend(list(added_keys))

    for key in common_keys:
        if es_mappings[key] != os_mappings[key]:
            diff_fields.append(key)

    return diff_fields

def detect_diff_jsons(es_file_path, os_file_path):
    """
    Runs detect_es_to_os_mapping_differences and writes the result to diffs.txt.
    """
    try:
        with open(es_file_path, 'r') as es_file:
            es_json_str = es_file.read()

        with open(os_file_path, 'r') as os_file:
            os_json_str = os_file.read()

        diff_fields = detect_es_to_os_mapping_differences(es_json_str, os_json_str)

        if not os.path.exists("diffs.txt"):  # check if file exists.
            with open("diffs.txt", "w") as diffs_file:  # create if it does not.
                diffs_file.write(f"{es_file_path} | {diff_fields}\n")
        else:
            with open("diffs.txt", "a") as diffs_file:  # append if it does.
                diffs_file.write(f"{es_file_path} | {diff_fields}\n")

    except FileNotFoundError:
        print("Error: One or both files not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

def process_template_folders(es_templates_dir, os_templates_dir):
    """
    Processes template files from es_templates and os_templates directories.
    """

    try:
        es_files = os.listdir(es_templates_dir)
        os_files = os.listdir(os_templates_dir)
    except FileNotFoundError as e:
        print(f"Error: Directory not found: {e}")
        return

    for es_file in es_files:
        if es_file in os_files and es_file.endswith(".json"):  # only process json files.
            es_file_path = os.path.join(es_templates_dir, es_file)
            os_file_path = os.path.join(os_templates_dir, es_file)  # Assumes same filename

            detect_diff_jsons(es_file_path, os_file_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_templates.py <es_templates_dir> <os_templates_dir>")
        sys.exit(1)

    es_templates_dir = sys.argv[1]
    os_templates_dir = sys.argv[2]

    process_template_folders(es_templates_dir, os_templates_dir)