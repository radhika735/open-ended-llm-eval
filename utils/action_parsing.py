import os
import json
import logging

from utils.exceptions import RetrievalError


class ActionParsingContext():
    def __init__(self, required_fields = ["action_id", "action_title", "key_messages"], load_from_cache=False, save_to_cache=False, all_actions_cache_file="action_data/bg_km_"):
        # doc_type can be "km" or "bg_km", with "km" for key messages and "bg_km" for background key messages
        self.__set_required_fields(required_fields=required_fields)
        self.__set_metadata_fields(required_fields=required_fields)
        self.__set_doc_type(required_fields=required_fields)
        self.__load_from_cache = load_from_cache
        self.__save_to_cache = save_to_cache
        self.__all_actions_cache_file = all_actions_cache_file

    def __set_required_fields(self, required_fields):
        # remove any fields which are not in the allowed list of fields:
        allowed_fields = ["action_id", "action_title", "effectiveness", "key_messages", "background_information"]
        self.__required_fields = [f for f in required_fields if f in allowed_fields]
        # remove duplicates from the list:
        self.__required_fields = list(set(self.__required_fields))
        # remove all instances of action_id and action_title from the list and self prepend for consistency:
            # self.__required_fields must begin with the elements "action_id" then "action_title" and there must not be duplicates of these in the required fields.
        # filtering out any elements which are "action_id" or "action_title":
        self.__required_fields = [f for f in required_fields if f not in ["action_id", "action_title"]]
        self.__required_fields = ["action_id", "action_title"] + self.__required_fields

    def __set_metadata_fields(self, required_fields):
        self.__metadata_fields = [f for f in required_fields if f not in ["key_messages", "background_information"]]

    def __set_doc_type(self, required_fields):
        if "key_messages" in required_fields:
            if "background_information" in required_fields:
                self.__doc_type = "bg_km"
            else:
                self.__doc_type = "km"

    def get_required_fields(self):
        return self.__required_fields
    
    def get_metadata_fields(self):
        return self.__metadata_fields

    def get_doc_type(self):
        return self.__doc_type
    
    def get_load_from_cache(self):
        return self.__load_from_cache
    
    def get_save_to_cache(self):
        return self.__save_to_cache

    def get_all_actions_cache_file(self):
        return self.__all_actions_cache_file
    


def load_cache(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            try:
                content = json.load(file)
                if not isinstance(content, list):
                    raise RetrievalError(f"Expected JSON file to contain a list, but contained {type(content)} instead: {filename}")
                else:
                    logging.info(f"Loaded parsed actions from cache file {filename}.")
                    return content
            
            except json.JSONDecodeError as e:
                raise RetrievalError(f"Failed to load JSON from file {filename}: {str(e)}.")
    else:
        raise RetrievalError(f"File not found to read from: {filename}.")



def save_to_cache(data, cache_filepath):
    if not os.path.exists(os.path.dirname(cache_filepath)):
        os.makedirs(os.path.dirname(cache_filepath))
    with open(cache_filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"Wrote to cache {cache_filepath}")



def parse_action(action_string, context : ActionParsingContext):
    """
    Parse an action string into its components.
    
    Args:
        action_string (str): The action string to parse.
        context (ActionParsingContext): The context for action retrieval (i.e. can use this to find the user's set required_fields)

    Returns:
        dict: A dictionary containing the parsed action components.
    """
    required_fields = context.get_required_fields()
    all_action_fields = {}
    parsed_action = {}
    lines = action_string.strip().splitlines()

    # Remove the line "Synopsis Details:" and lines after it.
    for i, line in enumerate(lines):
        if line == "Synopsis Details:":
            lines = lines[:i]
            break

    # Extract action id, action title and effectiveness rating.
    action_id, action_title = lines[0].split(": ", 1)
    effectiveness = lines[1] if len(lines) > 1 else ""

    # Parse (optional) background information and (mandatory) key messages.
        # (whether background information is in the file depends on the contents of the action_string passed as argument).
    bg_index = None
    km_index = None

    for line in lines[2:]:
        if line.startswith("Background information and definitions:"):
            bg_index = lines.index(line)
        if line.startswith("Key Messages:"):
            km_index = lines.index(line)

    if bg_index is not None:
        # Background information exists in the action file, extract it and store it.
        bg_lines = lines[bg_index:km_index]
        background_information = "\n".join(lines[bg_index:km_index])
        all_action_fields["background_information"] = background_information.strip()

    # Extract key messages.
    key_messages = "\n".join(lines[km_index:]) if km_index is not None else ""

    # Store extracted information.
        # (apart from background information which will optionally have been stored earlier)
    all_action_fields.update({
        "action_id": action_id.strip(),
        "action_title": action_title.strip(),
        "effectiveness": effectiveness.strip(),
        "key_messages": key_messages.strip()
    })

    # Only return the required fields.
    for field in required_fields:
        if field not in all_action_fields:
            logging.warning(f"Invalid required field '{field}' given to function parse_action")
        else:
            parsed_action[field] = all_action_fields[field]

    return parsed_action



def get_parsed_action_by_id(id, context : ActionParsingContext):
    doc_type = context.get_doc_type()
    if doc_type == "km":
        data_dir="action_data/key_messages/km_all"
    elif doc_type == "bg_km":
        data_dir="action_data/background_key_messages/bg_km_all"
    else:
        raise ValueError("Invalid doc_type. Use 'km' for key messages or 'bg_km' for background key messages.")
    
    filename = f"action_{id}_clean.txt"
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as action_file:
            content = action_file.read()
        parsed_action = parse_action(action_string=content, context=context)
        return parsed_action
    else:
        return None



def get_all_parsed_actions(context : ActionParsingContext):
    """
    Get parsed actions (of all synopses) from the data directory.

    Args:
        context (ActionParsingContext): The context for action retrieval (e.g can use this to find the user's set doc_type, required_fields, metadata_fields)
    
    Returns:
        list: List of parsed action dictionaries
    """
    doc_type = context.get_doc_type()
    cache_file = context.get_all_actions_cache_file()
    if context.get_load_from_cache():
        try:
            parsed_actions = load_cache(filename=cache_file)
            return parsed_actions
        except RetrievalError as e:
            logging.error(f"Error loading from cache: {str(e)}. Proceeding to parse actions from files.")
            
    parsed_actions = []
    
    if doc_type == "km":
        data_dir="action_data/key_messages/km_all"
    elif doc_type == "bg_km":
        data_dir="action_data/background_key_messages/bg_km_all"
    else:
        raise ValueError("Invalid doc_type. Use 'km' for key messages or 'bg_km' for background key messages.")
    
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as action_file:
                file_contents = action_file.read()
                parsed_action = parse_action(action_string=file_contents, context=context)
                if context.get_save_to_cache():
                    save_to_cache(data=parsed_action, cache_filepath=os.path.join(cache_dir, filename))
                parsed_actions.append(parsed_action)

    return parsed_actions



def get_parsed_action_as_str(action):
    action_string = f"{action['action_id']}: {action['action_title']}"
    for k,v in action.items():
        if k not in ["action_id", "action_title"]:
            cleaned_key_name = " ".join(k.split("_"))
            cleaned_key_name = cleaned_key_name.title()
            action_string += f"\n{cleaned_key_name}: {v}"
    return action_string



def get_parsed_action_metadata(action, context : ActionParsingContext):
    metadata_fields = context.get_metadata_fields()
    metadata = {}
    for k,v in action.items():
        if k in metadata_fields:
            metadata[k] = v
    return metadata



def get_synopsis_data_as_str(synopsis : str, doc_type="bg_km"):
    no_gaps_synopsis = "".join(synopsis.split())
    try:
        if doc_type == "bg_km":
            synopsis_file_path = f"action_data/background_key_messages/bg_km_synopsis_concat/bg_km_{no_gaps_synopsis}_concat.txt"
        elif doc_type == "km":
            synopsis_file_path = f"action_data/key_messages/km_synopsis_concat/km_{no_gaps_synopsis}_concat.txt"
        else:
            logging.error(f"Invalid argument {doc_type} given to parameter 'doc_type' in function 'get_synopsis_data_as_str'.")
            raise RetrievalError(f"Unable to retrieve synopsis data for doc_type: {doc_type}. Valid options are 'km' or 'bg_km'.")

        with open(synopsis_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if content == "":
            logging.error(f"No content in concatenated actions files for synopsis {synopsis} (see {synopsis_file_path}).")
            raise RetrievalError(f"No content in concatenated actions file for synopsis {synopsis} (see {synopsis_file_path}).")
        else:
            return content
    
    except FileNotFoundError:
        logging.error(f"Concatenated actions file for synopsis {synopsis} not found: {synopsis_file_path}")
        raise RetrievalError(f"Concatenated actions file for synopsis {synopsis} not found: {synopsis_file_path}")



def main():
    logging.basicConfig(filename="logfiles/action_retrieval.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    context = ActionParsingContext(
        required_fields=["action_id", "action_title", "key_messages"]
    )

    ## Testing parsed actions with bg km
    docs = get_all_parsed_actions(context=context)
    for i in range(100, 105):
        print(docs[i])




if __name__ == "__main__":
    main()