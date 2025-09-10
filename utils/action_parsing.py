import os
import json
import logging

from utils.exceptions import RetrievalError


class ActionParsingContext():
    def __init__(
            self, 
            required_fields = ["action_id", "action_title", "key_messages"],
            all_actions_cache_file=None,
            separated_actions_cache_dir=None
        ):
        # doc_type can be "km" or "bg_km", with "km" for key messages and "bg_km" for background key messages
        self.__set_required_fields(required_fields=required_fields)
        self.__set_metadata_fields(required_fields=required_fields)
        self.__set_doc_type(required_fields=required_fields)
        self.__set_all_actions_cache_file(all_actions_cache_file=all_actions_cache_file, required_fields=required_fields,  doc_type=self.__doc_type)
        self.__set_separated_actions_cache_dir(separated_actions_cache_dir=separated_actions_cache_dir, required_fields=required_fields, doc_type=self.__doc_type)

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

    def __set_all_actions_cache_file(self, all_actions_cache_file, required_fields, doc_type):
        if all_actions_cache_file is None:
            if "effectiveness" in required_fields:
                self.__all_actions_cache_file = f"action_data/parsed_{doc_type}_eff_all_cache.json"
            else:
                self.__all_actions_cache_file = f"action_data/parsed_{doc_type}_noeff_all_cache.json"
        else:
            self.__all_actions_cache_file = all_actions_cache_file

    def __set_separated_actions_cache_dir(self, separated_actions_cache_dir, required_fields, doc_type):
        if separated_actions_cache_dir is None:
            if "effectiveness" in required_fields:
                self.__separated_actions_cache_dir = f"action_data/parsed_{doc_type}_eff_separated_cache"
            else:
                self.__separated_actions_cache_dir = f"action_data/parsed_{doc_type}_noeff_separated_cache"
        else:
            self.__separated_actions_cache_dir = separated_actions_cache_dir

    def get_required_fields(self):
        return self.__required_fields
    
    def get_metadata_fields(self):
        return self.__metadata_fields

    def get_doc_type(self):
        return self.__doc_type    
    
    def get_all_actions_cache_file(self):
        return self.__all_actions_cache_file
    
    def get_separated_actions_cache_dir(self):
        return self.__separated_actions_cache_dir


def load_cache(filepath):
    if not os.path.exists(filepath):
        raise RetrievalError(f"File not found to read from: {filepath}.")
    else:
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                content = json.load(file)
                logging.debug(f"Loaded parsed actions from cache file {filepath}.")
                return content                    
        except json.JSONDecodeError as e:
            raise RetrievalError(f"Failed to load JSON from file {filepath}: {str(e)}.")
        

def load_separated_cache(cache_dir, cache_filename):
    if cache_dir is None:
        raise RetrievalError(f"'None' cache_dir given to function load_from_separated_cache.")
    else:
        content = load_cache(filepath=os.path.join(cache_dir, cache_filename))
        if not isinstance(content, dict):
            raise RetrievalError(f"Expected JSON file to contain a dictionary, but contained {type(content)} instead: {os.path.join(cache_dir, cache_filename)}")
        else:
            return content
    

def load_all_cache(cache_filepath):
    if cache_filepath is None:
        raise RetrievalError("'None' filename given to function load_cache.")
    else:
        content = load_cache(filepath=cache_filepath)
        if not isinstance(content, list):
            raise RetrievalError(f"Expected JSON file to contain a list, but contained {type(content)} instead: {cache_filepath}")
        else:
            return content


def save_cache(data, cache_filepath):
    cache_dir = os.path.dirname(cache_filepath)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(cache_filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_separated_cache(data, cache_dir, cache_filename):
    if cache_dir is None:
        raise RetrievalError(f"'None' cache_dir given to function save_to_separated_cache.")
    else:
        save_cache(data=data, cache_filepath=os.path.join(cache_dir, cache_filename))


def save_all_cache(data, cache_filepath):
    if cache_filepath is None:
        raise RetrievalError(f"'None' cache_filepath given to function save_all_cache.")
    else:
        save_cache(data=data, cache_filepath=cache_filepath)


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



def get_parsed_action_by_id(id, context : ActionParsingContext, load_from_separated_cache=True, save_to_separated_cache=True):
    separated_cache_dir = context.get_separated_actions_cache_dir()
    if load_from_separated_cache:
        try:
            return load_separated_cache(cache_dir=separated_cache_dir, cache_filename=f"action_{id}_clean.json")
        except RetrievalError as e:
            logging.error(f"Error loading from cache: {str(e)}. Proceeding to parse action from text file.")
            # If loading from cache fails, fall back to parsing the file.

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
        
        if save_to_separated_cache:
            save_separated_cache(data=parsed_action, cache_dir=separated_cache_dir, cache_filename=f"action_{id}_clean.json")
        
        return parsed_action
    
    else:
        return None



def get_all_parsed_actions(context : ActionParsingContext, load_from_all_cache=True, save_to_all_cache=True, saved_to_separated_cache=True):
    """
    Get parsed actions (of all synopses) from the data directory.

    Args:
        context (ActionParsingContext): The context for action retrieval (e.g can use this to find the user's set doc_type, required_fields, metadata_fields)
    
    Returns:
        list: List of parsed action dictionaries
    """
    doc_type = context.get_doc_type()
    all_cache_file = context.get_all_actions_cache_file()
    separated_cache_dir = context.get_separated_actions_cache_dir()

    if load_from_all_cache:
        try:
            parsed_actions = load_all_cache(cache_filepath=all_cache_file)
            return parsed_actions
        except RetrievalError as e:
            logging.error(f"Error loading from cache: {str(e)}. Proceeding to parse actions from files.")
            
    logging.debug("Loading and parsing all actions from text files...")
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
                if saved_to_separated_cache:
                    save_separated_cache(data=parsed_action, cache_dir=separated_cache_dir, cache_filename=filename.replace(".txt", ".json"))
                parsed_actions.append(parsed_action)
    
    if save_to_all_cache:
        save_all_cache(data=parsed_actions, cache_filepath=all_cache_file)

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
    logging.basicConfig(filename="logfiles/action_parsing.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    context = ActionParsingContext(
        required_fields=["action_id", "action_title", "key_messages"]
    )

    ## Testing parsed actions with bg km
    docs = get_all_parsed_actions(context=context)
    for i in range(100, 105):
        print(docs[i])




if __name__ == "__main__":
    main()