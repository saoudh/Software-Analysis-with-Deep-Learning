from os import listdir
from os.path import join, isfile
import json
from random import randint
import sys
from code_completion import Code_Completion

#########################################
## START of part that students may change

# if directory for training and query data is given, then it is used, otherwise default directory is used
if len(sys.argv) == 2:
    training_dir=sys.argv[1]+"/programs_800/"
    query_dir = sys.argv[1]+"/programs_200/"
else:
    training_dir = "../programs_800/"
    query_dir = "../programs_200/"

# model file name and location
model_file="mymodel/model1"
use_stored_model = True

max_hole_size = 2
# abstracting the token identifier-names
simplify_tokens = True
## END of part that students may change
#########################################

def simplify_token(token):
    if token["type"] == "Identifier":
        token["value"] = "ID"
    elif token["type"] == "String":
        token["value"] = "\"STR\""
    elif token["type"] == "RegularExpression":
        token["value"] = "/REGEXP/"
    elif token["type"] == "Numeric":
        token["value"] = "5"


# load sequences of tokens from files
def load_tokens(token_dir):
    # join concatenates sequence of strings,i.e. token_dir+f. The following instruction checks all files in the token_dir
    # wether they are tokens and put them into an array
    token_files = [join(token_dir, f) for f in listdir(token_dir) if isfile(join(token_dir, f)) and f.endswith("_tokens.json")]
    # following instruction writes the content of the token-files to an array
    token_lists = [json.load(open(f, encoding='utf8')) for f in token_files]
    if simplify_tokens:
        for token_list in token_lists:
            # assign standard values for simple AST-types like String and Numbers
            for token in token_list:
                simplify_token(token)
    return token_lists

# removes up to max_hole_size tokens
def create_hole(tokens):
    hole_size = min(randint(1, max_hole_size), len(tokens) - 1)
    hole_start_idx = randint(1, len(tokens) - hole_size)
    prefix = tokens[0:hole_start_idx]
    expected = tokens[hole_start_idx:hole_start_idx + hole_size]
    suffix = tokens[hole_start_idx + hole_size:]
    return(prefix, expected, suffix)

# checks if two sequences of tokens are identical
def same_tokens(tokens1, tokens2):
    if len(tokens1) != len(tokens2):
        return False
    for idx, t1 in enumerate(tokens1):
        t2 = tokens2[idx]
        if t1["type"] != t2["type"] or t1["value"] != t2["value"]:
            return False  
    return True

#########################################
## START of part that students may change
code_completion = Code_Completion()
## END of part that students may change
#########################################

# train the network
training_token_lists = load_tokens(training_dir)

if use_stored_model:
    code_completion.load(training_token_lists, model_file)
else:
    code_completion.train(training_token_lists, model_file)

# query the network and measure its accuracy
query_token_lists = load_tokens(query_dir)
correct = incorrect = 0
for tokens in query_token_lists:
    (prefix, expected, suffix) = create_hole(tokens)

    completion = code_completion.query(prefix, suffix)
    if same_tokens(completion, expected):
        correct += 1
    else:
        incorrect += 1
accuracy = correct / (correct + incorrect)
print("Accuracy: " + str(correct) + " correct vs. " + str(incorrect) + " incorrect = "  + str(accuracy))

