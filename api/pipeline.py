# все необходимые импорты
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import faiss
import json
import re
from razdel import tokenize as natasha_tokenize
import pickle
import os
from typing import List, Dict, Set, Tuple, Union
from faiss import Index
from transliterate import translit
from scipy.special import softmax 


# предполагается, что все данные, включая building.csv, лежат в указанной папке,
# и она находится по соседству с текущим файлом pipeline.py
PATH_TO_DATA_FOLDER = 'additional_data'
BUILDING_DF_FILENAME = 'building_20230808.csv'

# все утилиты, необходимые для работы скрипта'
PATH_TO_UTILITIES_FOLDER = 'utilities'

INDEX_FILENAME = 'tfidf_faiss.index'  # индекс
PATH_TO_INDEX = PATH_TO_UTILITIES_FOLDER + '/' + INDEX_FILENAME

VECTORIZER_FILENAME = 'tfidf_vectorizer.pickle'  # векторайзер tf-idf
PATH_TO_VECTORIZER = PATH_TO_UTILITIES_FOLDER + '/' + VECTORIZER_FILENAME

PREPROCESSING_MAPPER_FILENAME = 'preprocessing_mapper.json'  # маппер для предобработки
PATH_TO_PREPROCESSING_MAPPER = PATH_TO_UTILITIES_FOLDER + '/' + PREPROCESSING_MAPPER_FILENAME

POSTPROCESSING_MAPPER_FILENAME = 'postprocessing_mapper.json'  # маппер для постобработки
PATH_TO_POSTPROCESSING_MAPPER = PATH_TO_UTILITIES_FOLDER + '/' + POSTPROCESSING_MAPPER_FILENAME

# глобальные переменные
PATH_TO_BUILDING_DF = PATH_TO_DATA_FOLDER + '/' + BUILDING_DF_FILENAME
DF = pd.read_csv(PATH_TO_BUILDING_DF)
BASED_ADDRESSES = None
VECTORIZER = None
TFIDFINDEX = None
PREPROCESSING_MAPPER = None
POSTPROCESSING_MAPPER = None

# регулярки для парсинга адресов
REGEXPS = {
    'россия': r'рос+ия$',
    'санкт-петербург': r'((сан.?т.?|)п.?т.?рб.?рг|питер|спб)$',
    'москва': r'(м?сква|мск)$',
    'город': r'(\(г\.\)|город|г\.|г|гор|городок|грд)$',
    'дорога': r'до.ог.$',
    'улица': r'(улиц.|ул\.|ул)$',
    'дом': r'(дом|д\.|д?мик|д)$',
    'квартира': r'(квартир*\s|кв\.|кв)$',
    'район': r'(район|р-н\s|р)$',
    'область': r'(область|обл\.|обл)$',
    'край': r'(край|кр\.)$',
    'поселок': r'(пос*л*|п\.|пгт)$',
    'корпус': r'(корпус|к)$',
    'васильевского острова': r'(в(\s|\.)о(\s|\.)|васьк.|во)$',
    'петроградская сторона': r'п\.с\.$',
    'территория снт': r'тер. снт$',
    'территория': r'тер$',
    'участок': r'уч.?ст.?к.?$',
    'линия': r'(л-я|л.?ния)$',
    'проспект': r'(п-кт|проспект|пр|пр-кт)$',
    'аллея': r'(ал.?ея|ал)$',
    'бульвар': r'(б.+вар|б-р)$',
    'шоссе': r'ш.+се$',
}


def preprocess_based_addresses(based_addresses: List[str] = BASED_ADDRESSES) -> pd.Series:
    """
    Preprocesses a list of based addresses using predefined patterns for substitution.

    This function takes a list of based addresses and performs preprocessing
    by replacing certain patterns with their corresponding expanded forms.

    The patterns to be replaced are specified in the 'patterns' dictionary within the function.

    The based addresses can be provided as a parameter, and if not provided, a default list is used.

    Args:
        based_addresses (list or Series): A list or pandas Series containing the based addresses to be preprocessed.

    Returns:
        pandas Series: A Series containing the preprocessed based addresses with patterns replaced as per the defined patterns.
    """
    patterns = {'в.о.': 'васильевского острова',
                'тер. снт': 'территория снт',
                'п.с.': 'петроградской стороны'}
    based_addresses = pd.Series(based_addresses)

    for bad_pattern, good_pattern in patterns.items():
        based_addresses = based_addresses.apply(
            lambda x: x.replace(bad_pattern, good_pattern) if x.find(bad_pattern) else x)
    return based_addresses


def train_tfidf_vectorizer(vectorizer: TfidfVectorizer = VECTORIZER, based_addresses: List[str] = BASED_ADDRESSES) -> \
        Tuple[TfidfVectorizer, np.array]:
    """
    Trains a TF-IDF vectorizer on a collection of based addresses.

    This function trains a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer on a collection of based addresses.

    The vectorizer converts the text data into numerical features that represent the importance of words in each address.

    The resulting vectorizer and the transformed vectors can be used for various natural language processing tasks.

    Args:
        vectorizer (TfidfVectorizer, optional): An instance of TfidfVectorizer or a compatible vectorizer.
        If not provided, a default vectorizer is used.
        based_addresses (list or Series): A list or pandas Series containing the based addresses
        for training the vectorizer.

    Returns:
        tuple: A tuple containing the trained vectorizer and the array of TF-IDF based vectors.

    """
    based_vectors = vectorizer.fit_transform(based_addresses)
    return vectorizer, based_vectors.toarray()


def build_index(based_vectors: np.array, vec_dim: int = 768):
    """
    Builds a Faiss index for efficient similarity search.

    This function constructs a Faiss index using the given based vectors,
    enabling fast and memory-efficient similarity search.

    The index is built using the IndexFlatL2 implementation,
    which is suitable for L2 (Euclidean distance) distance computation.

    Args:
        based_vectors (numpy array): A 2D numpy array containing the based vectors used to construct the index.
        vec_dim (int, optional): The dimensionality of the vectors. Default is 768.

    Returns:
        faiss.IndexFlatL2: A Faiss index built on the based vectors for efficient similarity search.

    """
    index = faiss.IndexFlatL2(vec_dim)
    index.add(based_vectors)
    return index


def preprocess_abbreviations(data):
    """
    Preprocesses abbreviations and their full forms from given data.

    This function takes a list of pairs where each pair consists of an abbreviation (short form)
    and its corresponding full form (long form).

    It then processes the data to create a dictionary of normalized lowercase abbreviations mapped
    to their corresponding normalized lowercase full forms.

    The abbreviations and full forms are matched based on common words in the full form
    that correspond to the abbreviation.

    Args:
        data (list of tuples): A list of tuples, where each tuple contains an abbreviation (short form)
        and its corresponding full form (long form).

    Returns:
        dict: A dictionary containing normalized lowercase abbreviations as keys and their corresponding
        normalized lowercase full forms as values.

    """
    abbreviations_dict = {}
    for full, short in data:
        matches = re.findall(r'\b([а-яёА-ЯЁa-zA-Z]+)\.', short)
        for match in matches:
            full_form = re.search(r'\b' + match + r'[а-яёА-ЯЁa-zA-Z]+\b', full)
            if full_form:
                abbreviations_dict[match.lower() + '.'] = full_form.group(0).lower()
    return abbreviations_dict


def get_data(df_name):
    """
    Reads and processes data from a CSV file to extract relevant information.

    This function reads a CSV file specified by `df_name`, drops rows with missing values,
    and filters the data to include only rows where the "is_actual" column is True.

    It then extracts the "name" and "short_name" columns from the DataFrame and returns the data as a list of tuples.

    Args:
        df_name (str): The name or path of the CSV file to read.

    Returns:
        list of tuples: A list of tuples containing pairs of "name" and "short_name" extracted from the CSV file.

    """
    df = pd.read_csv(df_name).dropna()
    df = df[df["is_actual"]]
    data = list(zip(df['name'], df['short_name']))
    return data


def create_abbreviations_dict(folder_path: str, filenames: List[str] = ["area_20230808.csv",
                                                                        "areatype_20230808.csv", "geonim_20230808.csv",
                                                                        "geonimtype_20230808.csv",
                                                                        "prefix_20230808.csv",
                                                                        "subrf_20230808.csv", "town_20230808.csv"]) -> \
        Dict[str, str]:
    """
    Creates a dictionary of abbreviations and their corresponding full forms from CSV files.

    This function takes a folder path and a list of filenames corresponding to CSV files containing abbreviation data.

    It processes each CSV file, extracts the abbreviation and full form pairs using the 'get_data' function,
    and creates a consolidated dictionary of normalized lowercase abbreviations
    mapped to their corresponding normalized lowercase full forms.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
        filenames (list, optional): A list of filenames of CSV files to process.
        Default filenames correspond to specific data categories.

    Returns:
        dict: A dictionary containing normalized lowercase abbreviations as keys
        and their corresponding normalized lowercase full forms as values.

    """
    abbreviations_dict = {}
    df_names = list(map(lambda path: folder_path + '/' + path, filenames))
    for df_name in df_names:
        abbreviations_dict.update(preprocess_abbreviations(get_data(df_name)))
    return abbreviations_dict


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes a given text into a list of tokens.

    This function takes a string as input and tokenizes it using the Natasha library's tokenizer.

    The input text is split into individual tokens, and the resulting tokens are extracted as a list of strings.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list of str: A list of tokens extracted from the input text.

    """
    assert isinstance(text, str)
    tokens = list(natasha_tokenize(text))
    return [_.text for _ in tokens]


def prepare_text(init_text: str, mapper: Dict[str, str] = PREPROCESSING_MAPPER, keywords=REGEXPS) -> str:
    """
    Prepares and preprocesses the given text using a set of transformations.

    This function takes an input text and performs a series of preprocessing steps to enhance its readability and consistency.

    The preprocessing includes lowercasing, expanding common abbreviations, tokenization, keyword substitution, and more.

    Args:
        init_text (str): The input text to be prepared.
        mapper (dict, optional): A dictionary mapping certain tokens to their expanded forms. Default is 'PREPROCESSING_MAPPER'.
        keywords (dict, optional): A dictionary containing regex patterns for keyword substitution. Default is 'REGEXPS'.

    Returns:
        str: The preprocessed and transformed version of the input text.

    """
    text = init_text.lower()
    text = translit(text, 'ru')
    text = re.sub(r'(\d+)([а-я])', r"\1-\2", text)
    text = re.sub(r'(лит|литер?|л\.?)\s+(\w)', r'литера \2', text)
    tokens = tokenize_text(text)
    for i, token in enumerate(tokens):
        for key, pattern in keywords.items():
            if re.match(pattern, token, re.IGNORECASE):
                tokens[i] = key
                break
    text = ' '.join(tokens)
    text = text.replace(' .', '.')
    tokens = text.split(' ')
    for i, token in enumerate(tokens):
        try:
            tokens[i] = mapper[token]
        except KeyError:
            pass
    text = ' '.join(tokens)
    text = re.sub(r'[^а-яa-z0-9\-\s,/]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r' ,', ',', text)
    return text


def postprocess_search_result(address_list: List[str]) -> List[str]:
    """
    Performs post-processing on a list of addresses after search results.

    This function takes a list of addresses as input and performs post-processing to revert specific full forms of patterns
    back to their abbreviated forms.

    The patterns and their corresponding replacements are specified in the 'patterns' dictionary within the function.

    Args:
        address_list (list): A list of addresses to be post-processed.

    Returns:
        list: A list of addresses with specific full forms replaced by their corresponding abbreviated forms.

    """
    patterns = {'в.о.': 'васильевского острова',
                'тер. снт': 'территория снт',
                'п.с.': 'петроградской стороны'}

    for short_pattern, full_pattern in patterns.items():
        address_list = list(map(lambda x: x.replace(full_pattern, short_pattern), address_list))
    return address_list


def get_topn_index_for_query(query_vector: np.ndarray, index: Index, topn: int = 100) -> tuple:
    """
    Retrieves the top-N most similar indexes and their similarity scores for a given query vector.

    This function takes a query vector and a Faiss index as input, and returns the top-N most similar indexes along
    with their similarity scores. The similarity scores are computed using the cosine similarity based on the distances
    between the query vector and the indexed vectors.

    Args:
        query_vector (numpy.ndarray): The query vector for which to retrieve the most similar indexes.
        index (faiss.Index): The Faiss index used for similarity search.
        topn (int, optional): The number of top results to retrieve. Default is 100.

    Returns:
        tuple: A tuple containing two elements - a numpy array of similarity scores and a numpy array of corresponding indexes.

    """
    distances, indexes = index.search(query_vector, topn)
    distances = distances[0]
    similarity = softmax(distances)
    # if topn != 1:
    #     distances = np.round((distances - distances.min()) / (distances.max() - distances.min()), 2)
    # similarity = 1 - distances
    return similarity, indexes


def find_address_by_index(indexes: List[int], based_addresses: List[str] = BASED_ADDRESSES) -> List[str]:
    """
    Retrieves matching addresses from based addresses using the provided indexes.

    This function takes a list of indexes and a list of based addresses as input.

    It returns a list of matching addresses by retrieving the addresses corresponding
    to the given indexes from the list of based addresses.

    Args:
        indexes (list of int): The list of indexes for which to retrieve matching addresses.
        based_addresses (list of str, optional): A list of based addresses. Default is 'BASED_ADDRESSES'.

    Returns:
        list of str: A list of matching addresses retrieved from the based addresses using the provided indexes.

    """
    match_addresses = []
    for ind in indexes[0]:
        match_addresses.append(based_addresses[ind])
    return match_addresses


def get_building_ids(df: pd.DataFrame = DF) -> Dict[str, List[int]]:
    """
    Retrieves a dictionary mapping full addresses to lists of corresponding building IDs.

    This function takes a DataFrame containing building data as input (or uses the default DataFrame 'DF')
    and processes it to create a dictionary.

    The keys of the dictionary are normalized lowercase full addresses,
    and the values are lists of building IDs corresponding to each full address.

    Args:
        df (pandas.DataFrame, optional): The DataFrame containing building data. Default is 'DF'.

    Returns:
        dict: A dictionary mapping normalized lowercase full addresses to lists of building IDs.

    """
    df["full_address"] = df["full_address"].apply(lambda x: x.lower())
    grouped = df.groupby("full_address")["id"].apply(list)
    address_to_ids = grouped.to_dict()
    return address_to_ids


def extract_elements(text: str) -> Set[str]:
    """
    Extracts elements from the given text using a regular expression pattern.

    This function takes a string as input and extracts individual elements using a regular expression pattern.

    The extracted elements are returned as a set, ensuring that each element is unique.

    Args:
        text (str): The input text from which to extract elements.

    Returns:
        set: A set of extracted elements from the input text.

    """
    pattern = r'\b(\d+|[а-яА-Я])\b'
    elements = re.findall(pattern, text)
    return set(elements)


def predict_top_query_string(query, topn):
    preprocessed_query = prepare_text(query, PREPROCESSING_MAPPER, REGEXPS)
    query_vector_prepared = VECTORIZER.transform([preprocessed_query]).toarray()

    top_faiss_similarities, top_faiss_indexes_prepared = get_topn_index_for_query(query_vector_prepared,
                                                                                  TFIDFINDEX, topn=topn)

    top_faiss_addresses_prepared = find_address_by_index(top_faiss_indexes_prepared, BASED_ADDRESSES)
    top_faiss_addresses_clear = postprocess_search_result(top_faiss_addresses_prepared)

    top_faiss_addresses_ids = [POSTPROCESSING_MAPPER[top_address][0] for top_address in top_faiss_addresses_clear]

    result_json_list = []
    for address, address_id, sim in zip(top_faiss_addresses_clear, top_faiss_addresses_ids, top_faiss_similarities):
        result_json_list.append({'id': address_id,
                                 'address': address,
                                 'relative_dist': sim})

    return result_json_list


def predict_top_query_csv(query: str):
    preprocessed_query = prepare_text(query, PREPROCESSING_MAPPER, REGEXPS)
    query_vector_prepared = VECTORIZER.transform([preprocessed_query]).toarray()

    top_faiss_similarities, top_faiss_indexes_prepared = get_topn_index_for_query(query_vector_prepared,
                                                                                  TFIDFINDEX, topn=100)

    top_faiss_addresses_prepared = find_address_by_index(top_faiss_indexes_prepared, BASED_ADDRESSES)
    top_faiss_addresses_clear = postprocess_search_result(top_faiss_addresses_prepared)

    query_alone_symbols = extract_elements(preprocessed_query)
    best_sim = -1
    max_intersections = -1
    best_address = None

    for sim, address in zip(top_faiss_similarities, top_faiss_addresses_clear):
        alone_symbols = extract_elements(address)
        num_intersections = len(alone_symbols.intersection(query_alone_symbols))

        if num_intersections > max_intersections:
            max_intersections = num_intersections
            best_address = address
            best_sim = sim

    if not best_address:
        best_address = top_faiss_addresses_clear[0]
        best_sim = top_faiss_similarities[0]

    return best_address, POSTPROCESSING_MAPPER[best_address][0], best_sim


def prepare_environment():
    """
    Prepares the environment for the address prediction system.

    This function sets up the environment for the address prediction system. It preprocesses the data, trains a TF-IDF vectorizer,
    builds a Faiss index, creates abbreviation mappers, and handles data persistence.

    Args:
        DF (pandas.DataFrame): The main DataFrame containing building data.
        PATH_TO_DATA_FOLDER (str): Path to the folder containing data files.
        PATH_TO_INDEX (str): Path to save the Faiss index.
        PATH_TO_VECTORIZER (str): Path to save the TF-IDF vectorizer.
        PATH_TO_PREPROCESSING_MAPPER (str): Path to save the preprocessing mapper.
        PATH_TO_POSTPROCESSING_MAPPER (str): Path to save the postprocessing mapper.
        PATH_TO_BUILDING_DF (str): Path to the building DataFrame.

    Returns:
        None

    """
    global DF, BASED_ADDRESSES, VECTORIZER, TFIDFINDEX, PREPROCESSING_MAPPER, POSTPROCESSING_MAPPER

    BASED_ADDRESSES = DF[DF['is_actual']]['full_address'].str.lower().unique()
    BASED_ADDRESSES = preprocess_based_addresses(BASED_ADDRESSES)
    print(f"mean by is_updated: {DF['is_updated'].mean()}")

    if ((DF['is_updated'].mean() > 0) or
            (not (os.path.exists(PATH_TO_INDEX)
                  and os.path.exists(PATH_TO_VECTORIZER)
                  and os.path.exists(PATH_TO_PREPROCESSING_MAPPER)
                  and os.path.exists(PATH_TO_POSTPROCESSING_MAPPER)))):
        print('started retraining...')

        VECTORIZER = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), max_features=768)
        VECTORIZER, tfidf_matrix = train_tfidf_vectorizer(VECTORIZER, BASED_ADDRESSES)

        TFIDFINDEX = build_index(tfidf_matrix)

        PREPROCESSING_MAPPER = create_abbreviations_dict(PATH_TO_DATA_FOLDER)
        with open(PATH_TO_PREPROCESSING_MAPPER, 'w') as file:
            file.write(json.dumps(PREPROCESSING_MAPPER))

        with open(PATH_TO_VECTORIZER, 'wb') as f:
            pickle.dump(VECTORIZER, f)

        faiss.write_index(TFIDFINDEX, PATH_TO_INDEX)

        POSTPROCESSING_MAPPER = get_building_ids(DF[DF['is_actual']])
        with open(PATH_TO_POSTPROCESSING_MAPPER, 'w') as file:
            file.write(json.dumps(POSTPROCESSING_MAPPER))

        DF['is_updated'] = False
        DF.to_csv(PATH_TO_BUILDING_DF, index=False)

        print('finished retraining')
    else:
        print('started collecting already trained...')
        with open(PATH_TO_VECTORIZER, 'rb') as f:
            VECTORIZER = pickle.load(f)

        TFIDFINDEX = faiss.read_index(PATH_TO_INDEX)

        with open(PATH_TO_PREPROCESSING_MAPPER, 'r') as file:
            PREPROCESSING_MAPPER = json.loads(file.read())

        with open(PATH_TO_POSTPROCESSING_MAPPER, 'r') as file:
            POSTPROCESSING_MAPPER = json.loads(file.read())
        print('finished collecting already trained')


# prepare_environment()

if __name__ == '__main__':
    query = 'торфиная дорога 6'
    print(predict_top_query_string(query, topn=5)[0])
