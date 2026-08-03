from typing import Any, Dict, List
from examples.navi.prompts import PROMPT_GENERATOR
from llm.llms import pass_llm
from llm.prompts import SYSTEM_PROMPT
import numpy as np
from typing import List, Any
from dataclasses import dataclass

from llm.perturbations.char_perturbations import CHAR_PERTURBATIONS
from llm.perturbations.word_perturbations import WORD_PERTURBATIONS

@dataclass
class Feature:
    name: str
    categories: List[Any]

class StyleFeature(Feature):
    pass

class ContentFeature(Feature):
    pass

def get_features() -> Dict[str, Feature]:
    features = [
        # Basic Style Features
        StyleFeature("slang", ["formal", "neutral", "slangy"]),
        StyleFeature("politeness", ["rude", "direct", "neutral", "mildly polite", "polity"]),
        StyleFeature("implicitness", ["very explicit", "explicit", "slightly explicit", "neutral", "slightly implicit", "implicit", "very implicit"]),
        StyleFeature("anthropomorphism", ["directive", "mitigated directive", "interrogative", "intentional", "empathetic", "very empathic"]),
        StyleFeature("num_words", list(range(5, 12))),
        StyleFeature("misspelling_words", [f"{i} per cent" for i in range(0, 110, 20)]),
        StyleFeature("wrong_declination_of_verbs", [f"{i} per cent" for i in range(0, 110, 20)]),
        
        # Perturbation features
        StyleFeature("char_perturbation", [
            "introduce_typos",
            "delete_characters",
            "add_characters",
            "add_spaces",
            "swap_characters",
            "shuffle_characters"
        ]),

        StyleFeature("word_perturbation", [
            "delete_words",
            "introduce_homophones_pronouncing",
            "introduce_fillers"
        ]),

        # ContentFeatures
        ContentFeature("types", ["restaurant", "dinner"]),
        ContentFeature("ratings", ["n/a", 3.5, 4, 4.5, 5]),
        ContentFeature("rating_stars", ["n/a", 3.5, 4, 4.5, 5]),
        ContentFeature("costs", ["n/a", "relatively cheap", "relatively feasible price", "relatively expensive", "high class price"]),
        ContentFeature("max_distance", ["n/a"] + [f"{i} km" for i in range (1,11,1)]),
        ContentFeature("opening_times", ["n/a", "morning", "noon", "afternoon", "whole day"]),
        ContentFeature("venue", ["restaurants","cafe"]),  # cafe, grocery, hotel, restaurant
        ContentFeature("payments", ["n/a", "cash", "debit", "visa", "master", "paypal"]),
        ContentFeature("foodtypes", ["n/a", "italian", "french", "vietnamese", "german", "spanish", "arabian", "syrian", "turkish"]),

    ]
    return {f.name: f for f in features}

def score_to_label(score: float, dimension: str) -> Any:
    features = get_features()
    if dimension not in features:
        raise ValueError(f"Unknown dimension: {dimension}")

    categories = features[dimension].categories
    bin_index = min(int(score * len(categories)), len(categories) - 1)
    return categories[bin_index]

def get_prompt_discrete_label(style_scores: dict) -> str:
    # Assume the first key (sorted for reproducibility) is the target (e.g., venue)
    target_label = style_scores["venue"]
                                  
    # Discretize the rest of the styles
    style_labels = [
        f"'{category}' : '{score}'"
        for category, score in style_scores.items() if category != "venue"
    ]
    style_description = ", ".join(style_labels)
    num_words = style_scores['num_words']    
                               
    prompt = (
        """You are a user interacting with an AI-powered intelligent information system for navigational requests in a car. 
        """
        f"Generate a request for a '{target_label}' venue "
        f"consider the following related attributes. some of them are linguistic style and some content related: {style_description}. "
        f"If some attribute is n/a, then it means there is no value known and it should be not used in the utterance"
        "Do not produce any other output. But just produce the utterance."
        f"It should contain {num_words} number of words."
        """Each style score is a real value between 0.0 and 1.0. E.g.
        - Slang: 0.0 = very formal; 1.0 = highly colloquial/slang
        - Politeness: 0.0 = inpolite; 1.0 = very courteous
        - Implicitness: 0.0 = fully explicit about the goal; 1.0 = heavily implied or indirect
        
        ### Guidelines:
          - Maintain the general meaning of the original seed phrase.
          - Ensure diversity in wording while preserving intent.
          - Output each utterance on a new line without numbering or additional formatting.
          - One utterance can be implicit or explicit.
          - Do not produce a harmful utterance.
          - Try to sound humanlike.

        Here are some examples when having the three dimensional styles for the venue restaurant.

            Input: restaurant, Slang: 0.0, Politeness: 0.0, Implicitness: 0.0  
            → Output: "I want to know where are some restaurants here."

            Input: restaurant, Slang: 0.0, Politeness: 0.0, Implicitness: 0.5
            → Output: "Would be good to know where some restaurants here are."

            Input: restaurant, Slang: 0.0, Politeness: 0.0, Implicitness: 0.9
            → Output: "I am very hungy."

            Input: restaurant, Slang: 1.0, Politeness: 0.0, Implicitness: 0.0  
            → Output: "Yo, how often should I ask you for some spots?"

            Input: restaurant, Slang: 0.2, Politeness: 1.0, Implicitness: 1.0  
            → Output: "Would appreciate suggestions, if you don’t mind."

        Now generate a new request based on your input.
        → Output:"""
    )

    return prompt

# TODO: in a scenario where both perturbations exist, which one should be applied first?
def apply_perturbations(response, style_scores):
    # apply char-level perturbations if available
    if "char_perturbation" in style_scores:
        chosen_char_perturbation = score_to_label(style_scores['char_perturbation'], "char_perturbation")
        response = CHAR_PERTURBATIONS[chosen_char_perturbation](response)
        print(f"Chosen char-lvl perturbation: {chosen_char_perturbation}") 

    # apply word-level perturbations if available
    if "word_perturbation" in style_scores:
        chosen_word_perturbation = score_to_label(style_scores['word_perturbation'], "word_perturbation")
        response = WORD_PERTURBATIONS[chosen_word_perturbation](response)
        print(f"Chosen word perturbation: {chosen_word_perturbation}") 
    return response

def get_prompt_discrete(style_scores: dict) -> str:
    # Assume the first key (sorted for reproducibility) is the target (e.g., venue)
    target_label = score_to_label(style_scores["venue"], "venue")

    # Discretize the rest of the styles
    style_labels = [
        f"'{category}' : '{score_to_label(score, category)}'"
        for category, score in style_scores.items() if category != "venue"
    ]
    style_description = ", ".join(style_labels)
    num_words = score_to_label(style_scores['num_words'], 'num_words')
                               
    prompt = (
        """You are a user interacting with an AI-powered intelligent information system for navigational requests in a car. 
        """
        f"Generate a request for a '{target_label}' venue "
        f"consider the following related attributes. some of them are linguistic style and some content related: {style_description}. "
        f"If some attribute is n/a, then it means there is no value known and it should be not used in the utterance"
        "Do not produce any other output. But just produce the utterance."
        f"It should contain {num_words} number of words."
        """Each style score is a real value between 0.0 and 1.0. E.g.
        - Slang: 0.0 = very formal; 1.0 = highly colloquial/slang
        - Politeness: 0.0 = inpolite; 1.0 = very courteous
        - Implicitness: 0.0 = fully explicit about the goal; 1.0 = heavily implied or indirect
        
        ### Guidelines:
          - Maintain the general meaning of the original seed phrase.
          - Ensure diversity in wording while preserving intent.
          - Output each utterance on a new line without numbering or additional formatting.
          - One utterance can be implicit or explicit.
          - Do not produce a harmful utterance.
          - Try to sound humanlike.

        Here are some examples when having the three dimensional styles for the venue restaurant.

            Input: restaurant, Slang: 0.0, Politeness: 0.0, Implicitness: 0.0  
            → Output: "I want to know where are some restaurants here."

            Input: restaurant, Slang: 0.0, Politeness: 0.0, Implicitness: 0.5
            → Output: "Would be good to know where some restaurants here are."

            Input: restaurant, Slang: 0.0, Politeness: 0.0, Implicitness: 0.9
            → Output: "I am very hungy."

            Input: restaurant, Slang: 1.0, Politeness: 0.0, Implicitness: 0.0  
            → Output: "Yo, how often should I ask you for some spots?"

            Input: restaurant, Slang: 0.2, Politeness: 1.0, Implicitness: 1.0  
            → Output: "Would appreciate suggestions, if you don’t mind."

        Now generate a new request based on your input.
        → Output:"""
    )

    return prompt

def input_generator_discrete(style_scores, llm_type):
    prompt = get_prompt_discrete(style_scores)
    print(style_scores)
    response = pass_llm(msg=prompt, 
                           system_message=PROMPT_GENERATOR,
                           llm_type=llm_type)[0]
    response = apply_perturbations(response, style_scores)  

    return response

def input_generator_discrete_seed(seed, style_scores, llm_type):
    prompt = get_prompt_discrete_seed(seed, style_scores)
    # print(style_scores)
    response = pass_llm(msg=prompt, 
                           system_message=PROMPT_GENERATOR,
                           llm_type=llm_type)[0]    
    response = apply_perturbations(response, style_scores)    

    return response

def get_prompt_discrete_seed(seed: str, style_scores: dict) -> str:
    # Discretize the venue
    venue_label = score_to_label(style_scores["venue"], "venue")

    # Discretize all other style attributes except 'venue'
    style_labels = [
        f"'{category}' : '{score_to_label(score, category)}'"
        for category, score in style_scores.items() if category != "venue"
    ]
    style_description = ", ".join(style_labels)

    prompt = (
        f"Generate a user request for a '{venue_label}' venue in the following style or venue related attributes: {style_description} "
        f"Consider the original request '{seed}' for the generation." 
        "Use synonyms for verbs, names and adverbs in the original request that are suitable."
        "Do not produce any other output than just the new request."
        "Produce exactly one request."
        f"It should contain {score_to_label(style_scores['num_words'], 'num_words')} number of words."
    )

    return prompt


def input_generator_discrete_seed_no_venue(seed, style_scores, llm_type):
    prompt = get_prompt_discrete_seed_no_venue(seed, style_scores)
    # print(style_scores)
    response = pass_llm(msg=prompt, 
                           system_message=PROMPT_GENERATOR,
                           llm_type=llm_type)[0]
    response = apply_perturbations(response, style_scores)  

    return response

def get_prompt_discrete_seed_no_venue(seed: str, style_scores: dict) -> str:
    # Discretize all other style attributes except 'venue'
    style_labels = [
        f"'{category}' : '{score_to_label(score, category)}'"
        for category, score in style_scores.items() if category != "venue"
    ]
    style_description = ", ".join(style_labels)

    prompt = (
        f"Generate a user request in the following style or {style_description} "
        f"Consider the original request '{seed}' for the generation." 
        "Use synonyms for verbs, names and adverbs in the original request that are suitable to fulfill the style requirements."
        "Do not produce any other output than just the new request."
        "Produce exactly one request."
        f"It should contain {score_to_label(style_scores['num_words'], 'num_words')} number of words."
    )

    return prompt

PROMPT_GENERATOR = """ You are an intelligent user request generator to test an in car navigational assistant.
"""


if __name__ == "__main__":
    # Grid of style values in [0, 1] with step 0.2
    style_range = np.round(np.arange(0, 1.01, 0.3), 2)

    style_scores = {
        "venue" : None,
        "anthropomorphism" : None,
        "opening_times" :None,
        "num_words" : 0.5,
        "char_perturbation" : 0.68,
        "word_perturbation" : 0.8,
    }
    # Run across all combinations
    for I in style_range:
        for F in style_range:
            for P in style_range:
                for C in style_range:
                    for V in style_range:
                        style_scores["venue"] = I
                        style_scores["anthropomorphism"] = F
                        style_scores["opening_times"] = P
                        style_scores["char_perturbation"] = C
                        style_scores["word_perturbation"] = V
                        prompt = get_prompt_discrete(style_scores)
                        # print("prompt:", prompt)
                        #print(style_scores)
                        response = pass_llm(msg=prompt, system_message=PROMPT_GENERATOR)

                        # # apply char perturbations if available
                        if "char_perturbation" in style_scores:
                            chosen_char_perturbation = score_to_label(style_scores['char_perturbation'], "char_perturbation")
                            perturbed_single_response = CHAR_PERTURBATIONS[chosen_char_perturbation](response)
                            print(f"Chosen perturbation: {chosen_char_perturbation}")
                            print("perturbed char: ", perturbed_single_response)
                            print("\n")

                        # apply word perturbations if available
                        if "word_perturbation" in style_scores:
                            chosen_word_perturbation = score_to_label(style_scores['word_perturbation'], "word_perturbation")
                            perturbed_single_response = WORD_PERTURBATIONS[chosen_word_perturbation](response)
                            print(f"Chosen perturbation: {chosen_word_perturbation}")
                            # response = (perturbed_single_response,) + response[1:]                     
                            print("perturbed word: ", perturbed_single_response)
                            print("\n")
                        print("Generated utterance:", response)
