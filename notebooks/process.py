"""
Provides functions to process property detail data. Should
probs be moved to analyzers.
"""

from typing import List
import pandas as pd
from keras.layers import TextVectorization

### Classify the fields ###

IGNORED_FIELDS = [
    "responsivePhotos",  # list of photos of property (different zooms)
    "collections",  # similar homes, recs, etc.
    "originalPhotos",  # home photos
    "responsivePhotosOriginalRatio",
    "topNavJson",  # top nav bar metadata?
    "nearbyHomes",  # could be useful in clustering/graph approach?
    "contactFormRenderData",  # data on the realtors/agents
    "priceHistory",  # NOTE contains the price label, can't have it in feature set!
    "submitFlow",
    "adTargets",  # duplicated data about the property
    "vrModel",
    "postingContact",
    "tourEligibility",
    "openHouseSchedule",
    "pals",
    "listedBy",
    "homeInsights",  # Keywords in the description, we should Word2Vec instead
    "sellingSoon",  # prediction from Zillow?
    "associations",  # duplicated elsewhere
    "staticMap",
    "richMedia",
    "formattedChip",
    "onsiteMessage",
    "attributionInfo",
    "feesAndDues",  # duplicated elsewhere
    "rooms",  # TODO this is a list of dicts, need to handle differently, drop for now
]

# TODO need to encode these (e.g. create dense embeddings)
EMBDED_FIELDS = [
    "schools",
    "taxHistory",  # collapse into scalars (e.g. avg, last)?
    "nearbyCities",
    "nearbyZipcodes",
    "nearbyNeighborhoods",
]

# TODO need to word2vec this?
TEXT_FIELDS = [
    "description",
]

NESTED_FIELDS = [
    "resoFacts",
]

### Functions to process the fields ###


def ignore_fields(data: dict, ignore: List[str] = IGNORED_FIELDS) -> None:
    """
    Remove fields from the dataframe.
    """
    for k in ignore:
        try:
            del data[k]
        except KeyError:
            pass


def embed_fields(data: dict, embed: List[str] = EMBDED_FIELDS) -> None:
    """
    Embed fields in the dataframe. Currently just ignore them.
    """
    for k in embed:
        try:
            del data[k]
        except KeyError:
            pass  # already deleted


def unpack_nested_fields(data: dict, nested: List[str] = NESTED_FIELDS) -> None:
    """
    Unpack nested fields in the dataframe.
    """
    for k in nested:
        if k in data:
            for sub_k, v in data[k].items():
                data[sub_k] = v  # NOTE may overwrite, presumably v is the same
            del data[k]


def drop_urls(data: dict, kws: List[str] = ["url", "link", "thumb"]) -> None:
    """
    Drop URL fields from the dataframe.
    """
    url_cols = [k for k in data.keys() if any([s for s in kws if s in k.lower()])]
    for k in url_cols:
        del data[k]


def handle_atAGlanceFacts(data: dict) -> None:
    """
    Unpack the atAGlanceFacts field in the data.
    """
    if "atAGlanceFacts" in data:
        for d in data["atAGlanceFacts"]:
            data[f"fact.{d['factLabel']}"] = d["factValue"]
        del data["atAGlanceFacts"]


def process_dict(data: dict) -> dict:
    """
    Process the dictionary data.
    """
    data = list(
        data["props"]["pageProps"]["componentProps"]["gdpClientCache"].values()
    )[0]["property"]
    unpack_nested_fields(data)
    ignore_fields(data)
    embed_fields(data)
    drop_urls(data)
    handle_atAGlanceFacts(data)
    return data


### Multi-hot Encoding ###

TO_ENCODE = [
    "appliances",
    "communityFeatures",
    "cooling",
    "flooring",
    "heating",
    "interiorFeatures",
    "laundryFeatures",
    "lotFeatures",
    "livingQuarters",
    "otherFacts",
    "parkingFeatures",
    "patioAndPorchFeatures",
    "poolFeatures",
    # 'rooms',  # NOTE this is a list of dicts, need to handle differently
    "sewer",
    "view",
    "waterSource",
    "waterfrontFeatures",
    "windowFeatures",
    "constructionMaterials",
    "exteriorFeatures",
    "foundationDetails",
    "propertySubType",
    "securityFeatures",
]


def multi_hot_encode_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Multi-hot encode a column in a dataframe.
    """
    df[col] = df[col].apply(lambda x: x if x is not None else [])
    if not df[col].apply(lambda x: len(x) == 0).all():
        # Only encode if not all empty arrays
        col_str = df[col].apply(lambda x: " ".join([a.replace(" ", "") for a in x]))
        vectorize_layer = TextVectorization(max_tokens=100, output_mode="binary")
        vectorize_layer.adapt(col_str.to_numpy())
        encoded = pd.DataFrame(vectorize_layer(col_str.values).numpy())
        encoded.columns = [f"{col}.{v}" for v in vectorize_layer.get_vocabulary()]
        encoded.drop([f"{col}.[UNK]"], axis=1, inplace=True)
        df = pd.concat([df, encoded], axis=1)
    df.drop(col, axis=1, inplace=True)
    return df


def multi_hot_encode(df: pd.DataFrame, cols: List[str] = TO_ENCODE) -> pd.DataFrame:
    """
    Apply multi-hot encoding to a list of columns.
    """
    for col in cols:
        if col in df:
            df = multi_hot_encode_col(df, col)
    return df
