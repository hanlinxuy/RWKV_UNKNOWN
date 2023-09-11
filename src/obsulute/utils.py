import os
import json
import re


def sorted_fn(text):
    n = re.findall(r"rwkv\-(.+?)\.", text)
    n = n[0]
    n = int(n)
    return n


def get_model(proj_path):
    files = os.listdir(proj_path)
    model_weights = [x for x in files if x.endswith(".pth")]
    model_weights = sorted(model_weights, key=sorted_fn)
    model = model_weights[-1]
    model = f"{proj_path}/{model}"
    return model


def next_model(proj_path):
    try:
        files = os.listdir(proj_path)
        model_weights = [x for x in files if x.endswith(".pth")]
        model_weights = sorted(model_weights, key=sorted_fn)
        model = model_weights[-1]
        n = re.findall(r"rwkv\-(.+?)\.", model)
        n = n[0]
        n = int(n)
        return n + 1
    except:
        return 0
