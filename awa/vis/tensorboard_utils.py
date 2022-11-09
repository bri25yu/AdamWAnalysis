from typing import List, Tuple, Union

import os

from numpy import array, ndarray

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.python.summary.summary_iterator import summary_iterator


__all__ = ["get_property_and_steps"]


def load_eventfile(log_dir: str) -> List:
    # Get the eventfile_path
    eventfile_name = os.listdir(log_dir)[0]
    eventfile_path = os.path.join(log_dir, eventfile_name)

    return list(summary_iterator(eventfile_path))


def filter_summaries_by_tag(summaries: List, tag: str) -> List[Tuple]:
    """
    Filters summaries for all events 
    """
    value_is_tag = lambda v: v.tag == tag
    get_value_tag_from_event = lambda e: next(filter(value_is_tag, e.summary.value), None)

    filtered = []
    for event in summaries:
        value = get_value_tag_from_event(event)
        if value is None:
            continue

        filtered.append((event, value))

    return filtered


def get_first_simple_value(summaries: List[Tuple]) -> float:
    """
    Takes in the output of `filter_summaries_by_tag`
    """
    return next(iter(summaries))[1].simple_value


def get_first_tag_simple_value(summaries: List, tag: str) -> float:
    filtered = filter_summaries_by_tag(summaries, tag)
    return get_first_simple_value(filtered)


def get_property_and_steps(log_dir: str, property_name: str) -> Tuple[List[float], List[Union[float, ndarray]]]:
    """
    Returns a tuple of steps and property values.

    The arrays are sorted ascending in steps.
    """
    experiment_summary = load_eventfile(log_dir)

    train_returns = filter_summaries_by_tag(experiment_summary, property_name)
    steps = [r[0].step for r in train_returns]
    returns = [r[1].simple_value for r in train_returns]

    steps = array(steps)
    returns = array(returns)

    sorted_idxs = steps.argsort()

    steps = steps[sorted_idxs]
    returns = returns[sorted_idxs]

    return steps, returns
