#!/usr/bin/env python
"""This script calculates the Mean Average Precision (MAP) measure for a TREC-CAR run file."""
__author__ = "Shubham Chatterjee"
__version__ = "8/16/19"

from typing import Dict, List
import sys
import argparse


def get_rankings(file_path: str) -> Dict[str, List[str]]:
    rankings: Dict[str, List[str]] = {}
    with open(file_path, 'r') as file:
        for line in file:
            line_parts = line.split(" ")
            queryID = line_parts[0]
            field2 = line_parts[2]
            _list: List[str] = []
            if queryID in rankings.keys():
                _list = rankings[queryID]
            _list.append(field2)
            rankings[queryID] = _list
    return rankings


def average_prec(ret_para_list: List[str], rel_para_list: List[str]) -> float:
    ret_i: int = 0
    rel_i: int = 0
    s: float = 0.0
    avg_prec_list: List[float] = []

    for paraID in ret_para_list:
        ret_i = ret_i + 1
        if paraID in rel_para_list:
            rel_i = rel_i + 1
            avg_prec_list.append(rel_i / ret_i)

    for prec in avg_prec_list:
        s += prec

    avg_prec = s / len(rel_para_list)

    return round(avg_prec, 4)


def mean_average_precision(run_file: str, qrel_file:str):
    run_file_dict = get_rankings(run_file)
    qrel_file_dict = get_rankings(qrel_file)
    p, prec_dict = mean_avg_prec(run_file_dict, qrel_file_dict)
    return p, prec_dict


def mean_avg_prec(run_file_dict: Dict[str, List[str]], qrel_file_dict: Dict[str, List[str]]):
    prec_dict: Dict[str, float] = {}
    s: float = 0.0
    for queryID in run_file_dict:
        ret_para_list = run_file_dict[queryID]
        if queryID in qrel_file_dict:
            rel_para_list = qrel_file_dict[queryID]
            ap = average_prec(ret_para_list, rel_para_list)
            prec_dict[queryID] = ap

    for i in prec_dict.values():
        s += i

    _map = s / len(prec_dict)
    return _map, prec_dict


def num_rel_ret(ret_para_list, rel_para_list):
    return len(list(set(ret_para_list) & set(rel_para_list)))


def main():
    parser = argparse.ArgumentParser("Calculate Precision at 1 for a TREC-CAR run file")
    parser.add_argument("--filepath", help="Path to the run file.", required=True)
    parser.add_argument("--qrelpath", help="Path to the ground truth (qrel) file.", required=True)
    parser.add_argument("--q", help="Whether per query or not. Default is False.",  action="store_true")
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    p, prec_dict = mean_average_precision(args.filepath, args.qrelpath)

    if args.q:
        for queryID in prec_dict:
            _map = prec_dict[queryID]
            print_string = "map" + "\t\t\t" + queryID + "\t" + "{:.4f}".format(_map)
            print(print_string)

    print("map" + "\t\t\t" + "all" + "\t" + "{:.4f}".format(p))


if __name__ == '__main__':
    main()