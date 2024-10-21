import json
from abc import ABCMeta, abstractmethod

import numpy as np

FEATURE_LIST = ['Node Type', 'Startup Cost',
                'Total Cost', 'Plan Rows', 'Plan Width']
LABEL_LIST = ['Actual Startup Time', 'Actual Total Time', 'Actual Self Time']

UNKNOWN_OP_TYPE = "Unknown"
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
OTHER_TYPES = ['Bitmap Index Scan']
OP_TYPES = [UNKNOWN_OP_TYPE, "Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort", "Limit"] \
           + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES


class PlanCardReplacer:
    def __init__(self, table_array, rows_array) -> None:
        self.table_array = table_array
        self.rows_array = rows_array
        self.SCAN_TYPES = SCAN_TYPES
        self.JOIN_TYPES = JOIN_TYPES
        self.SAME_CARD_TYPES = ["Hash", "Materialize",
                                "Sort", "Incremental Sort", "Limit"]
        self.OP_TYPES = ["Aggregate", "Bitmap Index Scan"] + \
                        self.SCAN_TYPES + self.JOIN_TYPES + self.SAME_CARD_TYPES
        self.table_idx_map = {}
        for arr in table_array:
            for t in arr:
                if t not in self.table_idx_map:
                    self.table_idx_map[t] = len(self.table_idx_map)

        self.table_num = len(self.table_idx_map)
        self.table_card_map = {}
        for i in range(len(table_array)):
            arr = table_array[i]
            card = rows_array[i]
            code = self.encode_input_tables(arr)
            if code in self.table_card_map:
                pass
            else:
                self.table_card_map[code] = card

    def replace(self, plan):
        input_card = None
        input_tables = []
        output_card = None

        if "Plans" in plan:
            children = plan['Plans']
            child_input_tables = None
            if len(children) == 1:
                child_input_card, child_input_tables = self.replace(children[0])
                input_card = child_input_card
                input_tables += child_input_tables
            else:
                for child in children:
                    _, child_input_tables = self.replace(child)
                    input_tables += child_input_tables

        node_type = plan['Node Type']
        if node_type in self.JOIN_TYPES:
            tag = self.encode_input_tables(input_tables)
            if tag not in self.table_card_map:
                print(input_tables)
                raise Exception("Unknown tag " + str(tag))
            card = self.table_card_map[tag]
            plan['Plan Rows'] = card
            output_card = card
        elif node_type in self.SAME_CARD_TYPES:
            if input_card is not None:
                plan['Plan Rows'] = input_card
                output_card = input_card
        elif node_type in self.SCAN_TYPES:
            input_tables.append(plan['Relation Name'])
        elif node_type not in self.OP_TYPES:
            raise Exception("Unknown node type " + node_type)

        return output_card, input_tables

    def encode_input_tables(self, input_table_list):
        l = [0 for _ in range(self.table_num)]
        for t in input_table_list:
            l[self.table_idx_map[t]] += 1

        code = 0
        for i in range(len(l)):
            code += l[i] * (10 ** i)
        return code
