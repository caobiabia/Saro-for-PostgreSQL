all_48_hint_sets = '''hashjoin,indexonlyscan
hashjoin,indexonlyscan,indexscan
hashjoin,indexonlyscan,indexscan,mergejoin
hashjoin,indexonlyscan,indexscan,mergejoin,nestloop
hashjoin,indexonlyscan,indexscan,mergejoin,seqscan
hashjoin,indexonlyscan,indexscan,nestloop
hashjoin,indexonlyscan,indexscan,nestloop,seqscan
hashjoin,indexonlyscan,indexscan,seqscan
hashjoin,indexonlyscan,mergejoin
hashjoin,indexonlyscan,mergejoin,nestloop
hashjoin,indexonlyscan,mergejoin,nestloop,seqscan
hashjoin,indexonlyscan,mergejoin,seqscan
hashjoin,indexonlyscan,nestloop
hashjoin,indexonlyscan,nestloop,seqscan
hashjoin,indexonlyscan,seqscan
hashjoin,indexscan
hashjoin,indexscan,mergejoin
hashjoin,indexscan,mergejoin,nestloop
hashjoin,indexscan,mergejoin,nestloop,seqscan
hashjoin,indexscan,mergejoin,seqscan
hashjoin,indexscan,nestloop
hashjoin,indexscan,nestloop,seqscan
hashjoin,indexscan,seqscan
hashjoin,mergejoin,nestloop,seqscan
hashjoin,mergejoin,seqscan
hashjoin,nestloop,seqscan
hashjoin,seqscan
indexonlyscan,indexscan,mergejoin
indexonlyscan,indexscan,mergejoin,nestloop
indexonlyscan,indexscan,mergejoin,nestloop,seqscan
indexonlyscan,indexscan,mergejoin,seqscan
indexonlyscan,indexscan,nestloop
indexonlyscan,indexscan,nestloop,seqscan
indexonlyscan,mergejoin
indexonlyscan,mergejoin,nestloop
indexonlyscan,mergejoin,nestloop,seqscan
indexonlyscan,mergejoin,seqscan
indexonlyscan,nestloop
indexonlyscan,nestloop,seqscan
indexscan,mergejoin
indexscan,mergejoin,nestloop
indexscan,mergejoin,nestloop,seqscan
indexscan,mergejoin,seqscan
indexscan,nestloop
indexscan,nestloop,seqscan
mergejoin,nestloop,seqscan
mergejoin,seqscan
nestloop,seqscan'''
all_48_hint_sets = all_48_hint_sets.split('\n')
all_48_hint_sets = [["enable_" + j for j in i.split(',')] for i in all_48_hint_sets]

# 收集选项
hint_sets = [
    set(hint_set) for hint_set in all_48_hint_sets
]

# 冗余性分析
redundant_sets = set()
for i in range(len(hint_sets)):
    for j in range(len(hint_sets)):
        if i != j and hint_sets[i].issubset(hint_sets[j]):
            redundant_sets.add(tuple(hint_sets[i]))

# 冲突性分析
conflict_pairs = [
    ({"enable_seqscan", "enable_indexscan"}, "seqscan 和 indexscan 冲突"),
    ({"enable_hashjoin", "enable_mergejoin"}, "hashjoin 和 mergejoin 冲突"),
    ({"enable_hashjoin", "enable_nestloop"}, "hashjoin 和 nestloop 冲突"),
    ({"enable_mergejoin", "enable_nestloop"}, "mergejoin 和 nestloop 冲突"),
]

conflicting_sets = set()
for hint_set in hint_sets:
    for conflict_pair, message in conflict_pairs:
        if hint_set & conflict_pair == conflict_pair:
            conflicting_sets.add(tuple(hint_set))

# 输出结果
print("冗余的 hintset:")
for r in redundant_sets:
    print(r)

print("\n存在冲突的 hintset:")
for c in conflicting_sets:
    print(c)
