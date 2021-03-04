from itertools import chain,combinations, permutations
from tqdm import tqdm
import itertools
import pandas as pd

def Apriori(data,min_sup):
    freqtab={}
    item = list(data["item"])
    freqtab = update_freqtab(freqtab, item, min_sup)
    freq_pattern = {(key,): value for key, value in freqtab.items()}
    data = join_data(data)
    row_num = len(data)
    curr_itemset = list(freqtab.copy().keys())
    curr_itemset = list(combinations(curr_itemset, 2))
    k = 3
    while True:
        ck = scan_database(list(data["item"]), curr_itemset, min_sup)
        if not ck:
            break
        freq_pattern.update(ck)
        curr_itemset = generate_subset(list(ck.keys()), k)
        k += 1
    frequent_set = []
    for a in freq_pattern.keys():
        tmp = set()
        for p in a:
            tmp.add(p)
        frequent_set.append(tmp)
    large_itemset = complete_frequent(list(data["item"]), frequent_set)
    freq = make_dataframe(large_itemset, row_num)
    rules = make_rules(large_itemset, row_num)
    return freq, rules
    
def fp_growth(data,min_sup):
    freqtab = {}  
    item = list(data["item"])
    freqtab = update_freqtab(freqtab,item,min_sup)
    data = join_data(data)
    row_num = len(data)
    order_freq = ordered_freq(freqtab,data)
    order_item = list(order_freq["item"])
    dataset = Tofrozenset(order_item)
    Fptree, HeaderTable = createTree(dataset, min_sup)
    frequent_set = []
    mining(set([]), Fptree, HeaderTable, min_sup, frequent_set)
    large_itemset = complete_frequent(list(data["item"]),frequent_set)
    freq = make_dataframe( large_itemset ,row_num)
    rules = make_rules( large_itemset ,row_num)
    return freq,rules
      
def complete_frequent(data, frequent_set):
    freqtab = {}
    for i in frequent_set:
        for j in data:
            if i.issubset(j.split(' ')):
                key = tuple(i)
                if key in freqtab:
                    freqtab[key] += 1
                else:
                    freqtab[key] = 1
    return freqtab

    
def scan_database(data, curr_itemset, min_sup):
    freqtab = {}
    for i in curr_itemset:     
        for j in data:
            if set(i).issubset(j.split(' ')):  # check if Cn in database
                if i in freqtab:
                    freqtab[i] += 1
                else:
                    freqtab[i] = 1
                    
    for key, value in dict(freqtab).items():    #delete item < minsup
        if value < min_sup:
            del freqtab[key]  
    
    return freqtab

def generate_subset(curr_itemset,n):
    tmp =[]
    for i in curr_itemset:
        for j in i:
            tmp.append(j)
    return list(combinations(sorted(set(tmp)), n))
        

def update_freqtab(freqtab, item, min_sup):
    #create freq table
    for i in item:
        if i in freqtab:
            freqtab[i]+=1
        else:
            freqtab[i] = 1
            
    #delete item < minsup
    for key,value in dict(freqtab).items():  
        if value < min_sup:
            del freqtab[key]  
            
    #sort freq tab     
    freqtab = {k: v for k, v in sorted(freqtab.items(), key = lambda x:x[1], reverse = True)}
    return freqtab
                        
def join_data(df):
    df["item"] = df.groupby("TID")["item"].transform(lambda x : ' '.join(x))
    df = df.drop_duplicates()
    return df

def ordered_freq(freqtab,df):
    order = {}
    item_col =[]  
    tid_col = []
    for index,row in df.iterrows():
        tmp = row["item"].split()        
        tmp_dict = {}               # get item in freq tab and sort by its value
        for i in tmp:
            if i in freqtab:
                tmp_dict[i] = freqtab[i]
        tmp_dict = {k: v for k, v in sorted(tmp_dict.items(), key = lambda x:x[1], reverse = True)}
        item_col.append(list(tmp_dict.keys()))
        tid_col.append(row["TID"])
    order["TID"] = tid_col
    order["item"] = item_col
    a = pd.DataFrame.from_dict(order)
    return a

#Tree structure for Fp-Growth
class Treenode:
    def __init__(self,name,num,parentNode):
        self.item = name
        self.count = num
        self.nextnode = None   # jump to next node with the same item
        self.parent = parentNode
        self.children = {}
        
    def up_count(self,num):
        self.count += num
        
def Tofrozenset(dataset):
    Dict = {}
    for tran in dataset:
        Dict[frozenset(tran)] = 0
    for tran in dataset:
        Dict[frozenset(tran)] += 1
    return Dict
    
def createTree(dataset, min_support):
    
    HeaderTable = {}
    for transaction in dataset:
        for item in transaction:
            HeaderTable[item] = HeaderTable.get(item,0) + dataset[transaction]
    for item in list(HeaderTable):
        if HeaderTable[item] < min_support:
            del HeaderTable[item]
        if item == 'Null':
            del HeaderTable['Null']
    
    order_Header = [v[0] for v in sorted(HeaderTable.items(),key = lambda p: p[1], reverse=True)]
    if len(order_Header) == 0:
        return None,None
    
    for item in HeaderTable:
        HeaderTable[item] = [HeaderTable[item],None] 
    
    tree = Treenode('Null', 1, None) # Actually a node
    for transaction, count in dataset.items():
        order_set = [element for element in order_Header if element in transaction ]
        if len(order_set) > 0:
            update_tree(order_set, tree, HeaderTable, count)
            
    return tree, HeaderTable

def update_tree(order_set, treenode, HeaderTable, count):
    
    if order_set[0] not in treenode.children:
        treenode.children[order_set[0]] = Treenode(order_set[0], count, treenode)
        if (HeaderTable[order_set[0]][1] == None): # if current header is the rightmost node
            HeaderTable[order_set[0]][1] = treenode.children[order_set[0]]
        else:
            update_link(HeaderTable[order_set[0]][1], treenode.children[order_set[0]])
    else:
        treenode.children[order_set[0]].up_count(count)
    if len(order_set) > 1:
        update_tree(order_set[1:], treenode.children[order_set[0]], HeaderTable, count)
        
    
# different nodes contain same item must connect to each other through nextnode property in class Treenode
def update_link(current_node, next_node):
    
    while current_node.nextnode != None:
        current_node = current_node.nextnode
    current_node.nextnode = next_node
    
# trace back from certain node to inital null node
def uptransverse(node, one_branch):
    one_branch.append(node.item)
    
    while node.parent != None:
        one_branch.append(node.parent.item)
        node = node.parent

        
def find_subtree_patt(one_item, current_node):
    cond_pattern_bases = {}
    while current_node != None:
        one_branch = []
        uptransverse(current_node, one_branch)
        if len(one_branch) > 1: # this branch bigger than only 'Null'
            cond_pattern_bases[frozenset(one_branch[1:])] = current_node.count
        current_node = current_node.nextnode
    return cond_pattern_bases

def mining(prenodes, tree, HeaderTable, min_support,frequent_set):
    if HeaderTable != None:
        one_itemset = [v[0] for v in sorted(HeaderTable.items(),key = lambda p: p[1][0])]
        for one_item in one_itemset:
            new_freq_set = prenodes.copy()
            new_freq_set.add(one_item)
            
            frequent_set.append(new_freq_set)
            cond_pattern_bases = find_subtree_patt(one_item, HeaderTable[one_item][1])
            cond_tree, cond_header = createTree(cond_pattern_bases, min_support)
            
            if cond_header != None:
                mining(new_freq_set, cond_tree, cond_header, min_support, frequent_set)

        
def make_dataframe(freq_pattern,row_num):
    tmp = pd.DataFrame(list(freq_pattern.items()), columns = ['itemsets','support'])
    tmp = tmp.sort_values(by=['support'])
    tmp  = tmp[['support', 'itemsets']]
    return tmp
    
def make_rules(freq_pattern,row_num):
    rules_dict = {}
    support = []
    for i in freq_pattern:
        if len(i) > 1:
            sub = get_subsets(i)
            for j in list(sub):
                key = str(j) + " -> " +str(tuple(set(i)-set(j)))
                per_j  = list(permutations(j))
                ind = 0
                while j not in freq_pattern:
                    j = per_j[ind]
                    ind += 1
                rules_dict[key] = freq_pattern[i]/freq_pattern[j]
                support.append(freq_pattern[i]/row_num)
    tmp = pd.DataFrame({'rules' : list(rules_dict.keys()), 'confidence' : list(rules_dict.values()), 'support' :support})
    return tmp

def get_subsets(v):
    return chain.from_iterable(sorted(combinations(sorted(set(v)),r))  for r in range(1,len(v)) )  #get subsets for make_rules
