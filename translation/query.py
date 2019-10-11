import statistics

def create_galago_query(query_dict, text):
    # print("printing query dict")
    # print(query_dict)
    probs = [query_dict[key] for key in query_dict]
    mean = statistics.mean(probs)

    text_splitted = text.split()
    for token in text_splitted:
        if token not in query_dict:
            query_dict[token] = mean
    probs = [query_dict[key] for key in query_dict]
    sm = sum(probs)
    new_query_dict = {}
    for key in query_dict:
        query_dict[key] /= sm
        if (query_dict[key] > 0.01):
            new_query_dict[key] = query_dict[key]
    query_dict = new_query_dict
    st = "#combine:"
    query_keys = query_dict.keys()
    for index, val in enumerate(query_keys):
        st += str(index) + "=" + str(query_dict[val])
        st += ":"
    st = st.rstrip(":")
    st += "("
    for index, val in enumerate(query_keys):
        st += str(val + " ")
    st += ")"
    return st