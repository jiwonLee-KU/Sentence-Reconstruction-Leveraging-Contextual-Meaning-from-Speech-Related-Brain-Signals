
states = [
    'how', 'is', 'the', 'weather', 'today',
    'what', 'time', 'it',
    'take', 'a', 'picture',
    'hey', 'google',
    'turn', 'flashlight', 'on'
]

transition_prob = {
    'how': {'is': 1.0},
    'is': {'the': 0.5, 'it': 0.5},
    'the': {'weather': 0.5, 'flashlight': 0.5},
    'weather': {'today': 1.0},
    'today': {},
    'what': {'time': 1.0},
    'time': {'is': 1.0},
    'it': {},
    'take': {'a': 1.0},
    'a': {'picture': 1.0},
    'picture': {},
    'hey': {'google': 1.0},
    'google': {},
    'turn': {'the': 1.0},
    'flashlight': {'on': 1.0},
    'on': {}
}

sentences = [
    ['how', 'is', 'the', 'weather', 'today'],
    ['what', 'time', 'is', 'it'],
    ['take', 'a', 'picture'],
    ['hey', 'google'],
    ['turn', 'the', 'flashlight', 'on']
]

observation_prob = {}
for sentence in sentences:
    for word in sentence:
        if word not in observation_prob:
            observation_prob[word] = {word: 0.9}

# in same sentence words have same observation probability to 1/len(sentence)
for sentence in sentences:
    for word1 in sentence:
        for word2 in sentence:
            if word1 != word2:
                if word2 not in observation_prob[word1]:
                    observation_prob[word1][word2] = (1/len(sentence))

for word1 in observation_prob:
    for word2 in observation_prob:
        if word2 not in observation_prob[word1]:
            observation_prob[word1][word2] = 0.1
# print(observation_prob)

scaled_dict = {}

for key, inner_dict in observation_prob.items():
    total = sum(inner_dict.values())
    scaled_inner_dict = {k: v / total for k, v in inner_dict.items()}
    scaled_dict[key] = scaled_inner_dict

# print(scaled_dict)
observation_prob = scaled_dict

initial_prob = {
    'how': 1/5,
    'is': 0,
    'the': 0,
    'weather': 0,
    'today': 0,
    'what': 1/5,
    'time': 0,
    'it': 0,
    'take': 1/5,
    'a': 0,
    'picture': 0,
    'hey': 1/5,
    'google': 0,
    'turn': 1/5,
    'flashlight': 0,
    'on': 0
}


def BTS_viterbi(obs):
    V = [{}]
    path = {}

    for state in states:
        V[0][state] = initial_prob[state] * observation_prob[state][obs[0]]
        path[state] = [state]


    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for state in states:
            (prob, last_state) = max(
                (V[t - 1][last_state] * transition_prob[last_state].get(state, 0) * observation_prob[state][obs[t]], last_state)
                for last_state in states
            )
            V[t][state] = prob
            new_path[state] = path[last_state] + [state]

        path = new_path

    (prob, state) = max((V[-1][state], state) for state in states)
    return (prob, path[state])