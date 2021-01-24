# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1


VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

NER_TO_ID = {'PER': 0, 'GPE': 1, 'ORG': 2, 'STRING': 3, 'VALUE': 4}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46, 'pad': 47}

EVENT_TO_ID = {'process':0, 'coming_to_believe':1, 'change':2, 'motion':3, 'statement':4, 'motion_direction':5, 'getting':6, 'communication':7, 'perception_active':8, 'self_motion':9, 'action':10, 'know':11, 'telling':12, 'body_movement':13, 'aiming':14, 'expressing_publicly':15, 'placing':16, 'change_sentiment':17, 'face_or_solve_problem':18, 'giving_or_bringing':19, 'creating_or_damaging':20, 'earnings_and_loss':21, 'social_event':22, 'causation':23, 'forming_relationship':24, 'criminal_operation':25, 'presence':26, 'vocalization':27, 'ingestion':28,'O':29 }

#EVENT_TO_ID = {'process':0, 'perception_active':1, 'change_mind_or_sentiment':2, 'face_or_solve_problem':3, 'expressing_publicly':4, 'motion':5, 'social_event':6, 'placing_or_presence':7, 'giving_or_bringing':8, 'creating_or_damaging':9, 'getting_or_aiming':10, 'action':11, 'O':12}