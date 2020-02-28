import json
import re
import nltk 
from nltk import pos_tag
from nltk.tokenize import MWETokenizer
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from datetime import datetime
import parsedatetime as pdt  
#import spacy
from parsedatetime import Constants
from nltk.tokenize import WordPunctTokenizer
import os
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words += [',', '.']

#os.chdir("/Users/duynguyen/DuyNguyen/Gitkraken/Elasticsearch_LSC_SettingUp/")

#nlp = spacy.load("en_core_web_sm")

#c = pdt.Constants()
#c.uses24 = True

#cal = pdt.Calendar(c)
#now = datetime.now()

simpletime = ['at', 'around', 'about', 'on']
period = ['while', "along", "as"]

preceeding = ['before', "afore"]
following = ['after']
location = ['across', 'along', 'around', 'at', 'behind', 'beside', 'near', 'by', 'nearby', 'close to', 
            'next to', 'from', 'in front of', 'inside', 'in', 'into', 'off', 'on',
            'opposite', 'out of', 'outside', 'past', 'through', 'to', 'towards']

all_words = period + preceeding + following 
all_prep = simpletime + period + preceeding + following
pattern = re.compile(f"\s?({'|'.join(all_words)}+)\s")

##### TIME TAGGER #####
# Preprocess tagging, mostly about time

def find_regex(regex, text, escape=False):
    regex = re.compile(regex, re.IGNORECASE + re.VERBOSE)
    for m in regex.finditer(text):
        result = m.group()
        start = m.start()
        while len(result)>0 and result[0] == ' ':
            result = result[1:]
            start += 1
        while len(result)>0 and result[-1] == ' ':
            result = result[:-1]
        yield (start, start + len(result), result)

class TimeTagger:
    def __init__(self):
        regex_lib = Constants()
        self.all_regexes = []
        for key, r in regex_lib.cre_source.items():
            # if key in ["CRE_MODIFIER"]:
            #     self.all_regexes.append(("TIMEPREP", r))
            if key in ["CRE_TIMEHMS", "CRE_TIMEHMS2",
                    "CRE_RTIMEHMS", "CRE_RTIMEHMS"]:
                self.all_regexes.append(("TIME", r)) # TIME (proper time oclock)
            elif key in ["CRE_DATE", "CRE_DATE3", "CRE_DATE4", "CRE_MONTH", "CRE_DAY", "",
                    "CRE_RDATE", "CRE_RDATE2"]:
                self.all_regexes.append(("DATE", r)) # DATE (day in a month)
            elif key in ["CRE_TIMERNG1", "CRE_TIMERNG2", "CRE_TIMERNG3", "CRE_TIMERNG4",
                    "CRE_DATERNG1", "CRE_DATERNG2", "CRE_DATERNG3", "CRE_NLP_PREFIX"]:
                self.all_regexes.append(("TIMERANGE", r)) # TIMERANGE
            elif key in ["CRE_UNITS", "CRE_QUNITS"]:
                self.all_regexes.append(("PERIOD", r)) # PERIOD
            elif key in ["CRE_UNITS_ONLY"]:
                self.all_regexes.append(("TIMEUNIT", r)) # TIMEUNIT
            elif key in ["CRE_WEEKDAY"]:
                self.all_regexes.append(("WEEKDAY", r)) # WEEKDAY
        # Added by myself
        self.all_regexes.append(("TIMEOFDAY", r"\b(afternoon|noon|morning|evening|night|twilight)\b"))
        self.all_regexes.append(("TIMEPREP", r"\b(before|after|while|late|early)\b"))

    def merge_interval(self, intervals):
        if intervals:
            intervals.sort(key=lambda interval: interval[0])
            merged = [intervals[0]]
            for current in intervals:
                previous = merged[-1]
                if current[0] <= previous[1] and current[-1] == previous[-1]:
                    if current[1] > previous[1]:
                        previous[1] = current[1]
                        previous[2] = current[2]
                else:
                    merged.append(current)
            return merged
        return []

    def find_time(self, sent):
        results = []
        for kind, r in self.all_regexes:
            for t in find_regex(r, sent):
                results.append([*t, kind])
        return self.merge_interval(results)

    def tag(self, sent):
        times = self.find_time(sent)
        intervals = dict([(time[0], time[1]) for time in times])
        tag_dict = dict([(time[2], time[3]) for time in times])
        tokenizer = WordPunctTokenizer()
        # for a in [time[2] for time in times]:  
        #     tokenizer.add_mwe(a.split())
        
        # --- FIXED ---
        original_tokens = tokenizer.tokenize(sent)
        original_tags = pos_tag(original_tokens)
        #print(original_tags)
        # --- END FIXED ---

        tokens = []
        current = 0
        for span in tokenizer.span_tokenize(sent):
            if span[0] < current:
                continue
            if span[0] in intervals:
                tokens.append(f'__{sent[span[0]: intervals[span[0]]]}')
                current = intervals[span[0]]
            else:
                tokens.append(sent[span[0]:span[1]])
                current = span[1]

        tags = pos_tag(tokens)
        
        new_tags = []
        for word, tag in tags:
            if word[:2] == '__':
                new_tags.append((word[2:], tag_dict[word[2:]]))
            else:
                tag = [t[1] for t in original_tags if t[0] == word][0] # FIXED
                new_tags.append((word, tag))
        return new_tags
    

#time_tagger = TimeTagger()
#time_tagger.tag("monday very early morning, 15 minutes before 15:00pm, around 13:00 from May 10th to 10 May 2016, between 2am to 4am")
#time_tagger.tag("After cooking, I went to the church")

##### GENERAL TAGGER #####
address_and_gps = json.load(open("data/address_and_gps.json"))
all_address = '|'.join([re.escape(a) for a in address_and_gps])

##### TAGGER #####
# Run time tagger, then assign location in location json file

class Tagger:
    def __init__(self, address_and_gps):
        self.tokenizer = MWETokenizer()
        self.time_tagger = TimeTagger()
        for a in address_and_gps:  
            self.tokenizer.add_mwe(a.split())
        # Rules defined
        self.specials = {
            "QUANTITY": ["at least", "more than", "less than", "at most", 
                         "not more than", "a number of"],
            "IN": ["in front of", "called"],
            "NN": ["cafe sign", "traffic light", "fire hydrant", "stop sign", "parking meter",
                   "baseball bat", "baseball glove", "cell phone", "teddy bear", "hair drier"
                   "airport vehicles", "airport vehicle", "screen"],
            "SPACE": ["living room", "fastfood restaurant", "restaurant kitchen", "restaurant",
                      "dining hall", "food court", "butchers shop", "restaurant patio", 
                      "coffee shop", "room", "hotel room", "kitchen", "office",
                      "airport", "salon"],
            "POSITION": ["side", "foreground", "background", "right", "left",
                         "image"],
            "LOCATION": ["home", "school", "oslo", "norway", "hotel", "tromso", "bank", 
                         "ireland", "china", "japan", "vietnam", 'dcu', 'dublin', 
                         'dublin city university'],
            "TOBE": ["am", "is", "are", "be", "is being", "am being", "are being", "being"],
            "WAS": ["was", "were", "had been", "have been"],
            "TIMEPREP": ["prior to", "then"],
            "POSITION_PREP": ["near", "distance to"]
        }
      
        for tag in ["QUANTITY", "IN", "NN", "SPACE", "POSITION", "LOCATION", 
                    "TOBE", "WAS", "TIMEPREP", "POSITION_PREP"]:
            for keyword in self.specials[tag]:
                if ' ' in keyword:
                    self.tokenizer.add_mwe(keyword.split())

    def tag(self, sent):
        sent = sent.replace(',', ' , ')
        token = self.tokenizer.tokenize(sent.lower().split()) # tokenize places and address
        sent = re.sub(r'\b(i)\b','I', ' '.join(token)) # replace i to I
        tags = self.time_tagger.tag(sent)
        new_tags = []
        for word, tag in tags:
            if '_' in word:
                new_tag = None
                word = word.replace('_', ' ')
                for t in ["QUANTITY", "IN", "NN", "SPACE", "POSITION", "POSITION_PREP",
                          "LOCATION", "TOBE", "WAS", "TIMEPREP"]:
                    if word in self.specials[t]:
                        new_tag = t
                        break
                if new_tag is None:
                    tag = 'LOCATION'
                else:
                    tag = new_tag
            else:
                for t in ["QUANTITY", "IN", "NN", "SPACE", "POSITION", "POSITION_PREP",
                          "LOCATION", "TOBE", "WAS", "TIMEPREP"]:
                    if word in self.specials[t]:
                        tag = t
                        break
            if tag in ['NN', 'NNS']: # fix error NN after NN --> should be NN after VBG
                try:
                    t1, t2 = new_tags[-1]
                except:
                    t1, t2 = None, None
                if t2 in ['NN', 'NNS']:
                    new_tags[-1] = (t1, 'VBG')
            new_tags.append((word, tag))
        return new_tags

#tagger = Tagger(address_and_gps)
#tagger.tag("After cooking, I went to the church near the Helix")

def flatten_tree(t):
    return " ".join([l[0] for l in t.leaves()])

def flatten_tree_tags(t, pos):
    if isinstance(t, nltk.tree.Tree):
        if t.label() in pos:
            return [flatten_tree(t), t.label()]
        else:
            return [flatten_tree_tags(l, pos) for l in t]
    else:
        return t

##### LOCATION TAG #####
class Location:
    def __init__(self, tree_tags):
        #print(tree_tags)
        self.name = []
        self.prep = ""
        self.extra = ""
        self.tree = tree_tags
        self.extract(tree_tags)
        self.name = ", ".join(self.name)
        #self.extra = " ".join(self.extra)
        if self.prep == "":
            self.prep = "None"
    
    def extract(self, t):
        if isinstance(t, nltk.tree.Tree):
            [self.extract(l) for l in t]
        else:
            if t[1] in ["LOCATION", "SPACE", "NN", "NNS"]:
                self.name.append(t[0])
            elif t[1] in ["PREP", "IN"]:
                self.prep = t[0]
            elif t[1] in ["RB", "PRP$"] and self.extra == "":
                if t[0] == "not" or t[0] != "my":
                    self.extra = "not"
                else:
                    self.extra = ""
    
    def __repr__(self):
        return f"({self.prep}) <{self.extra}> {self.name}"

##### OBJECT TAG #####
class Object:
    def __init__(self, tree_tags):
        #print(tree_tags)
        self.name = []
        self.position = []
        self.quantity = []
        tree_tags = flatten_tree_tags(tree_tags, ["NN"])
        self.extract(tree_tags)
        self.tree = tree_tags
        self.position = " ".join(self.position)
        if not self.position:
            self.position = "None"
        
    def extract(self, t):
        if not isinstance(t[0], str):
            [self.extract(l) for l in t]
        else:
            if t[1] in ["NN", "NNS"] :
                self.name.append(t[0])
                if len(self.quantity) < len(self.name):
                    if self.name[-1][-1] == "s":
                        self.quantity.append("many")
                    else:
                        self.quantity.append("one")
            elif t[1] in ["QUANTITY", "CD"]:
                self.quantity.append(t[0])
            elif t[1] in ["POSITION"]:
                self.position.append(t[0])
    
    def __repr__(self):
        return f"({self.position}) {'; '.join([f'{q}, {n}' for q, n in zip(self.quantity, self.name)])}"

##### TIME TAG #####
class Time:
    def __init__(self, tree_tags):
        #print(tree_tags)
        self.name = []
        self.period = []
        self.prep = []
        tree_tags = flatten_tree_tags(tree_tags, ["PERIOD", "TIMEOFDAY"])
        self.extract(tree_tags)
    
    def extract(self, t):
        if not isinstance(t[0], str):
            [self.extract(l) for l in t]
        else:
            if t[1] in ["TIMEOFDAY", "WEEKDAY", "TIME", "DATE"]:
                self.name.append(t[0])
            elif t[1] in ["PERIOD", "CD", "NN"]:
                self.period.append(t[0])
            elif t[1] in ["IN", "TIMEPREP", "TO"]:
                try:
                    if self.prep[-1] != 'from' and self.prep[-1] != 'to':
                        self.prep.append(t[0])
                except:
                    self.prep.append(t[0])

    def __repr__(self):
        #mystr = list(map(self.func, self.prep, self.period, self.name))
        #return ", ".join(mystr)
        return "; ".join(self.prep + self.period + self.name)

##### ACTION TAG #####
class Action:
    def __init__(self, tree_tags):
        #print(tree_tags)
        self.name = []
        self.in_obj = []
        self.in_loc = []
        self.obj = []
        self.loc = []
        self.time = []
        tree_tags = flatten_tree_tags(tree_tags, ["VERB_ING", "PAST_VERB", "VERB",
                                                  "SPACE", "NN", "NNS"])
        self.extract(tree_tags)
        self.calibrate()
        self.tree = tree_tags
        self.func = lambda x, y, z, t: f"{x};{y};{z};{t}" # time, action, object, location

    def extract(self, t):
        if not isinstance(t[0], str):
            [self.extract(l) for l in t]
        else:
            if t[1] in ["VERB_ING", "PAST_VERB", "VERB"]:
                self.name.append(t[0])
            elif t[1] in ["NN", "NNS"]: # prior object than location
                self.obj.append(t)
                if (len(self.name)-len(self.in_obj)) == 1:
                    self.in_obj.append(t[0])
            elif t[1] in ["LOCATION", "SPACE"]:
                self.loc.append(t)
                if (len(self.name)-len(self.in_loc)) == 1:
                    self.in_loc.append(t[0])
            elif t[1] in ["TIMEPREP"]:
                if t[0] == 'after':
                    self.time.append('past')
                elif t[0] == 'then' or t[0] == 'before':
                    self.time.append('future')
                else:
                    self.time.append('present')
            # In case there is no TIMEPREP --> default present
            if len(self.time) < len(self.name):
                self.time.append('present')
    
    def calibrate(self):
        n_action = len(self.name)
        for i in range(n_action - len(self.time)):
          self.time.append('present')
        for i in range(n_action - len(self.in_obj)):
          self.in_obj.append('')
        for i in range(n_action - len(self.in_loc)):
          self.in_loc.append('')

    def __repr__(self):
        mystr = list(map(self.func, self.time, self.name, self.in_obj, self.in_loc))
        return ", ".join(mystr)

class ElementTagger:
    def __init__(self):
        grammar = r"""
                      WEEKDAY: {<IN><DT><WEEKDAY>}
                      POSITION_PREP: {<JJ>*<POSITION_PREP>}
                      TIMEOFDAY: {<DT>?<RB>?<JJ>*<TIMEOFDAY>}
                      PERIOD: {<QUANTITY>?<PERIOD>}
                      TIMEPREP: {(<RB>|<PERIOD>)?<TIMEPREP>}
                      SPACE: {<SPACE><LOCATION>}
                      LOCATION: {(<OBJECT><LOCATION>|<SPACE>|<NNP>)(<VBD>|<VBN>|<\,>)<DT>?(<OBJECT><LOCATION>|<SPACE>|<NNP>)}
                      LOCATION: {<RB>?<IN>?(<DT>|<PRP\$>)?(<LOCATION>|<SPACE>)+}
                
                      NN: {(<NN>|<NNS>)+(<IN>(<NNS>|<NN>))?}
                      OBJECT: {(<EX><TOBE>|<QUANTITY>?<CD>?|<DT>|<PRP\$>)<JJ>*<SPACE>?(<NN>|<NNS>)+}
                      
                      TIME: {<TIMEPREP>?<IN>?(<DT>|<PRP\$>)?(<TIMEOFDAY>|<DATE>)}
                            {(<IN>|<TO>|<RB>)?(<TIMEPREP>|<IN>)?<TIME>}
                      POSITION: {(<IN>|<TO>)?(<DT>|<PRP\$>)<JJ>?<POSITION>}
                                {(<IN>|<TO>)<PRP>}
                      OBnPOS: {(<OBJECT><IN>)?<OBJECT><TOBE>?<JJ>?<POSITION>}
                      
                      VERB_ING: {<VBG><RP>?(<TO>|<IN>)?}
                      VERB_ING: {<VERB_ING>((<CC>|<\,>|<\,><CC>)<VERB_ING>)+}
                      ACTION_ING: {<TOBE>?<VERB_ING>}
                      ACTION_ING: {<TIMEPREP>?<ACTION_ING><RP>?(<TO>|<IN>)?<DT>?((<NN>|<OBJECT>)?(<LOCATION>|<SPACE>)|<OBnPOS>|<OBJECT>)}
                      ACTION_ING: {<TIMEPREP>?<ACTION_ING>(<CC>|<\,>)<ACTION_ING>}
                                  {<TIMEPREP><ACTION_ING>}
                      
                      PAST_VERB: {<RB>?(<WAS><RB>?<VBG>|<VBD>|<VBN>|<VBD><VBG>)<RB>?(<TO>|<IN>)?}
                      PAST_VERB: {<TIMEPREP>?<PAST_VERB>((<CC>|<\,>|<\,><CC>)<PAST_VERB>)+}
                      PAST_ACTION: {<TIMEPREP>?(<CC>|<PRP>)?<PAST_VERB><DT>?((<NN>|<OBJECT>)?(<LOCATION>|<SPACE>)|<OBnPOS>|<OBJECT>)<ACTION_ING>?}
                                   {<PRP><WAS><IN>?(<LOCATION>|<SPACE>)<ACTION_ING>}
                                   {<WAS><ACTION_ING>}
                                   {<TIMEPREP>?(<CC>|<PRP>)?<PAST_VERB><DT>?(?!<PAST_VERB>)}
                      
                      VERB: {(<VB>|<VBG>|<VBP>)<RP>?(<TO>|<IN>)?<DT>?<VB>?}
                      VERB: {<VERB>((<CC>|<\,>|<\,><CC>)<VERB>)+}
                      ACTION: {<TIMEPREP>?<PRP>?<VERB>((<NN>|<OBJECT>)?(<LOCATION>|<SPACE>)|<OBnPOS>|<OBJECT>)?}
                              {<TOBE><LOCATION>?<ACTION_ING>}
                              {<OBJECT><TOBE><VBN>}

                   """
                   #OBJECTS: {<OBJECT>((<CC>|<\,>)<OBJECT>)+}
                   #PERIOD_ACTION: {<TIMEPREP>(<ACTION_ING>|<PAST_ACTION>|<PAST_VERB>)}
        self.cp = nltk.RegexpParser(grammar)
    
    def express(self, t):
        if isinstance(t, nltk.tree.Tree):
            expressions = [self.express(l) for l in t]
            return f"({','.join(expressions)})"
        else:
            return t[0]

    def update(self, elements, t):
        if isinstance(t, nltk.tree.Tree):
            label = t.label()
        else:
            label = t[1]
        if label in ["LOCATION"]:
            elements["location"].append(Location(t))
        elif label in ["WEEKDAY", "TIMEOFDAY", "TIME"]:
            elements["time"].append(Time(t))
        elif label in ["PERIOD", "TIMEPREP"]:
            elements["period"].append(Time(t))
        elif label in ["OBJECT", "OBnPOS"]:
            elements["object"].append(Object(t))
        elif label in ["PAST_ACTION", "ACTION_ING", "ACTION"]:
            action_process = Action(t)
            elements["action"].append(action_process)
            for idx in range(len(action_process.obj)):
                elements['object'].append(Object(action_process.obj[idx]))
            for idx in range(len(action_process.loc)):
                elements['location'].append(Location(action_process.loc[idx]))
        #else:
        #    print("UNUSED:", t)
        return elements

    def tag(self, tags):
        elements = {"location": [],
                    "action": [],
                    "object":[],
                    "period":[],
                    "time":[]}
        for key, val in tags:
            if key in ['dinner', 'lunch', 'breakfast']:
                elements['time'].append(key)

        for n in self.cp.parse(tags):
            self.update(elements, n)
            # print(n)
        
        # Convert to string and Filter same result
        for key, value in elements.items():
            for idx_val in range(len(value)):
                value[idx_val] = str(value[idx_val])
            elements[key] = list(set(value))

        # Filter useless information
        idx_obj = 0
        useless_object = ['someone', 'of']
        while idx_obj < len(elements['object']):
            temp = elements['object'][idx_obj].split(', ')
            if len(temp[1]) == 1 or temp[1] in useless_object: # object only has 1 character --> useless for search
                elements['object'].pop(idx_obj)
            else:
                temp_name = temp[1].split()
                for val in useless_object:
                    try:
                        temp_name.remove(val)
                    except ValueError:
                        pass
                temp_name = " ".join(temp_name)
                elements['object'][idx_obj] = ", ".join([temp[0], temp_name])
                idx_obj += 1

        return elements

def extract_info_from_tag(tag_info):
    obj_dict = {}
    obj_past = []
    obj_present = []
    obj_future = []

    loc_dict = {}
    loc_past = []
    loc_present = []
    loc_future = []

    for action in tag_info['action']:
        extract_action = action.split(';')
        action_time = extract_action[0]
        if action_time == 'past':
            try:
                if extract_action[2] != '':
                    obj_past.append(extract_action[2]) 
                if extract_action[3] != '':
                    loc_past.append(extract_action[3])
            except:
                pass
        if action_time == 'present':
            try:
                if extract_action[2] != '':
                    obj_present.append(extract_action[2])
                if extract_action[3] != '':
                    loc_present.append(extract_action[3])
            except:
                pass
        if action_time == 'future':
            try:
                if extract_action[2] != '':
                    obj_future.append(extract_action[2])
                if extract_action[3] != '':
                    loc_future.append(extract_action[3])
            except:
                pass

    for obj in tag_info['object']:
        extract_object = obj.split(', ')
        obj_is = extract_object[1]
        if obj_is not in obj_past and obj_is not in obj_present and obj_is not in obj_future:
            if len(obj_past) > 0:
                obj_past.append(obj_is)
            if len(obj_present) > 0:
                obj_present.append(obj_is)
            if len(obj_future) > 0:
                obj_future.append(obj_is)

    for loc in tag_info['location']:
        extract_loc = loc.split()
        loc_is = extract_loc[2]
        negative = True if len(extract_loc[1]) > 2 else False
        if negative: # since it is NOT --> dont append it
            try:
                loc_past.remove(loc_is)
                loc_present.remove(loc_is)
                loc_future.remove(loc_is)
            except:
                pass
        else:
            if loc_is not in loc_past and loc_is not in loc_present and loc_is not in loc_future:
                if len(loc_past) > 0:
                    loc_past.append(loc_is)
                if len(loc_present) > 0:
                    loc_present.append(loc_is)
                if len(loc_future) > 0:
                    loc_future.append(loc_is)

    query_dict = {}      
    query_dict['present'] = obj_present + loc_present
    query_dict['past'] = obj_past + loc_past
    query_dict['future'] = obj_future + loc_future

    obj_dict['present'] = obj_present
    obj_dict['past'] = obj_past
    obj_dict['future'] = obj_future 

    loc_dict['present'] = loc_present
    loc_dict['past'] = loc_past
    loc_dict['future'] = loc_future 
    
    return query_dict, obj_dict, loc_dict
    

init_tagger = Tagger(address_and_gps)
e_tag = ElementTagger()

#sent = 'after walking to the bus station, I took the bus 109 to DCU which took 2 hours, then walked to my office, then played games on my laptop'
#sent = 'after walking to bus station, go to DCU, 2 hours, then walked to office and played game on laptop'

def extract_info_from_sentence(sent):
    sent = sent.replace(', ', ',')
    tense_sent = sent.split(',')

    past_sent = ''
    present_sent = ''
    future_sent = ''

    for current_sent in tense_sent:
        split_sent = current_sent.split()
        if split_sent[0] == 'after':
            past_sent += ' '.join(split_sent) + ', '
        elif split_sent[0] == 'then':
            future_sent += ' '.join(split_sent) + ', '
        else:
            present_sent += ' '.join(split_sent) + ', '

    past_sent = past_sent[0:-2]
    present_sent = present_sent[0:-2]
    future_sent = future_sent[0:-2]

    list_sent = [past_sent, present_sent, future_sent]

    info = {}
    info['past'] = {}
    info['present'] = {}
    info['future'] = {}

    for idx, tense_sent in enumerate(list_sent):
        tags = init_tagger.tag(tense_sent)
        obj = []
        loc = []
        period = []
        time = []
        timeofday = []
        for word, tag in tags:
            if word not in stop_words:
                if tag in ['NN', 'NNS']:
                    obj.append(word)
                if tag in ['SPACE', 'LOCATION']:
                    loc.append(word)
                if tag in ['PERIOD']:
                    period.append(word)
                if tag in ['TIMEOFDAY']:
                    timeofday.append(word)
                if tag in ['TIME', 'DATE', 'WEEKDAY']:
                    time.append(word)
        if idx == 0:
            info['past']['obj'] = obj
            info['past']['loc'] = loc
            info['past']['period'] = period
            info['past']['time'] = time
            info['past']['timeofday'] = timeofday            
        if idx == 1:
            info['present']['obj'] = obj
            info['present']['loc'] = loc
            info['present']['period'] = period
            info['present']['time'] = time
            info['present']['timeofday'] = timeofday   
        if idx == 2:
            info['future']['obj'] = obj
            info['future']['loc'] = loc
            info['future']['period'] = period
            info['future']['time'] = time
            info['future']['timeofday'] = timeofday           
    
    return info     

def extract_info_from_sentence_full_tag(sent):
    sent = sent.replace(', ', ',')
    tense_sent = sent.split(',')

    past_sent = ''
    present_sent = ''
    future_sent = ''

    for current_sent in tense_sent:
        split_sent = current_sent.split()
        if split_sent[0] == 'after':
            past_sent += ' '.join(split_sent) + ', '
        elif split_sent[0] == 'then':
            future_sent += ' '.join(split_sent) + ', '
        else:
            present_sent += ' '.join(split_sent) + ', '

    past_sent = past_sent[0:-2]
    present_sent = present_sent[0:-2]
    future_sent = future_sent[0:-2]

    list_sent = [past_sent, present_sent, future_sent]

    info = {}
    info['past'] = {}
    info['present'] = {}
    info['future'] = {}

    for idx, tense_sent in enumerate(list_sent):
        if len(tense_sent) > 2:
            tags = init_tagger.tag(tense_sent)
            info_full = e_tag.tag(tags)
            print(tags)
            print(info_full)
            obj = []
            loc = []
            period = []
            time = []
            timeofday = []
            
            if len(info_full['object']) != 0:
                for each_obj in info_full['object']:
                    split_term = each_obj.split(', ')
                    if len(split_term) == 2:
                        obj.append(split_term[1])
            
            if len(info_full['period']) != 0:
                for each_period in info_full['period']:
                    if each_period not in ['after', 'before', 'then', 'prior to']:
                        period.append(each_period)

            if len(info_full['location']) != 0:
                for each_loc in info_full['location']:
                    split_term = each_loc.split('> ')
                    if split_term[0][-3:] != 'not':
                        word_tag = pos_tag(split_term[1].split())
                        final_loc = []
                        for word, tag in word_tag:
                            if tag not in ['DT']:
                                final_loc.append(word)
                        final_loc = ' '.join(final_loc)
                        loc.append(final_loc)
            
            if len(info_full['time']) != 0:
                for each_time in info_full['time']:
                    if 'from' in each_time or 'to' in each_time:
                        timeofday.append(each_time)
                    else:
                        timetag = init_tagger.time_tagger.tag(each_time)
                        if timetag[-1][1] in ['TIME', 'TIMEOFDAY']:
                            timeofday.append(each_time)
                        elif timetag[-1][1] in ['WEEKDAY', 'DATE']:
                            time.append(timetag[-1][0])

            if idx == 0:
                info['past']['obj'] = obj
                info['past']['loc'] = loc
                info['past']['period'] = period
                info['past']['time'] = time
                info['past']['timeofday'] = timeofday            
            if idx == 1:
                info['present']['obj'] = obj
                info['present']['loc'] = loc
                info['present']['period'] = period
                info['present']['time'] = time
                info['present']['timeofday'] = timeofday   
            if idx == 2:
                info['future']['obj'] = obj
                info['future']['loc'] = loc
                info['future']['period'] = period
                info['future']['time'] = time
                info['future']['timeofday'] = timeofday           
    
    return info     

#sent = 'flower, vase'
#a = extract_info_from_sentence_full_tag(sent)

'''
example_sent = """In a coffee shop called The Helix; one person; plastic plant on the right. 
                  Keys on table; cafe sign on the left. 
                  Walked to the cafe two minutes ago. 
                  My colleague wears a white tshirt; red coffee cup. 
                  Monday. After, I drive to the shop."""

sent1 = """In a coffee shop in the afternoon called the Helix with at least one person in the background and a plastic plant on my right side.
There are keys on the table in front of me and you can see the cafe sign on the left side.
I walked to the cafe and it took less than two minutes to get there.
I am speaking to my colleague in the foreground who is wearing a white shirt and drinking coffee from a red paper cup.
It is a Monday and immediately after having the coffee, I drive to the shop."""
sent2 = "I am at home in the very early morning and I am in my living room watching football on the television. There is a lamp to the right of the image and a box of things to the left of the image. After watching television, I use a computer and then drive to work. It is a Thursday."
sent3 = "I am at home scanning receipts using a portable scanner. There are a number of receipts on the table in front of me and one long receipt is being scanned. Prior to this, I had been in the kitchen looking at a bowl of fruit. My arm and hand is clearly visible in the image and some vinyl records are visible to the left. It is late at night on a Monday."
sent4 = "I am walking out to an airplane across the airport apron. I stayed in an airport hotel on the previous night before checking out and walking a short distance to the airport. The weather is very nice, but cold, with a clear blue sky. There is a man walking to the airplane in front of me with a blue jacket, green shoes and a black bag. Red airport vehicles are visible in the image also, along with a small number of passengers walking to, and boarding the plane. I am in Oslo, Norway and it is early in on a Monday morning."
sent5 = "I am taking a photo of a white building with a unique blade-like design. The weather was cloudy and it was getting dark, being evening time. There are a number of buildings clearly visible in the image, including a hotel and a Norwegian style house. I had just walked from a sushi restaurant to the hotel where I was staying and I had taken the photo just before entering the hotel. A large yellow pipe is also visible in the image. Just before taking the photo, I had been walking beside the sea. This happened on a Wednesday."
sent6 = "Find the time when I was looking at an old clock, with flowers visible. There was a lamp also, and a small blue monster (perhaps a long rabbit) watching me. Maybe there were two monsters. It was a Monday or a Thursday. I was at home and in a bedroom."
sent7 = "A red car beside a white house on a cloudy day. I had driven for over an hour to get here. It was a saturday in August and it was in the early afternoon."
sent8 = "Walking through an airbridge after a flight in the early morning after a flight of about two hours. After the airport, I immediately got a taxi to a meeting. I think it was a cloudy day on a Monday. I was in Tromso in Norway."
sent9 = "I remember a doll’s house, a white dolls house. There were other people there, and candles too, I remember candles. There was some nice village scene in front of a lake on a picture. It was a Saturday."
sent10 = "Catching my reflection in the bathroom mirror early on a Saturday morning when I was preparing to brush my teeth. There were pink wall tiles and I was wearing a white shirt. I had just finished watching some TV and then I drove somewhere."
'''

'''
def main():
    sent = "after having dinner at a restaurant, went to a train station, then watched tv at room"
    init_tagger = Tagger(address_and_gps)
    tags = init_tagger.tag(sent)
    print(tags)
    e_tag = ElementTagger()
    info = e_tag.tag(tags)
    query_dict, obj_dict, loc_dict = extract_info_from_tag(info)
'''