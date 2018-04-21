from config import Config
from dataobject import CoNLLDataset
from data_utils import get_processing_word
from model import Model
import os
import re
import csv

class EntityRecognition():
    
    maximum = 4
    unprocessed_file_folder = "startupdaily"
    model = None
    doc_start_name = "-DOCSTART- -X- O O"
    unprocessed_words = None
    #process_folder = 'articles/200_testing'
    process_folder = 'articles/checking'
    process_result = 'articles_result'

    def __init__(self):
        config = Config()
        self.model = Model(config)
        self.model.build()
        self.model.restore_session(config.dir_model)

    def process(self):
        with open(self.process_result+"/"+"files.csv", 'w') as filecsv:
            filewriterfiles = csv.writer(filecsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            filewriterfiles.writerow(['Name of the file', 'Final founders', 'Final organisations', 'All names', 'All organisations'])
            for file in os.listdir(self.process_folder):
                if file not in '.DS_Store' and os.path.isdir(self.process_folder+"/"+file) != True:
                    print("Processing: {}".format(file))
                    text_file = open(self.process_folder+"/"+file, "r") 
                    text = text_file.read() 
                    words_raw = self.split_sentences(text)
                    self.unprocessed_words = words_raw
                    preds = self.model.predict(words_raw)
                    person = []
                    full_name = []
                    names = []
                    full_org_name = []
                    org_name = []
                    for i, word in enumerate(words_raw):
                        #print("Word: {} | Prediction: {}".format(word, preds[i]))
                        if preds[i] == "I-PER" and preds[i] not in person:
                            if self.refine_check(word, "person") == True:
                                person.append(word)
                                full_name.append(word)
                            else:
                                if len(full_name) > 0:
                                    names.append(full_name)
                                full_name = []

                        elif preds[i] == "I-ORG" and preds[i] not in org_name:
                            if self.refine_check(word, "org") == True:
                                full_org_name.append(word)
                            else:
                                if len(full_org_name) > 0:
                                    org_name.append(full_org_name)
                                full_org_name = []
                        else:
                            if len(full_name) > 0:
                                names.append(full_name)
                            full_name = []

                            if len(full_org_name) > 0:
                                org_name.append(full_org_name)
                            full_org_name = []


                        word = "{} -X- O {}".format(word, preds[i])

                    #print(person)
                    names = self.remove_duplicates(names, "person")
                    all_names, merged_pers = self.further_process(names, "person")
                    
                    #Organisation
                    orgs = self.remove_duplicates(org_name, "org")
                    all_orgs, merged_orgs = self.further_process(orgs, "org")
                    
                    #Important organisation
                    final_org = self.find_important(merged_orgs, text)

                    #Important names
                    imporatant_per = self.find_important(merged_pers, text)

                    #Further filtering of names
                    final_per = []
                    for each in imporatant_per:
                        if len(each.split(" ")) < 2:
                            for name in merged_pers:
                                if each in name and each != name and name not in final_per:
                                    final_per.append(name)
                        elif each not in final_per:
                            final_per.append(each)
                    
                    
                    #May be we can say if there is not final_per value then look for merged_orgs and if they have multiple word in name, we can pick it
                    if len(final_per) < 1:
                        for name in merged_pers:
                            if len(name.split(" ")) > 1:
                                final_per.append(name)
                        
                        #If still empty
                        if len(final_per) < 1 and len(merged_pers) > 0:
                            final_per.append(merged_pers[0])

                    #final_per is good as it will list only names displayed in two letters
                    print(final_per)
                
                    #We can even say first in the final_org will be the startup of the article if that is about startup.
                    print(final_org)

                    final_org_str = ""
                    if len(final_org) > 0:
                        final_org_str = final_org[0]

                        #Further purify
                        final_org_str = self.find_count_word_and_purify_org(merged_orgs, text, final_org_str)
                    
                        
                    self.write_the_result(file, ",".join(merged_pers), ",".join(merged_orgs), ",".join(final_per), final_org_str)
                    filewriterfiles.writerow([file, "|".join(final_per), final_org_str, "|".join(merged_pers), "|".join(merged_orgs)])

    '''
    //problem is most of the time highest weighted name may not be a startup and this cannot be applied with level of accuracy we have with model
    def find_highest_weighted_org(self, items, text):
        relavant = {}
        final_org = ""
        highest = 0
        for item in items:
            mo = re.findall(re.escape(item), text)
            relavant[item] = len(mo)
        for key, value in relavant.items():
            if highest == 0:
                highest = value
                final_org = key
            else:
                if value >= highest:
                    highest = value
                    final_org = key
        
        return final_org
    '''
    
    def find_count_word_and_purify_org(self, items, text, org):
        relavant = {}
        final_org = ""
        for item in items:
            mo = re.findall(re.escape(item), text)
            relavant[item] = len(mo)

        for key, value in relavant.items():
            if value > 3 and "|" not in key and org != key and org in key:
                final_org = key
        if final_org == "":
            final_org = org

        return final_org
    
    def find_important(self, items, text):
        relavant = {}
        most_important = []
        advanced = []
        for item in items:
            mo = re.findall(re.escape(item), text)
            relavant[item] = len(mo)
        for key, value in relavant.items():
            if value > 1 and "|" not in key:
                most_important.append(key)
        return most_important

    def write_the_result(self, file, names, orgs, imp_names, imp_orgs):
        print("Writing {} result to {}_result file in articles_result folder".format(file,file))
        with open(self.process_result+"/"+file+"_result", "w") as f:
            f.write("All names:")
            f.write("\n")
            f.write(names)
            f.write("\n")
            f.write("\n")
            f.write("All organisations:")
            f.write("\n")
            f.write(orgs)
            f.write("\n")
            f.write("\n")
            f.write("Important names:")
            f.write("\n")
            f.write(imp_names)
            f.write("\n")
            f.write("\n")
            f.write("Important Organisations:")
            f.write("\n")
            f.write(imp_orgs)

    def refine_check(self, item, _type):
        check = ["–", ".", "“", "’","\xa0"]
        check_status = True
        if _type == "person" or _type == "org" :
            for each_check in check:
                if each_check in item:
                    check_status = False
        
        return check_status
    
    def remove_duplicates(self, items, _type):
        processed = []
        if _type == "person":
            for item in items:
                if item not in processed:
                    processed.append(item)

        elif _type == "org":
            for item in items:
                if item not in processed:
                    processed.append(item)

        return processed
    

    def further_process(self, items, _type):
        total = []
        merged = []
        if _type == "person":
            for item in items:
                if len(item) >= 2:
                    divide = []
                    for each in item:
                        if "," in each:
                            divide.append(each)
                            total.append(divide)
                            divide = []
                        else:
                            divide.append(each)
                    if len(divide) > 0:
                        total.append(divide)
                        merged.append(" ".join(divide))
                    else:
                        total.append(item)
                        merged.append("".join(item))
                else:
                    total.append(item)
                    merged.append("".join(item))
        elif _type == "org":
            for item in items:
                if len(item) >= 2:
                    divide = []
                    for each in item:
                        if "Image:" in each: 
                            continue
                        elif "," in each:
                            divide.append(each)
                            total.append(divide)
                            divide = []
                        else:
                            divide.append(each)
                    if len(divide) > 0:
                        total.append(divide)
                        merged.append(" ".join(divide))
                    else:
                        total.append(item)
                        merged.append("".join(item))
                else:
                    total.append(item)
                    merged.append("".join(item))

        return total, merged

    def split_sentences(self, text):
        word_raw = text.strip().split(" ")
        words = []
        for word in word_raw:
            word = re.sub(u'\u201c','',word)
            word = re.sub(u'\u201d','',word)
            word = re.sub(u'\u201e','',word)
            word = re.sub(u'\u201f','',word)
            word = word.replace('[','')
            word = word.replace(']','')
            if "." in word:
                list_words = re.split("(\.)", word)
                for each in list_words:
                    if each != "":
                        words.append(each)
            elif "‘" in word:
                list_words = re.split("(\‘)", word)
                for each in list_words:
                    if each != "":
                        words.append(each)

            elif "," in word:
                list_words = re.split("(\,)", word)
                for each in list_words:
                    if each != "":
                        words.append(each)

            elif '”' in word:
                list_words = re.split('(\”)', word)
                for each in list_words:
                    if each != "":
                        words.append(each)

            elif '\n' in word:
                list_words = re.split('(\n)', word)
                for each in list_words:
                    if each != "":
                        words.append(each)

            elif ':' in word:
                list_words = re.split('(\:)', word)
                for each in list_words:
                    if each != "":
                        words.append(each)

            elif ';' in word:
                list_words = re.split('(\;)', word)
                for each in list_words:
                    if each != "":
                        words.append(each)

            elif '?' in word:
                list_words = re.split('(\?)', word)
                for each in list_words:
                    if each != "":
                        words.append(each)

            elif '’' in word:
                list_words = re.split('(\’)', word)
                for each in list_words:
                    if each != "":
                        words.append(each)

            elif '/' in word:
                list_words = re.split('(\/)', word)
                for each in list_words:
                    if each != "":
                        words.append(each)
            elif '"' in word:
                list_words = re.split('(\")', word)
                for each in list_words:
                    if each != "":
                        words.append(each)
            elif '!' in word:
                list_words = re.split('(\!)', word)
                for each in list_words:
                    if each != "":
                        words.append(each)
            else:
               words.append(word) 

        return words

er = EntityRecognition()
er.process()


