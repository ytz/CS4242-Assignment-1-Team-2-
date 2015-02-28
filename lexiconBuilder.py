
import csv

"""
This method combines the two lexicons resource that we found online and 
saves it as a csv file.

Credits - 'Test_Set_3802_Pairs.txt' :
    (1) Fei Liu, Fuliang Weng, Bingqing Wang, Yang Liu. Insertion, Deletion,
    or Substitution? Normalizing Text Messages without Pre-categorization
    nor Supervision. In Proceedings of the 49th Annual Meeting of the 
    Association for Computational Linguistics (ACL 2011), short paper, 
    pages 71-76.

    (2) Fei Liu, Fuliang Weng, Xiao Jiang. A Broad-Coverage Normalization
    System for Social Media Language. In Proceedings of the 50th Annual
    Meeting of the Association for Computational Linguistics (ACL 2012), 
    pages 1035-1044.

Credits - 'emnlp_dict.txt' :
	This dataset is made available under the terms of the Creative Commons Attribution 3.0 Unported 
	licence (http://creativecommons.org/licenses/by/3.0/), with attribution via citation of the 
	following paper:

	@InProceedings{han-cook-baldwin:2012:EMNLP-CoNLL,
	  author    = {Han, Bo  and  Cook, Paul  and  Baldwin, Timothy},
	  title     = {Automatically Constructing a Normalisation Dictionary for Microblogs},
	  booktitle = {Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning},
	  month     = {July},
	  year      = {2012},
	  address   = {Jeju Island, Korea},
	  publisher = {Association for Computational Linguistics},
	  pages     = {421--432},
	  url       = {http://www.aclweb.org/anthology/D12-1039}
"""


def cleanLexicon():
	uniMel_dict = open('emnlp_dict.txt', 'r')
	liufei_dict = open('Test_Set_3802_Pairs.txt','r')
	c = csv.writer(open("lexicon_dict.csv", "wb"))

	for line in liufei_dict:
		clean_line = line.rstrip().split('\t')[1].split(' ')
		c.writerow([clean_line[0],clean_line[2]])
	

	uniMel_dict = open('emnlp_dict.txt', 'r')
	for line in uniMel_dict:
		c.writerow(line.rstrip().split('\t'))

		
	liufei_dict.close()
	uniMel_dict.close()

def main():
	cleanLexicon()

main()
