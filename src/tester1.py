# from utils.rouge_score import evaluate_rouge
#
# src = 'Hello this is Sajad from the CS department. I"ve been living in Urmia city for most of my childhood, but decided to come to the US about a year ago. This was the hardest decision of my life as I had never this experience before. In the US, I"m studying CS now and am so happy to find good people to work with'
#
# # pred = 'I"m Sajad, decided to travel to US a year ago. killing the cats now'
# pred = 'Hello this is Sajad from the CS department. Just random text'
# tgt = 'This is Sajad from Urmia, decided to come to the US around a year ago. Now I"m so happy up here!'
#
# print(evaluate_rouge([pred], [tgt]))
import collections

x = collections.Counter([3,1,2,3,4,5,1,2,3,4,5,6])
# print(sorted(x, key= lambda x:x[0], reverse=True))
print(list(dict(sorted(x.items())).values()))