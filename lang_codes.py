z= """
- en - English
- ar - Arabic
- zh - Chinese
- fr - French
- de - German
- hi - Hindi
- id - Indonesian
- ga - Irish
- it - Italian
- ja - Japanese
- ko - Korean
- pl - Polish
- pt - Portuguese
- ru - Russian
- es - Spanish
- tr - Turkish
"""
lang_codes = {o.split('-')[1].strip(): o.split('-')[2].strip() for o in filter(lambda x:x,z.split('\n'))}
