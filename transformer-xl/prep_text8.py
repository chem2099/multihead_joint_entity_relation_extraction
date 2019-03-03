#!/usr/bin/env python
# coding=utf-8

import os
import sys
import zipfile
from itertools import chain
from io import open

if os.path.exists('train.txt'):
    print('Tokenized text8 already exists - skipping processing')
    sys.exit()

data = zipfile.ZipFile('text8.zip').extractall()
data = open('text8', 'r', encoding='utf-8').read()

print('Length of text8: {}'.format(len(data)))

num_test_chars = 5000000

train_data = data[: -2 * num_test_chars]
valid_data = data[-2 * num_test_chars: -num_test_chars]
test_data = data[-num_test_chars:]

for fn, part in [('train.txt', train_data), ('valid.txt', valid_data), ('test.txt', test_data)]:
    print('{} will have {} bytes'.format(fn, len(part)))
    print('- Tokenizing...')
    # Change space ' ' to underscore '_'
    part_str = ' '.join(['_' if c == ' ' else c for c in part.strip()])
    print('- Writing...')
    f = open(fn, 'w').write(part_str)
    f = open(fn + '.raw', 'w', encoding='utf-8').write(part)
#with open('news_train.tsv', 'r', encoding='utf8') as fin:
    #content = []
    #for line in fin:
        #line = line.strip()
        #if not line :
            #continue
        #content.append(list(line))
    #content = list(chain.from_iterable(content))
    #num_test_chars = len(content) // 10000
    
    #train_data = content[: -2 * num_test_chars]
    #valid_data = content[-2 * num_test_chars: -num_test_chars]
    #test_data = content[-num_test_chars:]
    
    #for fn, part in [('train.txt', train_data), ('valid.txt', valid_data), ('test.txt', test_data)]:
        #print(f'- Writing {fn}...')
        #part_str = ' _'.join(part)
        #f = open(fn, 'w', encoding='utf8').write(part_str)    
    