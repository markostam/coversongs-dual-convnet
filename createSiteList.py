# create root site list to scrape from colin's msd server
# python 3.5

import itertools

# get list of sites
alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
letters = itertools.product(alphabet,repeat=3)
sites = ('http://potbelly.ddns.net:30316/msd/mp3/{}/{}/{}/'.format(letter[0],letter[1],letter[2]) for letter in letters)

# write to file
with open('sitelist.txt', 'w') as sitelist:
    for site in sites:
        sitelist.write('{}\n'.format(site))


