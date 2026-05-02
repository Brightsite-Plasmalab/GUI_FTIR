import sys as sys
from radis.io.hitran import fetch_hitran

def main():
    df = fetch_hitran(sys.argv[1],  cache = 'regen', extra_params='all')

if __name__ == '__main__':
    main()