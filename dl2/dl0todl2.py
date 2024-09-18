import os
from eventlist_v4 import Eventlist
import argparse

def main():
    parser = argparse.ArgumentParser(description="Process HDF5 files.")
    parser.add_argument("inputdir", help="inputdir")
    args = parser.parse_args()

    inputdir = args.inputdir
    directory = inputdir
    #/data02/gammaflash/DL0/acquisizione_2022_08_03/rpg0
    outputdir = directory.replace("DL0", "DL2")

    # Assicurati di aver definito e importato il modulo eventlist correttamente
    # from qualche_modulo import eventlist
    eventlist = Eventlist()
    for file_name in os.listdir(directory):
        if file_name.endswith('.h5'):
            file_path = os.path.join(directory, file_name)
            eventlist.process_file(file_path, None, outputdir, False)

if __name__ == "__main__":
    main()
