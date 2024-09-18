import os

# Directory di base in cui cercare le directory DL0
base_directory = "/data02/gammaflash/PL/"

# Comando da eseguire per ogni sottodirectory DL0
dl0_to_dl2_command = "$PYTHONPATH/dl0todl2.py"

while True:
    # Esegui lo script sync.sh
    os.system("sh sync.sh")

    # Trova tutte le directory DL0
    dl0_directories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if d.startswith("DL0") and os.path.isdir(os.path.join(base_directory, d))]

    # Esegui lo script dl0todl2.py per ogni directory DL0 trovata
    for dl0_dir in dl0_directories:
        # Elenco tutte le directory di secondo livello nella directory DL0
        second_level_dirs = [d for d in os.listdir(dl0_dir) if os.path.isdir(os.path.join(dl0_dir, d))]
        for second_level_dir in second_level_dirs:
            # Elenco tutte le directory di terzo livello nella directory di secondo livello
            third_level_dirs = [d for d in os.listdir(os.path.join(dl0_dir, second_level_dir)) if os.path.isdir(os.path.join(dl0_dir, second_level_dir, d))]
            for third_level_dir in third_level_dirs:
                dirpath = os.path.join(dl0_dir, second_level_dir, third_level_dir)

                # Costruisci il percorso completo dello script dl0todl2.py
                script_path = "python3 " + dl0_to_dl2_command + " " + dirpath
                print(script_path)

                # Esegui lo script dl0todl2.py
                os.system(script_path)

