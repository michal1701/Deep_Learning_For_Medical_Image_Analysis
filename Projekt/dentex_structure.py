import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def explore_dentex_dataset():
    """
    Funkcja do eksploracji zestawu danych DENTEX
    """
    # Ścieżka do folderu DENTEX
    dentex_path = Path("DENTEX")
    
    if not dentex_path.exists():
        print(f"Folder {dentex_path} nie istnieje!")
        return
    
    print("=== EKSPLORACJA ZESTAWU DANYCH DENTEX ===\n")
    
    # Wyświetl strukturę folderów
    print("Struktura folderów:")
    for root, dirs, files in os.walk(dentex_path):
        level = root.replace(str(dentex_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Pokaż tylko pierwsze 10 plików
            print(f"{subindent}{file}")
        if len(files) > 10:
            print(f"{subindent}... i {len(files) - 10} więcej plików")
    
    # Znajdź pliki CSV
    csv_files = list(dentex_path.rglob("*.csv"))
    print(f"\nZnalezione pliki CSV: {len(csv_files)}")
    
    for csv_file in csv_files:
        print(f"\n--- Analiza pliku: {csv_file.name} ---")
        try:
            df = pd.read_csv(csv_file)
            print(f"Wymiary: {df.shape}")
            print(f"Kolumny: {list(df.columns)}")
            print(f"\nPierwsze 5 wierszy:")
            print(df.head())
            print(f"\nInformacje o danych:")
            print(df.info())
            print(f"\nStatystyki opisowe:")
            print(df.describe())
        except Exception as e:
            print(f"Błąd podczas wczytywania {csv_file}: {e}")
    
    # Znajdź pliki obrazów
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(dentex_path.rglob(f"*{ext}")))
        image_files.extend(list(dentex_path.rglob(f"*{ext.upper()}")))
    
    print(f"\nZnalezione pliki obrazów: {len(image_files)}")
    if image_files:
        print("Pierwsze 5 plików obrazów:")
        for img in image_files[:5]:
            print(f"  {img.name}")
    
    # Znajdź inne typy plików
    all_files = list(dentex_path.rglob("*.*"))
    file_types = {}
    for file in all_files:
        ext = file.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    
    print(f"\nRozkład typów plików:")
    for ext, count in sorted(file_types.items()):
        print(f"  {ext}: {count} plików")

def load_dentex_data(file_name=None):
    """
    Funkcja do wczytania konkretnego pliku z zestawu DENTEX
    """
    dentex_path = Path("DENTEX")
    
    if file_name:
        file_path = dentex_path / file_name
        if file_path.exists():
            if file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path)
            else:
                print(f"Plik {file_name} nie jest plikiem CSV")
        else:
            print(f"Plik {file_name} nie istnieje")
    else:
        csv_files = list(dentex_path.rglob("*.csv"))
        if csv_files:
            return pd.read_csv(csv_files[0])
        else:
            print("Nie znaleziono plików CSV w folderze DENTEX")
    
    return None

if __name__ == "__main__":
    explore_dentex_dataset()