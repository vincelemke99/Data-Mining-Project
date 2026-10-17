import pandas as pd
import numpy as np
import re

def impute_manufacturer(df):
    # Erweiterte WMI-Zuordnung mit zusätzlichen Herstellern, ohne "International", mit Genesis WMIs
    wmi_mapping = {
        # Nordamerikanische Hersteller
        '1G': 'General Motors',   # USA
        '2G': 'General Motors',   # Kanada
        '3G': 'General Motors',   # Mexiko
        '1G6': 'Cadillac',        # USA
        '1GC': 'Chevrolet',       # USA (LKW/SUV)
        '5GR': 'Hummer',          # USA (General Motors)
        '5GT': 'Hummer',          # USA (General Motors)
        '5GA': 'Buick',           # USA (General Motors)
        '1F': 'Ford',             # USA
        '2F': 'Ford',             # Kanada
        '3F': 'Ford',             # Mexiko
        '1C': 'Chrysler',         # USA
        '2C': 'Chrysler',         # Kanada
        '3C': 'Chrysler',         # Mexiko
        '1D': 'Dodge',            # USA
        '3D': 'Dodge',            # Mexiko
        'WDY': 'Dodge',           # USA (Sprinter)
        '1ZV': 'Ford',            # USA (ältere Modelle, z. B. Mustang)
        '1J': 'Jeep',             # USA
        '1L': 'Lincoln',          # USA
        '5YJ': 'Tesla',           # USA
        '7FC': 'Rivian',          # USA
        '1R': 'Ram',              # USA
        '3R': 'Ram',              # Mexiko
        '1P': 'Plymouth',         # USA
        '1A8': 'Chrysler',        # USA
        '2A8': 'Chrysler',        # Kanada
        '5B4': 'Workhorse',       # USA (Nutzfahrzeuge)
        '1BA': 'Blue Bird',       # USA (Schulbusse)
        '54D': 'Thomas Built',    # USA (Busse)
        '131': 'Pierce',          # USA (Feuerwehrfahrzeuge)
        '5PV': 'Hino',            # USA (Nutzfahrzeuge)
        '4UZ': 'Spartan Motors',  # USA (Spezialfahrzeuge)
        '1XP': 'Peterbilt',       # USA (LKW)
        '2NP': 'Peterbilt',       # Kanada (LKW)
        '4DR': 'IC Bus',          # USA (Schulbusse)
        '2NK': 'Kenworth',        # Kanada (LKW)
        '4P1': 'Collins Bus',     # USA (Kleinbusse)
        '4YD': 'Wells Cargo',     # USA (Anhänger)
        '3AK': 'Freightliner',    # Mexiko (LKW)
        '57W': 'Western Star',    # USA (LKW)

        # Japanische Hersteller
        'JT': 'Toyota',           # Japan
        '5T': 'Toyota',           # USA
        '4T': 'Toyota',           # USA
        '2T': 'Toyota',           # Kanada
        'ZN6': 'Toyota',          # Japan (Scion/Subaru FR-S/BRZ)
        'JTL': 'Lexus',           # Japan
        '5L': 'Lexus',            # USA
        'JH': 'Honda',            # Japan
        '1H': 'Honda',            # USA
        '2H': 'Honda',            # Kanada
        '5F': 'Honda',            # USA
        '5J6': 'Honda',           # USA (spezifisch)
        '5J8': 'Honda',           # USA (spezifisch)
        'SHH': 'Honda',           # Japan (spezifisch)
        '19X': 'Acura',           # USA
        'JH4': 'Acura',           # Japan
        '19U': 'Acura',           # USA (spezifisch)
        'JN': 'Nissan',           # Japan
        '1N': 'Nissan',           # USA
        '5N': 'Nissan',           # USA
        '3N': 'Nissan',           # Mexiko
        'JNK': 'Infiniti',        # Japan
        '5N1': 'Infiniti',        # USA
        'JM': 'Mazda',            # Japan
        '1M': 'Mazda',            # USA
        '4M': 'Mazda',            # USA
        '3MY': 'Mazda',           # Mexiko
        '1YV': 'Mazda',           # USA
        '3MZ': 'Mazda',           # Mexiko
        'JA': 'Mitsubishi',       # Japan
        '4A': 'Mitsubishi',       # USA
        'JF': 'Subaru',           # Japan
        '5F': 'Subaru',           # USA
        'SNA': 'Subaru',          # Japan
        '4S3': 'Subaru',          # USA (ältere Modelle)
        '4S4': 'Subaru',          # USA (ältere Modelle)
        '4S2': 'Subaru',          # USA (ältere Modelle)
        'JS': 'Suzuki',           # Japan
        '2S': 'Suzuki',           # Kanada
        'JYA': 'Yamaha',          # Japan (Motorräder)
        'JKA': 'Kawasaki',        # Japan (Motorräder)
        'JKB': 'Kawasaki',        # Japan (Motorräder)
        'MH3': 'Yamaha',          # Japan (Motorräder)

        # Koreanische Hersteller
        'KM': 'Hyundai',          # Südkorea
        '5N': 'Hyundai',          # USA
        'KN': 'Kia',              # Südkorea
        '5X': 'Kia',              # USA
        'KND': 'Kia',             # USA (spezifisch)
        'KMH': 'Genesis',         # Südkorea (Genesis-Fahrzeuge, früher Hyundai Genesis)
        'KM8': 'Genesis',         # Südkorea (SUVs wie GV70/GV80)
        '5NM': 'Genesis',         # USA (Hyundai Motor Manufacturing Alabama)
        'KMT': 'Genesis',         # Südkorea (spezifisch für Genesis)

        # Europäische Hersteller
        'WV': 'Volkswagen',       # Deutschland
        '3V': 'Volkswagen',       # Mexiko
        '1VW': 'Volkswagen',      # USA
        'WB': 'BMW',              # Deutschland
        '5U': 'BMW',              # USA
        'WA': 'Audi',             # Deutschland
        'TR': 'Audi',             # Ungarn
        'WF': 'Mercedes-Benz',    # Deutschland
        '4J': 'Mercedes-Benz',    # USA
        'W1K': 'Mercedes-Benz',   # Deutschland (spezifisch)
        'WDD': 'Mercedes-Benz',   # Deutschland (spezifisch)
        'WDB': 'Mercedes-Benz',   # Deutschland (ältere Modelle)
        'WDC': 'Mercedes-Benz',   # Deutschland (SUVs)
        'SAJ': 'Jaguar',          # UK
        'SJA': 'Jaguar',          # UK (ältere Modelle)
        'SAD': 'Jaguar',          # UK (ältere Modelle)
        'SAL': 'Land Rover',      # UK
        'WP': 'Porsche',          # Deutschland
        'YF': 'Volvo',            # Schweden
        '4V': 'Volvo',            # USA
        '7JR': 'Polestar',        # Schweden/China
        'ZF': 'Ferrari',          # Italien
        'ZHW': 'Lamborghini',     # Italien
        'ZA': 'Maserati',         # Italien
        'SC': 'Lotus',            # UK
        '523': 'Aston Martin',    # UK
        'SBM': 'McLaren',         # UK
        'WM': 'smart',            # Deutschland
        'VR': 'Alfa Romeo',       # Italien
        'ZAR': 'Alfa Romeo',      # Italien (spezifisch)
        'VF': 'Peugeot',          # Frankreich
        'VR': 'Renault',          # Frankreich
        'VN': 'Renault',          # Frankreich
        'UU': 'Dacia',            # Rumänien
        'YS': 'Saab',             # Schweden
        'W0': 'Opel',             # Deutschland
        'NM0': 'Mini',            # UK/Deutschland
        'GHN': 'MG',              # UK (ältere Modelle)
        'V12': 'Ducati',          # Italien (Motorräder)

        # Andere
        'L6': 'Geely',            # China
        'LJ': 'BYD',              # China
        '7S': 'Lucid',            # USA
        'LFM': 'Chery',           # China
        'YV': 'VinFast',          # Vietnam
        'KL': 'Daewoo',           # Südkorea
        'VBK': 'KTM',             # Österreich (Motorräder)
        'SMT': 'Triumph',         # UK (Motorräder)
    }

    # Liste ungültiger WMIs (Datenfehler)
    invalid_wmis = {'000', 'XXX', '123', 'NEW', '090', '338', '508', 'TES'}

    def get_manufacturer_from_vin(vin, model=None):
        if pd.isna(vin) or len(str(vin)) < 3:
            return np.nan
        vin = str(vin).upper()
        wmi_3 = vin[:3]
        wmi_2 = vin[:2]

        # Markiere ungültige WMIs
        if wmi_3 in invalid_wmis:
            return np.nan

        # Spezielle Validierung für ZN6 (Toyota/Subaru) vs. Maserati
        if wmi_3 == 'ZN6' and isinstance(model, str):
            if 'maserati' in model.lower():
                return 'Maserati'
        
        # Prüfe zunächst spezifische WMIs mit drei Zeichen
        if wmi_3 in wmi_mapping:
            return wmi_mapping[wmi_3]
        # Dann prüfe die ersten zwei Zeichen
        if wmi_2 in wmi_mapping:
            return wmi_mapping[wmi_2]
        return np.nan

    def get_manufacturer_from_description(description, model=None):
        if pd.isna(description):
            return np.nan
        description = str(description).lower()
        model_str = str(model).lower() if pd.notna(model) else ""
        
        for manufacturer in wmi_mapping.values():
            # Überspringe "International" explizit
            if manufacturer == 'International':
                continue
            # Verwende Wortgrenzen für präzise Suche
            manufacturer_lower = manufacturer.lower()
            pattern = rf'\b{re.escape(manufacturer_lower)}\b'
            if re.search(pattern, description):
                # Validierung mit Modellspalte, falls vorhanden
                if model_str:
                    if manufacturer_lower in model_str:
                        return manufacturer
                    # Spezielle Zuordnungen für bekannte Marken
                    if 'scion' in model_str and manufacturer == 'Toyota':
                        return manufacturer
                    if 'freightliner' in model_str and manufacturer != 'Freightliner':
                        continue
                    if 'duramax' in model_str and manufacturer not in ['Chevrolet', 'General Motors']:
                        continue
                return manufacturer
        return np.nan

    # Gib die Spaltennamen aus, um sicherzustellen, dass 'VIN' und 'description' existieren
    print("Spaltennamen im Datensatz:", df.columns.tolist())

    # Prüfe, ob die Spalten 'VIN' und 'description' existieren
    if 'VIN' not in df.columns:
        print("Fehler: Die Spalte 'VIN' wurde nicht gefunden. Bitte überprüfe die Spaltennamen.")
        raise KeyError("Spalte 'VIN' nicht gefunden.")
    if 'description' not in df.columns:
        print("Fehler: Die Spalte 'description' wurde nicht gefunden. Bitte überprüfe die Spaltennamen.")
        raise KeyError("Spalte 'description' nicht gefunden.")

    # Speichere ursprüngliche Hersteller-Werte, um imputierte Zeilen zu identifizieren
    df['manufacturer_original'] = df['manufacturer'].copy()
    df['imputation_method'] = np.nan  # Neue Spalte, um die Imputationsmethode zu speichern

    # Anzahl der fehlenden Hersteller vor der Imputation
    missing_before = df['manufacturer'].isna().sum()
    print(f"Fehlende Hersteller vor der Imputation: {missing_before}")

    # Schritt 1: Imputiere fehlende Hersteller basierend auf VIN und validiere mit model
    df['manufacturer'] = df.apply(
        lambda row: get_manufacturer_from_vin(row['VIN'], row.get('model')) if pd.isna(row['manufacturer']) else row['manufacturer'],
        axis=1
    )
    df.loc[(df['manufacturer_original'].isna()) & (df['manufacturer'].notna()), 'imputation_method'] = 'VIN'

    # Anzahl der fehlenden Hersteller nach der VIN-Imputation
    missing_after_vin = df['manufacturer'].isna().sum()
    filled_values_vin = missing_before - missing_after_vin
    print(f"Fehlende Hersteller nach VIN-Imputation: {missing_after_vin}")
    print(f"Anzahl der imputierten Hersteller durch VIN: {filled_values_vin}")

    # Schritt 2: Imputiere verbleibende fehlende Hersteller basierend auf description
    df['manufacturer'] = df.apply(
        lambda row: get_manufacturer_from_description(row['description'], row.get('model')) if pd.isna(row['manufacturer']) else row['manufacturer'],
        axis=1
    )
    df.loc[(df['manufacturer_original'].isna()) & (df['manufacturer'].notna()) & (df['imputation_method'].isna()), 'imputation_method'] = 'Description'

    # Anzahl der fehlenden Hersteller nach der Description-Imputation
    missing_after_desc = df['manufacturer'].isna().sum()
    filled_values_desc = missing_after_vin - missing_after_desc
    print(f"Fehlende Hersteller nach Description-Imputation: {missing_after_desc}")
    print(f"Anzahl der imputierten Hersteller durch Description: {filled_values_desc}")

    # Anzahl der Zeilen, wo VIN vorhanden ist, aber Hersteller nicht abgeleitet wurde
    non_imputed_with_vin = df[df['VIN'].notna() & df['manufacturer'].isna()].shape[0]
    print(f"Anzahl der Zeilen mit vorhandener VIN, aber nicht abgeleitetem Hersteller: {non_imputed_with_vin}")

    # Gib die Anzahl der unbekannten WMIs aus (ohne detaillierte Liste)
    if non_imputed_with_vin > 0:
        unknown_wmi_count = df[df['VIN'].notna() & df['manufacturer'].isna()]['VIN'].str[:3].nunique()
        print(f"\nAnzahl unterschiedlicher unbekannter WMIs: {unknown_wmi_count}")
    else:
        print("\nKeine unbekannten WMIs vorhanden.")

    # Identifiziere Zeilen, in denen der Hersteller imputiert wurde (entweder durch VIN oder Description)
    imputed_rows = df[
        (df['manufacturer_original'].isna()) & (df['manufacturer'].notna())
    ]

    # Identifiziere Zeilen, in denen der Hersteller durch VIN imputiert wurde
    imputed_vin_rows = imputed_rows[imputed_rows['imputation_method'] == 'VIN']

    # Identifiziere Zeilen, in denen der Hersteller durch Description imputiert wurde
    imputed_desc_rows = imputed_rows[imputed_rows['imputation_method'] == 'Description']

    # Wähle relevante Spalten für die Ausgabe
    display_columns = ['VIN', 'manufacturer', 'model']

    # Gib bis zu 30 imputierte Zeilen aus (gesamt)
    print("\nBis zu 30 Zeilen, in denen der Hersteller imputiert wurde (gesamt):")
    if not imputed_rows.empty:
        print(imputed_rows[display_columns].head(30))
    else:
        print("Keine Zeilen wurden imputiert (z. B. wegen fehlender gültiger VINs oder Beschreibungen).")

    # Gib bis zu 20 Zeilen aus, in denen der Hersteller durch Description imputiert wurde
    print("\nBis zu 20 Zeilen, in denen der Hersteller durch Description imputiert wurde:")
    if not imputed_desc_rows.empty:
        print(imputed_desc_rows[display_columns].head(20))
    else:
        print("Keine Zeilen wurden durch Description imputiert.")

    # Schritt 3: Imputiere verbleibende fehlende Hersteller basierend auf Model
    # Erstelle ein Mapping von Model -> Manufacturer basierend auf Zeilen, wo beide vorhanden sind
    model_manufacturer_counts = df[df['manufacturer'].notna() & df['model'].notna()][['model', 'manufacturer']].value_counts().reset_index(name='count')

    # Filtere Mappings, die mindestens 3-mal vorkommen
    reliable_mappings = model_manufacturer_counts[model_manufacturer_counts['count'] >= 3][['model', 'manufacturer']]

    # Erstelle ein Dictionary für das Mapping
    model_to_manufacturer = reliable_mappings.set_index('model')['manufacturer'].to_dict()

    # Imputiere fehlende Hersteller basierend auf dem Model
    df['manufacturer_imputed'] = df.apply(
        lambda row: model_to_manufacturer.get(row['model'], row['manufacturer']) if pd.isna(row['manufacturer']) and row.get('model') in model_to_manufacturer else row['manufacturer'],
        axis=1
    )

    # Markiere Zeilen, die durch Model imputiert wurden
    df.loc[(df['manufacturer_original'].isna()) & (df['manufacturer'].notna()) & (df['imputation_method'].isna()), 'imputation_method'] = 'Model'

    # Anzahl der fehlenden Hersteller nach der Model-Imputation
    missing_after_model = df['manufacturer'].isna().sum()
    filled_values_model = missing_after_desc - missing_after_model
    print(f"\nFehlende Hersteller nach Model-Imputation: {missing_after_model}")
    print(f"Anzahl der imputierten Hersteller durch Model: {filled_values_model}")

    # Identifiziere Zeilen, in denen der Hersteller durch Model imputiert wurde
    imputed_model_rows = df[df['imputation_method'] == 'Model'][['model', 'manufacturer']]

    # Gib bis zu 20 Zeilen aus, in denen der Hersteller durch Model imputiert wurde
    print("\nBis zu 20 Zeilen, in denen der Hersteller durch Model imputiert wurde:")
    if not imputed_model_rows.empty:
        print(imputed_model_rows.head(20))
    else:
        print("Keine Zeilen wurden durch Model imputiert.")

    # Gib die Anzahl der Zeilen aus, die noch keinen Hersteller haben
    print(f"\nAnzahl der Zeilen ohne Hersteller nach allen Imputationen: {missing_after_model}")

    # Entferne temporäre Spalten
    df = df.drop(columns=['manufacturer_original', 'imputation_method'])

    #  DROP missing values and return transformed column 
    return df[["id", "manufacturer_imputed"]].dropna()

#print(impute_manufacturer(pd.read_csv("../data/vehicles.csv")).head())