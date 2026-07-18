"""
generate_mapping.py

Creates a default mapping CSV with all symbols from symbols_list.txt
mapped to 'others'. You can then edit the CSV to assign symbols to indices.
"""

from pathlib import Path

SYMBOLS_FILE = Path(r"C:\Exchange\symbols_list.txt")
MAPPING_FILE = Path(r"C:\Exchange\data\index_mapping.csv")

def main():
    if not SYMBOLS_FILE.exists():
        print(f"Symbols list not found: {SYMBOLS_FILE}")
        return

    symbols = SYMBOLS_FILE.read_text().splitlines()
    symbols = [s.strip() for s in symbols if s.strip()]

    with open(MAPPING_FILE, "w") as f:
        f.write("symbol,index\n")
        for sym in symbols:
            f.write(f"{sym},others\n")

    print(f"Default mapping written to {MAPPING_FILE}")
    print(f"Total symbols: {len(symbols)}")

if __name__ == "__main__":
    main()