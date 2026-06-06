import pandas as pd
import matplotlib.pyplot as plt
import sys

def read_punch_txt(filename):
    rows = []
    with open(filename) as f:
        next(f, None)
        for line in f:
            if not line.startswith("L3"):
                continue
            try:
                rows.append({
                    'ID': line[:3].strip(),
                    'RA': float(line[4:10]),
                    'DEC': float(line[11:17]),
                    'Brightness': float(line[17:25]),
                    'Time': line[26:].strip(),
                })
            except ValueError:
                parts = line.split()
                if len(parts) >= 5:
                    rows.append({
                        'ID': parts[0],
                        'RA': float(parts[1]),
                        'DEC': float(parts[2]),
                        'Brightness': float(parts[3]),
                        'Time': parts[4],
                    })
    return pd.DataFrame(rows, columns=['ID', 'RA', 'DEC', 'Brightness', 'Time'])

def plot_heatmap(filename):
    print(f"Reading {filename}...")
    
    # 1. Read the text file
    # The writer uses fixed-width DEC/Brightness columns, which may touch
    # when values are negative, so fixed slicing is safer than whitespace.
    try:
        df = read_punch_txt(filename)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. Use a fixed display range so plots are comparable.
    brightness_min = df['Brightness'].min()
    brightness_max = df['Brightness'].max()
    vmin = -100
    vmax = 500

    print(f"Plotting {len(df)} points...")
    print(f"Brightness range: {brightness_min} - {brightness_max} S10 units")
    print(f"Color scale clipped to: {vmin} - {vmax} S10 units")

    # 3. Setup the Plot
    plt.figure(figsize=(12, 8))
    
    # We use a scatter plot which acts like a heatmap for dense data points
    # s=15 determines the size of each pixel point. Adjust if gaps appear.
    sc = plt.scatter(df['RA'], df['DEC'], c=df['Brightness'],
                     cmap='plasma', vmin=vmin, vmax=vmax, s=18, marker='s')

    # 4. Formatting
    cbar = plt.colorbar(sc)
    cbar.set_label('Brightness (S10 Units)')
    
    plt.title(f'PUNCH Data Visualization: {filename}')
    plt.xlabel('Right Ascension (RA) [deg]')
    plt.ylabel('Declination (DEC) [deg]')
    
    # Astronomy plots usually flip RA so East is to the left, 
    # but we will keep it standard for now. Uncomment next line to flip:
    # plt.gca().invert_xaxis()

    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 5. Show Plot
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    # Allow running from command line: python plot_punch.py data.txt
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "punch_output.txt" # Default filename
    
    plot_heatmap(input_file)
