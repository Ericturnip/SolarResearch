import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_heatmap(filename):
    print(f"Reading {filename}...")
    
    # 1. Read the text file
    # We skip the first row (date header) and use whitespace as the separator
    try:
        df = pd.read_csv(filename, delim_whitespace=True, skiprows=1, 
                         names=['ID', 'RA', 'DEC', 'Brightness'])
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. Filter for Visualization
    # We create a clipped version for plotting so stars don't wash out the image.
    # Typical K-Corona is 10-100 S10. Stars are >1000.
    # We set vmin=0 to ignore negative subtraction artifacts.
    # We set vmax=150 to highlight the solar wind structure.
    vmin = 0
    vmax = 150 

    print(f"Plotting {len(df)} points...")
    print(f"Color scale clipped to: {vmin} - {vmax} S10 units (to hide stars)")

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