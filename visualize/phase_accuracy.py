import pandas as pd
import matplotlib.pyplot as plt

def visualize_classification_report(csv_data):
    # Read CSV data into DataFrame
    df = pd.read_csv(csv_data)
    
    # Set class column as index for easier plotting
    df.set_index('Class', inplace=True)
    
    # Plot Precision, Recall, and F1-Score for each class
    df[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', figsize=(10, 6))
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score for Each Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    visualize_classification_report("visualize/results/classification_report.csv")