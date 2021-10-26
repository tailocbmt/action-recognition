import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logPath", 
                        default='action-recognition\src\R2Plus1D-PyTorch\POSEC3D-LSTM\log.csv',
                        help="Path to the csv log file")
    parser.add_argument("--save",
                        action='store_true', 
                        default=False, 
                        help="Save plot or not")
    return parser.parse_args()

def main(args):
    log = pd.read_csv(args.logPath)
    epochs = range(1,len(log)+1)
    for col in log.columns:
        plt.plot(epochs, log[col], label='{} loss'.format(col))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if args.save:
        plt.savefig('action-recognition\src\R2Plus1D-PyTorch\POSEC3D-LSTM\Loss.png')
    else:
        plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)