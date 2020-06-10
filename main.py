import sys
import argparse

sys.path.append('/local/home/henrikm/Fakenews_Classification/Preprocessing')
sys.path.append('/local/home/henrikm/Fakenews_Classification/BiLSTM')
sys.path.append('/local/home/henrikm/Fakenews_Classification/CNN')
sys.path.append('/local/home/henrikm/Fakenews_Classification/BiLSTM_Khan')
sys.path.append('/local/home/henrikm/Fakenews_Classification/C-GRU_Liu')
sys.path.append('/local/home/henrikm/Fakenews_Classification/C-LSTM')
sys.path.append('/local/home/henrikm/Fakenews_Classification/C-LSTM_Khan')
sys.path.append('/local/home/henrikm/Fakenews_Classification/CNN_LSTM')
sys.path.append('/local/home/henrikm/Fakenews_Classification/NB')
sys.path.append('/local/home/henrikm/Fakenews_Classification/SVM')
sys.path.append('/local/home/henrikm/Fakenews_Classification/T_Test')
sys.path.append('/local/home/henrikm/Fakenews_Classification/Visualization')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, default=1, metavar='PREPROCESSING |true|false|')
    parser.add_argument("-v", type=str, default=2, metavar='VISUALIZATION |plot|tfidf|timecascade|none|')
    parser.add_argument("-c", type=str, default=3, metavar='CLASSIFIER |bilstm|bilstmkhan|cgruliu|clstm|clstmkhan|cnn|cnnlstm|nb|svm|')       
    parser.add_argument("-s", type=str, default=4, metavar='SAMPLING |none|smote|undersampling|')
    parser.add_argument("-t", type=str, default=5, metavar='T-TEST |t-test|none|')
    args = parser.parse_args()
    if len(sys.argv)==1:
        print('See help flag, -h, for help on usage.')
        exit()
    print('')
    try:
        if args.p=='true':
            print('Running with preprocessing.')
            import Preprocessing1
            import Preprocessing2
            import Preprocessing3
            import Preprocessing4
            import Preprocessing5
            import Preprocessing6
            import Preprocessing7
            import Preprocessing8
            import Preprocessing9
        elif args.p=='false':
            print('Running without preprocessing.')
        else:
            print('Running without preprocessing.')
    except FileNotFoundError:
        print('Must run with preprocessing the first time to generate dataset.')
        exit()
    try:
        if args.v=='plot':
            print('Running with preprocessing.')
            import Plot1
            import Plot2
        elif args.v=='tfidf':
            import TFIDF1
            import TFIDF2
        elif args.v=='timecascade':
            import timecascade
        elif args.p=='none':
            print('Running without visualization.')
        else:
            print('Running without visualization.')
    except IndexError:
        print('See help flag, -h, for usage.')
        exit()
    try:
        if args.c=='bilstm':
            if args.s=='none':
                import BiLSTM
            elif args.s=='smote':
                import BiLSTM_Oversampling
            elif args.s=='undersampling':
                import BiLSTM_Undersampling
            else:
                print('See help flag, -h, for usage.')
                exit()
    except IndexError:
        print('See help flag, -h, for usage.')
        exit()
    try:
        if args.c=='bilstmkhan':
            if args.s=='none':
                import LSTM_Khan
            elif args.s=='smote':
                import LSTM_Khan_smote
            elif args.s=='undersampling':
                import LSTM_Khan_under
            else:
                print('See help flag, -h, for usage.')
                exit()
    except IndexError:
        print('See help flag, -h, for usage.')
        exit()    
    try:
        if args.c=='cgruliu':
            if args.s=='none':
                import Liu
            elif args.s=='smote':
                import Liu_smote
            elif args.s=='undersampling':
                import Liu_under
            else:
                print('See help flag, -h, for usage.')
                exit()
    except IndexError:
        print('See help flag, -h, for usage.')
        exit()
    try:
        if args.c=='clstm':
            if args.s=='none':
                import C_LSTM_no_sampling
            elif args.s=='smote':
                import C_LSTM_smote_grid
            elif args.s=='undersampling':
                import C_LSTM_under_grid
            else:
                print('See help flag, -h, for usage.')
                exit()
    except IndexError:
        print('See help flag, -h, for usage.')
        exit()
    try:
        if args.c=='clstmkhan':
            if args.s=='none':
                import C_LSTM_Khan
            elif args.s=='smote':
                import C_LSTM_Khan_smote
            elif args.s=='undersampling':
                import C_LSTM_Khan_under
            else:
                print('See help flag, -h, for usage.')
                exit()
    except IndexError:
        print('See help flag, -h, for usage.')
        exit()
    try:
        if args.c=='cnn':
            if args.s=='none':
                import CNN_no_sampling
            elif args.s=='smote':
                import CNN_smote_grid
            elif args.s=='undersampling':
                import CNN_under_grid
            else:
                print('See help flag, -h, for usage.')
                exit()
    except IndexError:
        print('See help flag, -h, for usage.')
        exit()
    try:
        if args.c=='cnnlstm':
            if args.s=='none':
                import CNN_LSTM
            elif args.s=='smote':
                import CNN_LSTM_smote
            elif args.s=='undersampling':
                import CNN_LSTM_under
            else:
                print('See help flag, -h, for usage.')
                exit()
    except IndexError:
        print('See help flag, -h, for usage.')
        exit()
    try:
        if args.c=='nb':
            if args.s=='none':
                import NB
            elif args.s=='smote':
                import NB_smote
            elif args.s=='undersampling':
                import NB_under
            else:
                print('See help flag, -h, for usage.')
                exit()
    except IndexError:
        print('See help flag, -h, for usage.')
        exit()
    try:
        if args.c=='svm':
            if args.s=='none':
                import SVM
            elif args.s=='smote':
                import SVM_smote
            elif args.s=='undersampling':
                import SVM_under
            else:
                print('See help flag, -h, for usage.')
                exit()
    except IndexError:
        print('See help flag, -h, for usage.')
        exit()
    try:
        if args.t=='t-test':
            if args.s=='none':
                import T_Test
            elif args.s=='smote':
                import T_Test_smote
            elif args.s=='undersampling':
                import T_Test_under
        elif args.t=='none':
            print('Running without T-Test')
    except IndexError:
        print('See help flag, -h, for usage.')
        exit()
if __name__ == "__main__":
    main()
