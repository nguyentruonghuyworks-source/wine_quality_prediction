import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1.DOC VA TIEN XU LY DU LIEU

def load_data(file_path):
    # sep=None de pandas tu nhan dang dau phan cach (file goc dung ';')
    data = pd.read_csv(file_path, sep=None, engine="python")
    return data


def split_train_test(X, y, test_ratio=0.2, seed=42):
    # tron ngau nhien cac dong roi chia train / test
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    n_test = int(n_samples * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


def standardize(X_train, X_test):
    # chuan hoa z-score, chi tinh mean va std tren tap train
    # (tranh bi ro ri thong tin tu tap test)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # tranh chia cho 0
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    # tra ve them mean va std de sau nay chuan hoa du lieu nguoi dung nhap
    return X_train_scaled, X_test_scaled, mean, std

# 2.CAC HAM DANH GIA

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    # tra ve TP, TN, FP, FN (lop duong = good = 1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def evaluate(y_true, y_pred, name):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # them 1e-10 de khong bi chia cho 0
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    print("Ket qua cua", name)
    print("Confusion matrix:")
    print("              Du doan bad | Du doan good")
    print("Thuc te bad  :   %5d     |   %5d" % (tn, fp))
    print("Thuc te good :   %5d     |   %5d" % (fn, tp))
    print("Accuracy  : %.4f" % acc)
    print("Precision : %.4f" % precision)
    print("Recall    : %.4f" % recall)
    print("F1-score  : %.4f" % f1)
    print()
    return acc, precision, recall, f1

# 3.VE BIEU DO BANG MATPLOTLIB

def plot_training_curves(errors_history, loss_history):
    # ve 2 do thi canh nhau va hien thi truc tiep bang plt.show()
    plt.figure(figsize=(11, 4))

    # do thi 1: so mau sai cua perceptron moi epoch
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(errors_history) + 1), errors_history)
    plt.title("Perceptron: so mau sai moi epoch")
    plt.xlabel("Epoch")
    plt.ylabel("So mau sai")
    plt.grid(True)

    # do thi 2: loss cua logistic regression moi iteration
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.title("Logistic Regression: loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()  # hien thi truc tiep, dong cua so bieu do thi chuong trinh chay tiep

# 4.THUAT TOAN PERCEPTRON

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0
        self.errors_history = []  # so mau bi phan loai sai moi epoch

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.errors_history = []

        for epoch in range(self.n_epochs):
            n_errors = 0
            # duyet cac mau theo thu tu ngau nhien moi epoch
            order = np.random.permutation(n_samples)
            for i in order:
                xi = X[i]
                yi = y[i]

                # du doan 0 hoac 1 bang ham buoc nhay
                y_hat = 1 if (np.dot(xi, self.w) + self.b) >= 0 else 0

                # chi cap nhat trong so khi du doan sai
                # (yi - y_hat) = +1 hoac -1 nen w se duoc keo ve dung phia
                if y_hat != yi:
                    update = self.lr * (yi - y_hat)
                    self.w = self.w + update * xi
                    self.b = self.b + update
                    n_errors += 1

            self.errors_history.append(n_errors)

            # neu khong con mau nao sai thi dung som (du lieu tach tuyen tinh)
            if n_errors == 0:
                print("Perceptron hoi tu o epoch", epoch + 1)
                break

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return np.where(z >= 0, 1, 0)

# 5.THUAT TOAN LOGISTIC REGRESSION

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = 0
        self.loss_history = []

    def sigmoid(self, z):
        # clip de tranh bi tran so khi exp(-z) qua lon
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, p):
        # ham mat mat binary cross-entropy
        eps = 1e-10
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_history = []

        # batch gradient descent
        for it in range(self.n_iters):
            z = np.dot(X, self.w) + self.b
            p = self.sigmoid(z)  # xac suat la good

            # dao ham cua loss theo w va b
            dw = np.dot(X.T, (p - y)) / n_samples
            db = np.sum(p - y) / n_samples

            # cap nhat tham so
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

            loss = self.compute_loss(y, p)
            self.loss_history.append(loss)

            if (it + 1) % 200 == 0:
                print("Logistic Regression - iter %d, loss = %.4f" % (it + 1, loss))

    def predict_proba(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        return np.where(p >= threshold, 1, 0)

# 6.NHAP LIEU VA DU DOAN

def input_wine_features(feature_names, X_raw):
    # nhap gia tri cho tung thuoc tinh tu ban phim
    # hien thi khoang gia tri (min - max) va trung binh trong du lieu de de nhap
    min_vals = X_raw.min(axis=0)
    max_vals = X_raw.max(axis=0)
    mean_vals = X_raw.mean(axis=0)

    values = []
    for i, name in enumerate(feature_names):
        while True:
            text = input("Nhap %s [%.3f - %.3f, tb = %.3f] (Enter = lay tb): "
                         % (name, min_vals[i], max_vals[i], mean_vals[i]))
            text = text.strip().replace(",", ".")

            # neu de trong thi lay gia tri trung binh
            if text == "":
                values.append(mean_vals[i])
                break

            try:
                value = float(text)
            except ValueError:
                print("  Gia tri khong hop le, hay nhap lai mot so!")
                continue

            if value < 0:
                print("  Gia tri khong duoc am, hay nhap lai!")
                continue

            # canh bao neu nam ngoai khoang trong du lieu nhung van cho dung
            if value < min_vals[i] or value > max_vals[i]:
                print("  Luu y: gia tri nam ngoai khoang du lieu goc, du doan co the kem chinh xac")
            values.append(value)
            break

    return np.array(values)


def predict_new_wine(x_new, mean, std, perceptron, log_reg):
    # chuan hoa mau moi bang mean va std cua tap train roi moi dua vao mo hinh
    x_scaled = ((x_new - mean) / std).reshape(1, -1)

    pred_per = perceptron.predict(x_scaled)[0]
    pred_log = log_reg.predict(x_scaled)[0]
    prob_good = log_reg.predict_proba(x_scaled)[0]

    label_text = {1: "GOOD (chat luong tot)", 0: "BAD (chat luong kem)"}

    print()
    print("----- Ket qua du doan -----")
    print("Perceptron          :", label_text[pred_per])
    print("Logistic Regression :", label_text[pred_log])
    print("  Xac suat good = %.2f%%, bad = %.2f%%" % (prob_good * 100, (1 - prob_good) * 100))
    print()
    
