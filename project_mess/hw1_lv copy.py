#!/usr/bin/env python3

import argparse
import pandas as pd
import sklearn as sk
import numpy as np
import math
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
import random

def sign_func(x):
    return np.where(x >= 0, 1, -1)

def encode(frame):
    holder = frame.copy(deep=True)
    encoder = sk.preprocessing.LabelEncoder()
    for feature in holder.columns:
        holder[feature] = encoder.fit_transform(holder[feature])
    return holder

class Perceptron:
    def __init__(self, data, itr, eta):

        self.data = data
        self.itr = itr
        self.eta = eta

        self.accuracy = None
        self.precision = None
        self.recall = None


        targs_df = encode(data.data.targets.copy(deep=True))

        self.one_vs_all = True if len(targs_df.nunique()) > 2 else False

        if self.one_vs_all:
            print("One-vs-all not yet implemented for multi-class (out of time).")
            self.targets = None
            self.features = None
            self.weights = None
            self.bias = None
        else:

            targs_df[targs_df == 0] = -1


            self.targets = targs_df.iloc[:, 0].to_numpy().ravel() 

            feats_df = encode(data.data.features.copy(deep=True))

            scaler = StandardScaler()
            feats_scaled = scaler.fit_transform(feats_df)

            self.features = feats_scaled 

            n_features = self.features.shape[1]
            self.weights = np.zeros(n_features, dtype=float)
            self.bias = 0.0

    def forward(self):
        alpha = self.features @ self.weights + self.bias
        self.fx = sign_func(alpha)

    def backward(self):
        miss = self.targets - self.fx
        w_update = self.features.T @ miss
        b_update = np.sum(miss)

        self.weights += self.eta * w_update
        self.bias += self.eta * b_update
    
    def stats(self):
        tp = np.sum((self.targets == 1) & (self.fx == 1))
        fp = np.sum((self.targets == -1) & (self.fx == 1))
        fn = np.sum((self.targets == 1) & (self.fx == -1))
        tn = np.sum((self.targets == -1) & (self.fx == -1))

        total = len(self.targets)

        self.accuracy = (tp + tn) / total if total > 0 else None
        self.precision = (tp / (tp + fp)) if (tp+fp) > 0 else None
        self.recall = (tp / (tp + fn)) if (tp+fn) > 0 else None

    def model_state(self):
        print("Weights shape:", self.weights)
        print("Bias:", self.bias)
        print("Accuracy:", self.accuracy)
        print("Precision:", self.precision)
        print("Recall:", self.recall)

    def go(self):
        if self.one_vs_all or self.targets is None:
            print("Multi-class not handled. (Out of time to implement).")
            return

        for val in range(self.itr):
            self.forward()
            self.backward()
            self.stats()
            if (val+1) % 10 == 0 or val == 0:
                print(f"\nEpoch {val+1}/{self.itr}")
                self.model_state()

def plot_decision_boundary(targets, feat_2d, weights, bias, names):
    # Separate the data points by class
    pos_idx = (targets == 1)
    neg_idx = (targets == -1)
    
    plt.scatter(feat_2d[pos_idx, 0], feat_2d[pos_idx, 1], label="Class +1", color="blue")
    plt.scatter(feat_2d[neg_idx, 0], feat_2d[neg_idx, 1], label="Class -1", color="red")
    
    # Decision boundary: weights[0]*x + weights[1]*y + bias = 0  =>  y = -(bias + weights[0]*x)/weights[1]
    x_min, x_max = feat_2d[:, 0].min() - 1, feat_2d[:, 0].max() + 1
    x_vals = np.linspace(x_min, x_max, 200)
    
    if abs(weights[1]) > 1e-9:
        y_vals = -(bias + weights[0] * x_vals) / weights[1]
        plt.plot(x_vals, y_vals, 'k--', label="Decision Boundary")
    else:
        x_boundary = -bias / weights[0] if abs(weights[0]) > 1e-9 else 0
        plt.axvline(x_boundary, color="k", linestyle="--", label="Decision Boundary")
    
    plt.xlabel(f"Feature {names[0]}")
    plt.ylabel(f"Feature {names[1]}")
    plt.title("Perceptron Decision Boundar: " + names[2] + ".")
    plt.legend()
    plt.show()

def main(args):
    print(f"Model: {args.model}")
    # Load dataset from ucimlrepo
    match args.model:
        case "iris":
            data = fetch_ucirepo(id=53)
        case "mush":
            data = fetch_ucirepo(id=73)
        case "musk":
            data = fetch_ucirepo(id=74)
        case _:
            raise Exception("Enter: 'iris' / 'mush' / 'musk' only.")

    print(f"Maximum Iterations: {args.iter}")
    print(f"Learning Rate: {args.lr}")

    mod = Perceptron(data, args.iter, args.lr)
    mod.go()

    map_1 = args.map_1
    map_2 = args.map_2
    
    features = data.data.features.columns

    if(map_1 != "rnd" and map_2 != "rnd"):
        index_1 = features.get_loc(map_1)
        index_2 = features.get_loc(map_2)
    else:
        if(map_1 == "rnd"):
            index_1 = random.randint(0, len(features) - 1)
        if(map_2 == "rnd"):
            index_2 = random.randint(0, len(features) - 1)
        while(index_1==index_2):
            index_2 = random.randint(0, len(features) - 1)

    feat_1 = mod.features[:, index_1]
    feat_2 = mod.features[:, index_2]

    feat_2d = np.stack((feat_1, feat_2), axis=1)
    print(feat_2d.shape)

    name_1 = features[index_1]
    name_2 = features[index_2]
    name_3 = args.model
    names = [name_1, name_2, name_3]

    weights = (mod.weights[index_1], mod.weights[index_2])
    print(mod.weights.shape)

    plot_decision_boundary(mod.targets, feat_2d, weights, mod.bias, names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceptron Classifier")
    parser.add_argument("--model", type=str, default="iris", help="['iris','mush','musk']")
    parser.add_argument("--iter", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--map_1", type=str, default="rnd", help="Feature 1 (to map)")
    parser.add_argument("--map_2", type=str, default="rnd", help="Feature 2 (to map)")

    args = parser.parse_args()
    main(args)