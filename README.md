# Advanced Bayesian Optimization for Hyperparameter Tuning of Deep Neural Networks

This project implements **Bayesian Optimization (BO)** to tune the hyperparameters of a moderately complex deep neural network (CNN) using the **Expected Improvement (EI)** acquisition function.  
The goal is to efficiently discover near-optimal learning rate, number of filters, batch size, and dense layer size—using far fewer trials than random search or grid search.

Bayesian Optimization is applied on top of a baseline CNN trained on a synthetic multi-class dataset (CIFAR-like).  
All results, logs, and BO traces are included in this repository.

---

## 🚀 Project Structure

advanced-bo-hpo/
│── README.md → Main project documentation
│── bayesian_opt.py → BO driver script
│── multiclass_cnn_baseline.py → Baseline CNN model
│── best_model_baseline.pth → Saved best CNN model
│── bo_results.csv → All BO iterations + hyperparameters
│── validation_scores.csv → Baseline model validation metrics
│── optimization_trace.txt → BO optimization trace log
│── search_space.json → Hyperparameter definitions



---

## 📌 1. Baseline CNN Model

The baseline CNN (in `multiclass_cnn_baseline.py`):

- 3 convolutional layers  
- ReLU activations  
- MaxPooling  
- Flatten + Dense layers  
- Softmax classification  
- Trained for a small number of epochs to simulate expensive objective evaluation  

Output file:
best_model_baseline.pth


---

## 📌 2. Bayesian Optimization Pipeline

BO is implemented using:
- **scikit-optimize (`skopt`)**
- **Gaussian Process surrogate model**
- **Expected Improvement (EI)** acquisition function

Script:


bayesian_opt.py


Each BO iteration:
1. Samples candidate hyperparameters  
2. Trains the CNN for a fixed number of epochs  
3. Evaluates validation accuracy  
4. Updates Gaussian Process  
5. Writes logs + results

Outputs:


bo_results.csv → all hyperparameters + accuracy
optimization_trace.txt → BO sequence log
search_space.json → hyperparameter definitions


---

## 🎯 Hyperparameters Tuned

| Hyperparameter | Range |
|----------------|-------|
| Learning rate  | 1e-5 → 1e-2 |
| Batch size     | 16 → 128 |
| CNN Filters    | 16 → 64 |
| Dense Units    | 64 → 256 |

---

## 📊 Sample Results

### **Baseline Performance**
Loaded from `validation_scores.csv`:

| Metric | Value |
|--------|-------|
| Accuracy | ~0.70 |
| Loss | ~1.05 |

### **After Bayesian Optimization**
Best row in `bo_results.csv`:

| Hyperparameters | Best Accuracy |
|------------------|--------------|
| lr=0.0008, filters=48, dense=128, batch=64 | ~0.82 |

**Improvement: +12% accuracy**

---

## 🧠 Why Bayesian Optimization?

Unlike grid/random search, BO:

- Handles expensive models  
- Learns from previous trials  
- Narrows search efficiently  
- Uses Gaussian Processes to model objective  
- Maximizes Expected Improvement (EI)  

This results in **faster convergence and better accuracy**.

---

## 📥 How to Run

### **Install Requirements**
```bash
pip install torch torchvision scikit-optimize numpy pandas

1. Train Baseline
python multiclass_cnn_baseline.py

2. Run Bayesian Optimization
python bayesian_opt.py

Conclusion

This project demonstrates:

✔ Implementation of a deep CNN
✔ Complete Bayesian Optimization loop
✔ Analysis of acquisition function (EI)
✔ Comparison vs baseline
✔ Final optimized hyperparameters and improved performance
