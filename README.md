# **QChem-GNN: Quantum Chemistry Graph Neural Network**

*Explainable AI: GAT Layer-2 Attention Weights dynamically prioritizing chemical bonds based on their contribution to thermodynamic stability.*  
This repository contains a production-grade PyTorch Geometric pipeline designed to predict the thermodynamic properties of molecules (specifically Internal Energy ![][image1] at 0K) directly from their 3D atomic structures.

## **Architecture Highlights**

* **Spatial Graph Formulation:** Atoms are modeled as nodes and chemical bonds as edges. Euclidean distances between atoms are expanded into 16 Radial Basis Function (RBF) bins, injecting 3D spatial geometry directly into the network.  
* **Physics-Informed Residuals:** The model bypasses learning raw energies by subtracting the baseline thermodynamic energy of isolated atoms (atomref). The GNN strictly learns the complex interaction energy of the chemical bonds.  
* **Explainable AI (XAI):** Implements Graph Attention Networks (GATConv) to dynamically weight the importance of specific chemical bonds. The repository includes visualization scripts that map these learned attention weights back onto the 2D molecular geometry.

## **Setup & Installation**

Ensure you have PyTorch installed with CUDA support. Then, install the dependencies using the provided requirements.txt:  
pip install \-r requirements.txt

## **Usage**

To execute the entire pipeline (Data ingestion, training, evaluation, and visualization):  
python main.py

*Note: The script automatically downloads the QM9 dataset (\~130k molecules) to ./data/QM9 on its first run.*

## **Outputs & Artifacts**

The pipeline automatically saves model weights as qchem\_gnn\_best.pth to ./checkpoints/ and generates professional metrics and Explainable AI plots to ./plots/.

## **Performance & Metrics**

After training on the QM9 dataset, the model generates the following telemetry and evaluation metrics to validate its chemical accuracy:

### **1\. Test Set Parity Plot**

Comparison of the predicted internal energy (![][image1]) against the true quantum mechanical calculations.

### **2\. Training Dynamics**

The model achieves stable convergence, avoiding severe overfitting across the epochs.

### **3\. Error Analysis**

Error distributions across the test set and segmented by the number of atoms in the molecule.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAaCAYAAACzdqxAAAABS0lEQVR4Xu2UPyxDURTGDwkWwULMRlsHqxpqMWAgFiFSQw1GicFaSSVIDAxGEZtVSAwSIqmBxFL/mpibGLoRwndyTr3TU80VnSTvl/yG73yvN+/d+/qIYmLqsQxv4Af8hPcwDzu0H4AP2vE1j3BCuyC8yDt89oXhDI77YYgxkjva8YXSAouw2RchtkkWHvWFMgj3/fA38L69wnZfKFk444ch+kju9tQXhivY64chMiQLL/lC6YbXbpaCe3AVrsGm6lo4JFk44QtlGuZMboVPsFPzFpyL6ogCLFP9Ez+B/SbzAfN7X2EWHpn8DQ9f/FBZgBtuxlt2YfIUyRPUMATfSPa6Qg/chLtU+yQr8NzkSVgyuYokyVvB23JH8i/jH/zEIrw0me+YX9eGGYG3Js/DY5P/TBvJN6VLM38G0lHdGMPwAK6THK4/h5j/yhcIcDxVztVXsAAAAABJRU5ErkJggg==>