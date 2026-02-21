# FairWool AI Hackathon Prototype

This repository contains the prototype implementation for FairWool AI, a wool grading and price suggestion system.

## Project Structure

- `data/`: Directory for dataset (train/val/test splits).
- `train_classifier.py`: PyTorch script to fine-tune a vision transformer for wool grading.
- `generate_price_data.py`: Script to generate synthetic price data.
- `app.py`: Streamlit application for the demo.
- `requirements.txt`: Python dependencies.

## Instructions

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Dataset**:
    Follow the instructions in the Research Pack to collect and label images.
    Organize them into `data/train`, `data/val`, `data/test` with subfolders `A`, `B`, `C`.

3.  **Train Classifier**:
    ```bash
    python train_classifier.py
    ```

4.  **Generate Price Data**:
    ```bash
    python generate_price_data.py
    ```

5.  **Run Application**:
    ```bash
    streamlit run app.py
    ```
