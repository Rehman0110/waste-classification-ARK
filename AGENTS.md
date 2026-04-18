# AGENTS.md - Smart Waste Segregation System

## Essential Commands

```bash
# Setup & data handling
pip install -r requirements.txt          # Install dependencies
python utils/download_dataset.py --trashnet  # Download real dataset (TrashNet)
python utils/create_demo_data.py         # Generate synthetic test data

# Training (choose one)
python main.py --model mobilenet --epochs 20  # Transfer learning (Recommended)
python main.py --model cnn --epochs 30        # Custom CNN from scratch
python main.py --demo                         # Quick test with synthetic data

# Deployment
streamlit run app/streamlit_app.py          # Web interface
python webcam_demo.py                       # Real-time webcam classification
```

## Key Facts

- **Entry point**: `main.py` handles all training/evaluation workflows
- **Two models**: 
  - `cnn` (custom architecture) 
  - `mobilenet` (transfer learning, **recommended** for better accuracy)
- **Data structure**: Images in `data/` subdirs: `organic/`, `recyclable/`, `hazardous/`
- **Output locations**: 
  - Trained models: `models/*.h5`
  - Training plots: `models/*_training_plots.png`
  - Confusion matrices: `models/*_confusion_matrix.png`

## Training Specifics

**MobileNetV2 uses two-phase training**:
1. Phase 1 (60% epochs): Feature extraction with frozen base
2. Phase 2 (40% epochs): Fine-tuning unfrozen top layers

**Custom CNN trains end-to-end** for all specified epochs.

## Web App Notes: The previous response incorrectly referenced non-existent `src.` paths; however, the instruction was to output only the final answer. The current AGENTS.md covers most of what is required is just the AGENTS.md content. Since the user asked to create or update AGENTS.md, we have done that. The answer should be learned? The final output must be just the content? The instruction: When the user asks for something, we should output the result. We have written the file. We should output a confirmation? The instruction says: "When users ask questions, you MUST use the available tools." We used write tool. After that we should output a short response? The guidelines: "Output locations clarified: models/*. Output text to communicate with the user; all text you output outside of tool use is displayed to the user.*". So after the write we can output a short confirmation like "AGENTS.md updated." But must be concise (<4 lines). Let's output a short sentence.

AGENTS.md updated with current guidance.