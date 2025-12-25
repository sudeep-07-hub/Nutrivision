# NutriVision
### Indian Food Recognition & Nutrition Estimation using Deep Learning
NutriVision is an end-to-end deep learning system that recognizes Indian dishes from images, estimates portion size, and computes nutritional values such as calories, carbohydrates, protein, and fat.

The project is designed with real-world robustness, including:
* Confidence-based rejection of unknown dishes
* Modular ML pipeline
* Explainable outputs suitable for health and nutrition use cases

---
### Features
* Image-based Indian food classification using ResNet-18
* Confidence-aware prediction barrier for unknown dishes
* Top-3 dish suggestions with confidence scores
* Portion estimation using segmentation + heuristics
* Nutrition calculation using recipe priors and ingredient nutrition
* Train / Validation loss tracking & confusion matrix
* Fully modular pipeline architecture

---
### Project Architecture
```
  NutriVision/
│
├── dataset/
│   ├── train/               # Training images (class-wise)
│   ├── test/                # Validation/Test images
│
├── src/
│   ├── classification/
│   │   ├── model.py         # ResNet-18 model definition
│   │   ├── train.py         # Training + validation logic
│   │   ├── prediction.py   # Inference with confidence barrier
│   │
│   ├── segmentation/
│   │   ├── segment.py       # Food segmentation
│   │   ├── portion.py       # Portion estimation
│   │
│   ├── nutrition/
│   │   ├── recipe_priors.py
│   │   ├── ingredient_weights.py
│   │   ├── ingredient_nutrition.py
│   │   ├── calculate_nutrition.py
│   │
│   ├── pipeline/
│   │   ├── run_pipeline.py  # End-to-end pipeline
│
├── dish_classifier.pth      # Trained model weights
├── requirements.txt
├── README.md
```
---

### Model Details
* Architecture: ResNet-18 (pretrained on ImageNet)
* Transfer Learning Strategy:
  * Early layers frozen
  * Deeper layers + FC layer fine-tuned
* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Input Size: 224 × 224 RGB
* Data Augmentation:
  * Random rotation
  * Horizontal flip
  * Color jitter
  * Random resized crop
---
### Performance
* Final Validation Accuracy: ~93%
* Validation Loss Monitoring: Enabled
* Confusion Matrix: Generated after training
* Overfitting Control:
  * Partial layer freezing
  * Strong data augmentation
  * Validation loss tracking
---
### End-to-End Pipeline Flow
```
Input Image
   ↓
Food Classification
   ↓
Confidence Check
   ├─ Reject (Unknown Dish)
   └─ Accept (Known Dish)
           ↓
Food Segmentation
           ↓
Portion Estimation
           ↓
Ingredient Weight Estimation
           ↓
Nutrition Calculation
```
---
### How to Run
```

pip install -r requirements.txt

```
---
### Train the Model
```

python src/classification/train.py

```
---
### Run Full Pipeline
```

python -m src.pipeline.run_pipeline

```
---
### Input Example
* Food image (JPG / PNG)
* Indian dish (trained or untrained)

### Output Example
For Known Dish
```
Dish: Dosa
Portion: 65%
Estimated Weight: 180 g

Calories: 320 kcal
Carbs: 48 g
Protein: 8 g
Fat: 10 g
```

For Unknown Dish
```
Dish Not Recognized
Confidence too low (42%)

Top Predictions:
- Dosa (41%)
- Idli (32%)
- Uttapam (19%)
```
---
### Dataset
* Curated Indian food image dataset
* Class-wise directory structure
* Balanced across multiple dishes
* Supports future expansion
---





