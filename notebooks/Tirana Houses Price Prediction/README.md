# 🏠 House Price Predictor - Tirana

Predicts apartment prices in Tirana, Albania using machine learning models.

## 📊 Dataset
- **4,505 listings** with 24 features
- **Variables:** Price, area (m²), rooms, location, amenities, floor level
- **Location:** Tirana, Albania

## 🤖 Models
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Random Forest | €42,441 | €25,323 | 0.8569 |
| Gradient Boosting | €47,478 | €29,676 | 0.8209 |
| XGBoost | €45,614 | €26,968 | 0.8347 |

**Best:** Random Forest (lowest error, highest R²)

## 🔑 Top Features
1. **Area (m²)** - 58% importance
2. **Neighborhood** - 24% importance
3. **Distance to center** - 8% importance

## 📦 Installation
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
jupyter notebook house_price_predictor.ipynb
```

## 📝 Key Finding
**Size matters most.** Area explains 58% of price variance. Location (neighborhood) is second at 24%. Individual amenities have minimal impact (<1%).

## 📁 Files
- `house_price_predictor.ipynb` - Main notebook
- `data_cleaner.py` - Data cleaning
- `extractors.py` - Text extraction utilities